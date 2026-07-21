"""Tests for ``cothis.session`` startup archival wiring (#86).

Covers the three acceptance criteria of #86:

- **Startup trigger**: ``Session.new`` / ``Session.load`` run
  ``run_archival_pass`` once per process per 24h (verified via
  ``archive_state.last_run``).
- **Cold read in place**: ``Session.load`` of an archived session
  rebuilds ``messages`` from the cold DB via ATTACH; no rows are
  copied to hot until the next write.
- **Promote-on-first-write**: the first ``append_message`` after a
  cold load moves the rows cold→hot atomically with
  ``updated_at = now``, and drops the archive index entry.

Tests are offline (no LLM, no network).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cothis.session import Session
from cothis.session.archive import (
    ArchiveIndex,
    archive_session,
)
from cothis.session.storage import Storage

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path
    from typing import Any


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _user_text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _seed_session(
    db_path: Path, cwd: Path, *, model: str = "m", texts: list[str]
) -> str:
    s = Session.new(db_path, cwd=cwd, model=model, flush_sync=True)
    sid = s.session_id
    for i, t in enumerate(texts):
        role = "user" if i % 2 == 0 else "assistant"
        s.append_message(role, [_user_text(t)])
    s.close()
    return sid


def _archive_state_last_run(db_path: Path) -> str | None:
    """Return ``archive_state.last_run`` or ``None`` if missing."""
    import sqlite3

    if not db_path.is_file():
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT value FROM archive_state WHERE key='last_run'"
        ).fetchone()
        return row[0] if row is not None else None
    finally:
        conn.close()


def _set_updated_at(db_path: Path, sid: str, updated_at: str) -> None:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE sessions SET updated_at=? WHERE id=?", (updated_at, sid)
        )
        conn.commit()
    finally:
        conn.close()


def _clear_archive_state(db_path: Path) -> None:
    """Drop ``archive_state.last_run`` so the next pass isn't throttled.

    ``Session.new`` / ``Session.load`` run the startup archival pass
    (#86), which stamps ``last_run``. Tests that drive archival
    directly need a clean slate or the 24h throttle trips.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM archive_state WHERE key='last_run'")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Startup trigger (run_archival_pass at Session.new / Session.load)
# ---------------------------------------------------------------------


def test_startup_archival_pass_runs_on_session_new(tmp_path: Path) -> None:
    """Session.new stamps ``archive_state.last_run`` once."""
    db_path = tmp_path / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    try:
        assert _archive_state_last_run(db_path) is not None
    finally:
        s.close()


def test_startup_archival_pass_runs_on_session_load(tmp_path: Path) -> None:
    """Session.load stamps ``archive_state.last_run`` once."""
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["hi"])

    # Clear the last_run set by _seed_session's Session.new call so we
    # can observe Session.load setting it.
    _clear_archive_state(db_path)

    s = Session.load(db_path, sid, flush_sync=True)
    try:
        assert _archive_state_last_run(db_path) is not None
    finally:
        s.close()


def test_startup_archival_pass_actually_archives_idle_session(
    tmp_path: Path,
) -> None:
    """Session.new's startup pass moves idle sessions to the cold DB."""
    db_path = tmp_path / "session.db"
    old_sid = _seed_session(db_path, tmp_path, texts=["old"])
    _set_updated_at(db_path, old_sid, "2026-04-13T00:00:00+00:00")

    # Clear ``last_run`` set by _seed_session's Session.new so the next
    # Session.new actually runs the pass instead of throttling out.
    _clear_archive_state(db_path)

    # New Session triggers the pass; the 90-day-idle session moves out.
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    try:
        hot = Storage(db_path)
        try:
            assert hot.load_session(old_sid) is None
        finally:
            hot.close()
        idx = ArchiveIndex(db_path.parent / "archive" / "index.json")
        assert idx.get(old_sid) is not None
    finally:
        s.close()


# ---------------------------------------------------------------------
# Cold read in place (Session.load of an archived session)
# ---------------------------------------------------------------------


def test_load_archived_session_reads_messages_from_cold_db(
    tmp_path: Path,
) -> None:
    """Hot miss + archive-index hit → rebuild messages from cold DB.

    Verifies no copy back to hot: after ``Session.load`` the hot DB
    still has zero rows for the session, and the archive index entry
    is intact.
    """
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["alpha", "beta", "gamma"])
    archive_dir = db_path.parent / "archive"
    index = ArchiveIndex(archive_dir / "index.json")

    # Move the session to cold; hot now has zero rows for it.
    archive_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        archive_db_name="2026-07.db",
        archived_at="2026-07-20T00:00:00+00:00", index=index,
    )

    hot = Storage(db_path)
    try:
        assert hot.load_session(sid) is None
    finally:
        hot.close()

    s = Session.load(db_path, sid, flush_sync=True)
    try:
        # The 3 texts (alternating user/assistant/user) survived the
        # cold read.
        assert len(s.messages) == 3
        assert s.messages[0]["content"][0]["text"] == "alpha"
        # Flag is set: first write will promote.
        assert s._cold is True
    finally:
        s.close()

    # Hot still has zero rows — load did not copy back.
    hot = Storage(db_path)
    try:
        assert hot.load_session(sid) is None
    finally:
        hot.close()
    # Index entry intact until the first write promotes.
    assert index.get(sid) is not None


def test_load_archived_session_missing_in_cold_db_drops_stale_index(
    tmp_path: Path,
) -> None:
    """Index entry pointing at a missing cold row is dropped + KeyError."""
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["one"])
    archive_dir = db_path.parent / "archive"
    index = ArchiveIndex(archive_dir / "index.json")

    archive_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        archive_db_name="2026-07.db",
        archived_at="2026-07-20T00:00:00+00:00", index=index,
    )

    # Tamper: delete the cold DB file. The index still points at it.
    (archive_dir / "2026-07.db").unlink()

    with pytest.raises(KeyError):
        Session.load(db_path, sid, flush_sync=True)

    # Stale entry removed (re-read from disk — Session.load mutated
    # its own in-memory index, not the test's variable).
    reloaded = ArchiveIndex(archive_dir / "index.json")
    assert reloaded.get(sid) is None


# ---------------------------------------------------------------------
# Promote-on-first-write
# ---------------------------------------------------------------------


def test_first_write_after_cold_load_promotes_session(tmp_path: Path) -> None:
    """``append_message`` after a cold load moves the rows back to hot."""
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["alpha", "beta"])
    archive_dir = db_path.parent / "archive"
    index = ArchiveIndex(archive_dir / "index.json")

    archive_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        archive_db_name="2026-07.db",
        archived_at="2026-07-20T00:00:00+00:00", index=index,
    )

    s = Session.load(db_path, sid, flush_sync=True)
    try:
        assert s._cold is True
        # First write — promote fires.
        s.append_message("user", [_user_text("new turn")])
        assert s._cold is False
    finally:
        s.close()

    # Hot DB now has the session row + old blocks + the new block.
    hot = Storage(db_path)
    try:
        sr = hot.load_session(sid)
        assert sr is not None
        blocks = hot.load_blocks(sid)
        # 2-seed session = 2 blocks (user + assistant); append_message
        # adds 1 more block for "new turn" → 3 total.
        assert len(blocks) == 3
    finally:
        hot.close()

    # Index entry dropped (re-read from disk — promote mutated its
    # own in-memory index, not the test's variable).
    reloaded = ArchiveIndex(archive_dir / "index.json")
    assert reloaded.get(sid) is None


def test_promote_bumps_updated_at_to_now(tmp_path: Path) -> None:
    """Promote overwrites ``updated_at`` so the session isn't re-archived."""
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["seed"])
    archive_dir = db_path.parent / "archive"
    index = ArchiveIndex(archive_dir / "index.json")

    # Force the pre-archive updated_at into the deep past so we can
    # detect that promote moved it forward.
    _set_updated_at(db_path, sid, "2026-04-13T00:00:00+00:00")
    archive_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        archive_db_name="2026-04.db",
        archived_at="2026-07-20T00:00:00+00:00", index=index,
    )

    s = Session.load(db_path, sid, flush_sync=True)
    try:
        s.append_message("user", [_user_text("post-archive write")])
    finally:
        s.close()

    hot = Storage(db_path)
    try:
        sr = hot.load_session(sid)
        assert sr is not None
        # The promoted updated_at must be later than the archived-at
        # boundary (2026-07-20). i.e. recent, not the stale 2026-04-13.
        assert sr.updated_at > "2026-07-20T00:00:00+00:00"
    finally:
        hot.close()


def test_drain_one_survives_promote_failure(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``_drain_one`` logs + drops rows on promote_session failure (#144).

    The cold-promote block is isolated: a raise from ``promote_session``
    (e.g. KeyError when the cold DB row was hand-deleted) drops the
    current batch with a ``CRITICAL`` log and the consumer stays alive
    for subsequent writes.
    """
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["archived"])
    archive_dir = db_path.parent / "archive"
    index = ArchiveIndex(archive_dir / "index.json")
    archive_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        archive_db_name="2026-07.db",
        archived_at="2026-07-20T00:00:00+00:00", index=index,
    )

    # Load the session from cold (sets _cold=True).
    s = Session.load(db_path, sid, flush_sync=True)
    try:
        assert s._cold is True

        # Patch promote_session to raise — simulates a corrupt cold DB.
        # ``promote_session`` is imported by name into ``cothis.session``
        # (``__init__.py``), so patch at that level, not in ``archive``.
        def _failing_promote(**kwargs: object) -> None:
            raise KeyError("session not in cold DB")

        monkeypatch.setattr(
            "cothis.session.promote_session", _failing_promote
        )

        # First write: promote fails. The row is dropped (not written
        # to hot), but the call doesn't raise.
        s.append_message("user", [_user_text("after-promote-failure")])

        # _cold is still True — promote didn't succeed.
        assert s._cold is True

        # Second write: restore promote_session. The consumer is still
        # alive — this write drains normally.
        monkeypatch.undo()

        s.append_message("user", [_user_text("second-write")])
        # _cold cleared on the successful promote.
        assert s._cold is False
    finally:
        s.close()

    # Hot DB now has the second write (promote succeeded on retry).
    hot = Storage(db_path)
    try:
        blocks = hot.load_blocks(sid)
        texts = [b.content for b in blocks if b.content]
        assert "second-write" in texts
    finally:
        hot.close()
