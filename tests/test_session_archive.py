"""Tests for ``cothis.session.archive`` — cold/hot archival (#36).

Covers the storage-layer operations:

- **Cold DB schema**: mirrors hot ``sessions`` + ``blocks`` without
  ``archive_state``.
- **Archival transaction**: ATTACH ``archive/YYYY-MM.db``; INSERT into
  cold; DELETE from hot; VACUUM; DETACH. Atomic + idempotent re-run.
- **Archive index** (``archive/index.json``): ``session_id →
  {archive_db, archived_at}``; lookup doesn't scan every archive.
- **Promote-back**: the first new write moves the session back to the
  hot DB atomically with ``updated_at = now`` and updates the index.
- **Hot-or-cold delete**: ``cothis delete`` works regardless of location.

Tests are offline (no LLM, no network) — they build temp DBs and verify
the SQL + JSON contracts directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cothis.session import Session
from cothis.session.archive import (
    ArchiveIndex,
    archive_session,
    promote_session,
    run_archival_pass,
)
from cothis.session.storage import Storage

if TYPE_CHECKING:
    import json
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
    """Create a session with alternating user/assistant turns; return id."""
    s = Session.new(db_path, cwd=cwd, model=model, flush_sync=True)
    sid = s.session_id
    for i, t in enumerate(texts):
        role = "user" if i % 2 == 0 else "assistant"
        s.append_message(role, [_user_text(t)])
    s.close()
    return sid


def _set_updated_at(db_path: Path, sid: str, updated_at: str) -> None:
    """Force a session's updated_at (for aging-it-out tests)."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE sessions SET updated_at=? WHERE id=?", (updated_at, sid)
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------
# ArchiveIndex
# ---------------------------------------------------------------------


def test_archive_index_round_trip(tmp_path: Path) -> None:
    """Write + read ``archive/index.json``; missing file → empty index."""
    index_path = tmp_path / "archive" / "index.json"
    index = ArchiveIndex(index_path)
    assert len(index) == 0  # missing file → empty

    index.set("a" * 32, "2026-07.db", "2026-07-20T00:00:00+00:00")
    index.save()

    reloaded = ArchiveIndex(index_path)
    assert len(reloaded) == 1
    entry = reloaded.get("a" * 32)
    assert entry is not None
    assert entry.archive_db == "2026-07.db"
    assert entry.archived_at == "2026-07-20T00:00:00+00:00"


def test_archive_index_get_unknown_returns_none(tmp_path: Path) -> None:
    """Unknown id → ``None`` (caller surfaces "not archived")."""
    index = ArchiveIndex(tmp_path / "archive" / "index.json")
    assert index.get("a" * 32) is None


def test_archive_index_remove_drops_entry(tmp_path: Path) -> None:
    """Promote-back discards the index entry; cold lookup no longer hits."""
    index_path = tmp_path / "archive" / "index.json"
    index = ArchiveIndex(index_path)
    index.set("a" * 32, "2026-07.db", "2026-07-20T00:00:00+00:00")
    index.save()

    reloaded = ArchiveIndex(index_path)
    assert reloaded.get("a" * 32) is not None
    reloaded.remove("a" * 32)
    reloaded.save()

    final = ArchiveIndex(index_path)
    assert final.get("a" * 32) is None


# ---------------------------------------------------------------------
# Archival transaction
# ---------------------------------------------------------------------


def test_archive_session_moves_rows_to_cold_db(tmp_path: Path) -> None:
    """``archive_session`` ATTACHes the cold DB, copies rows, deletes from hot."""
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["one", "two", "three"])
    archive_dir = tmp_path / "archive"

    archive_session(
        hot_db_path=db_path,
        archive_dir=archive_dir,
        session_id=sid,
        archive_db_name="2026-07.db",
        archived_at="2026-07-20T00:00:00+00:00",
        index=ArchiveIndex(archive_dir / "index.json"),
    )

    # Hot DB no longer has the session.
    hot = Storage(db_path)
    try:
        assert hot.load_session(sid) is None
        assert hot.load_blocks(sid) == []
    finally:
        hot.close()

    # Cold DB has both rows and blocks.
    cold = Storage(archive_dir / "2026-07.db")
    try:
        sr = cold.load_session(sid)
        assert sr is not None
        assert sr.id == sid
        blocks = cold.load_blocks(sid)
        assert len(blocks) == 3
    finally:
        cold.close()


def test_archive_session_is_idempotent_on_rerun(tmp_path: Path) -> None:
    """Re-running archival on an already-archived session is a no-op."""
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["one"])
    archive_dir = tmp_path / "archive"
    index = ArchiveIndex(archive_dir / "index.json")

    archive_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        archive_db_name="2026-07.db",
        archived_at="2026-07-20T00:00:00+00:00", index=index,
    )
    # Re-run with same args — should not raise, should not duplicate.
    archive_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        archive_db_name="2026-07.db",
        archived_at="2026-07-20T00:00:00+00:00", index=index,
    )

    cold = Storage(archive_dir / "2026-07.db")
    try:
        blocks = cold.load_blocks(sid)
        assert len(blocks) == 1  # not duplicated
    finally:
        cold.close()


def test_archive_session_updates_index(tmp_path: Path) -> None:
    """The index records where the session landed + when."""
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["one"])
    archive_dir = tmp_path / "archive"
    index = ArchiveIndex(archive_dir / "index.json")

    archive_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        archive_db_name="2026-07.db",
        archived_at="2026-07-20T00:00:00+00:00", index=index,
    )

    entry = index.get(sid)
    assert entry is not None
    assert entry.archive_db == "2026-07.db"
    assert entry.archived_at == "2026-07-20T00:00:00+00:00"


# ---------------------------------------------------------------------
# Startup archival pass
# ---------------------------------------------------------------------


def test_run_archival_pass_moves_idle_sessions(tmp_path: Path) -> None:
    """Sessions idle past the threshold move to the monthly cold DB."""
    db_path = tmp_path / "session.db"
    old_sid = _seed_session(db_path, tmp_path, texts=["old"])
    new_sid = _seed_session(db_path, tmp_path, texts=["new"])
    # Force the "old" session's updated_at back 100 days.
    _set_updated_at(db_path, old_sid, "2026-04-13T00:00:00+00:00")
    archive_dir = tmp_path / "archive"

    run_archival_pass(
        hot_db_path=db_path,
        archive_dir=archive_dir,
        threshold_days=90,
        now_iso="2026-07-20T00:00:00+00:00",
    )

    hot = Storage(db_path)
    try:
        assert hot.load_session(old_sid) is None  # moved out
        assert hot.load_session(new_sid) is not None  # stayed
    finally:
        hot.close()

    index = ArchiveIndex(archive_dir / "index.json")
    assert index.get(old_sid) is not None
    assert index.get(new_sid) is None


def test_run_archival_pass_throttles_via_archive_state(tmp_path: Path) -> None:
    """The pass records ``last_run`` in ``archive_state`` and skips if < 24h old."""
    db_path = tmp_path / "session.db"
    _seed_session(db_path, tmp_path, texts=["old"])
    _set_updated_at(db_path, _first_session_id(db_path), "2026-04-13T00:00:00+00:00")
    archive_dir = tmp_path / "archive"

    # First run: archives.
    run_archival_pass(
        hot_db_path=db_path, archive_dir=archive_dir, threshold_days=90,
        now_iso="2026-07-20T00:00:00+00:00",
    )

    # Seed another old session, run again within 24h — should skip.
    second_sid = _seed_session(db_path, tmp_path, texts=["second-old"])
    _set_updated_at(db_path, second_sid, "2026-04-13T00:00:00+00:00")

    run_archival_pass(
        hot_db_path=db_path, archive_dir=archive_dir, threshold_days=90,
        now_iso="2026-07-20T12:00:00+00:00",  # 12h later
    )

    # The second session was NOT archived (pass skipped).
    hot = Storage(db_path)
    try:
        assert hot.load_session(second_sid) is not None
    finally:
        hot.close()


def _first_session_id(db_path: Path) -> str:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT id FROM sessions LIMIT 1").fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Promote-back
# ---------------------------------------------------------------------


def test_promote_session_brings_archived_session_back_to_hot(tmp_path: Path) -> None:
    """The first new write promotes the session: rows copied back, index removed."""
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["one", "two"])
    archive_dir = tmp_path / "archive"
    index = ArchiveIndex(archive_dir / "index.json")

    archive_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        archive_db_name="2026-07.db",
        archived_at="2026-07-20T00:00:00+00:00", index=index,
    )

    promote_session(
        hot_db_path=db_path,
        archive_dir=archive_dir,
        session_id=sid,
        index=index,
    )

    # Hot DB has the session back; index no longer references it.
    hot = Storage(db_path)
    try:
        assert hot.load_session(sid) is not None
        assert len(hot.load_blocks(sid)) == 2
    finally:
        hot.close()
    assert index.get(sid) is None


def test_promote_session_sets_updated_at_now(tmp_path: Path) -> None:
    """Promoted sessions get ``updated_at = now`` so they aren't immediately re-archived."""
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["x"])
    archive_dir = tmp_path / "archive"
    index = ArchiveIndex(archive_dir / "index.json")
    archive_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        archive_db_name="2026-07.db",
        archived_at="2026-07-20T00:00:00+00:00", index=index,
    )

    promote_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        index=index,
        now_iso="2026-09-01T00:00:00+00:00",
    )

    hot = Storage(db_path)
    try:
        sr = hot.load_session(sid)
        assert sr is not None
        assert sr.updated_at == "2026-09-01T00:00:00+00:00"
    finally:
        hot.close()
