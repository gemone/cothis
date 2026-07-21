"""Tests for ``cothis.session.Session.delete`` cold extension (#87).

Covers the four acceptance criteria of #87:

- **Hot delete** still works (existing #35 path).
- **Cold delete** succeeds when the session was archived.
- **Index entry** dropped on cold delete.
- **Leaf-only** check applies across both DBs (a hot parent with cold
  children is refused; a cold parent with hot children is refused).

Tests are offline (no LLM, no network).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cothis.session import Session, SessionHasChildrenError
from cothis.session.archive import (
    ArchiveIndex,
    archive_session,
    cold_session_children,
    delete_cold_session,
)
from cothis.session.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _user_text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _seed_session(
    db_path: Path, cwd: Path, *, model: str = "m", texts: list[str],
    parent_id: str | None = None,
) -> str:
    s = Session.new(db_path, cwd=cwd, model=model, flush_sync=True)
    sid = s.session_id
    if parent_id is not None:
        # Bypass the public API to inject a fork link directly for
        # the leaf-only tests. The session row is written on first
        # append below, then patched via a direct SQL UPDATE.
        s._parent_id = parent_id
        s._parent_seq = 0
    for i, t in enumerate(texts):
        role = "user" if i % 2 == 0 else "assistant"
        s.append_message(role, [_user_text(t)])
    s.close()

    if parent_id is not None:
        import sqlite3

        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "UPDATE sessions SET parent_id=? WHERE id=?",
                (parent_id, sid),
            )
            conn.commit()
        finally:
            conn.close()
    return sid


def _archive_one(
    db_path: Path, archive_dir: Path, sid: str, *, archive_db: str = "2026-07.db"
) -> ArchiveIndex:
    """Move sid to the named cold DB; return the (already-saved) index."""
    index = ArchiveIndex(archive_dir / "index.json")
    archive_session(
        hot_db_path=db_path, archive_dir=archive_dir, session_id=sid,
        archive_db_name=archive_db,
        archived_at="2026-07-20T00:00:00+00:00", index=index,
    )
    return index


# ---------------------------------------------------------------------
# Existing hot-path behavior (no regression)
# ---------------------------------------------------------------------


def test_delete_hot_session_succeeds(tmp_path: Path) -> None:
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["hi"])
    Session.delete(db_path, sid)
    hot = Storage(db_path)
    try:
        assert hot.load_session(sid) is None
        assert hot.load_blocks(sid) == []
    finally:
        hot.close()


def test_delete_missing_in_both_raises_keyerror(tmp_path: Path) -> None:
    db_path = tmp_path / "session.db"
    # Initialise the hot DB so Storage() can open it.
    _seed_session(db_path, tmp_path, texts=["seed"])
    with pytest.raises(KeyError):
        Session.delete(db_path, "0" * 32)


# ---------------------------------------------------------------------
# Cold delete
# ---------------------------------------------------------------------


def test_delete_cold_session_succeeds(tmp_path: Path) -> None:
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["archived"])
    archive_dir = db_path.parent / "archive"
    _archive_one(db_path, archive_dir, sid)

    # Sanity: hot miss + cold hit before delete.
    hot = Storage(db_path)
    try:
        assert hot.load_session(sid) is None
    finally:
        hot.close()

    Session.delete(db_path, sid)

    # Hot still misses; index entry dropped; cold DB row gone.
    hot = Storage(db_path)
    try:
        assert hot.load_session(sid) is None
    finally:
        hot.close()
    reloaded = ArchiveIndex(archive_dir / "index.json")
    assert reloaded.get(sid) is None
    # The cold DB file is still there (we only delete rows + VACUUM).
    assert (archive_dir / "2026-07.db").is_file()
    import sqlite3

    conn = sqlite3.connect(archive_dir / "2026-07.db")
    try:
        assert conn.execute(
            "SELECT 1 FROM sessions WHERE id=?", (sid,)
        ).fetchone() is None
    finally:
        conn.close()


def test_delete_cold_session_drops_index_entry(tmp_path: Path) -> None:
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["archived"])
    archive_dir = db_path.parent / "archive"
    _archive_one(db_path, archive_dir, sid)
    assert ArchiveIndex(archive_dir / "index.json").get(sid) is not None

    Session.delete(db_path, sid)
    reloaded = ArchiveIndex(archive_dir / "index.json")
    assert reloaded.get(sid) is None


def test_delete_cold_session_handles_stale_index(tmp_path: Path) -> None:
    """Index entry pointing at a missing cold DB → KeyError + drop."""
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["archived"])
    archive_dir = db_path.parent / "archive"
    _archive_one(db_path, archive_dir, sid)
    (archive_dir / "2026-07.db").unlink()

    with pytest.raises(KeyError):
        Session.delete(db_path, sid)
    reloaded = ArchiveIndex(archive_dir / "index.json")
    assert reloaded.get(sid) is None


def test_delete_cold_session_unit_function_idempotent(tmp_path: Path) -> None:
    """``delete_cold_session`` returns False on second call (index empty)."""
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["archived"])
    archive_dir = db_path.parent / "archive"
    index = _archive_one(db_path, archive_dir, sid)

    assert delete_cold_session(
        hot_db_path=db_path, archive_dir=archive_dir,
        session_id=sid, index=index,
    ) is True
    # Second call — index already empty.
    assert delete_cold_session(
        hot_db_path=db_path, archive_dir=archive_dir,
        session_id=sid, index=index,
    ) is False


# ---------------------------------------------------------------------
# Leaf-only across both DBs
# ---------------------------------------------------------------------


def test_delete_refuses_cold_session_with_hot_child(tmp_path: Path) -> None:
    """A cold parent with a hot child is refused (leaf-only spans DBs)."""
    db_path = tmp_path / "session.db"
    parent_sid = _seed_session(db_path, tmp_path, texts=["parent"])
    # Child stays hot (recently touched).
    child_sid = _seed_session(
        db_path, tmp_path, texts=["child"], parent_id=parent_sid,
    )
    archive_dir = db_path.parent / "archive"
    _archive_one(db_path, archive_dir, parent_sid)

    with pytest.raises(SessionHasChildrenError) as exc_info:
        Session.delete(db_path, parent_sid)
    assert child_sid in exc_info.value.children

    # Nothing deleted — cold row + index entry both still there.
    reloaded = ArchiveIndex(archive_dir / "index.json")
    assert reloaded.get(parent_sid) is not None


def test_delete_refuses_hot_session_with_cold_child(tmp_path: Path) -> None:
    """A hot parent with a cold child is refused (leaf-only spans DBs)."""
    db_path = tmp_path / "session.db"
    parent_sid = _seed_session(db_path, tmp_path, texts=["parent"])
    child_sid = _seed_session(
        db_path, tmp_path, texts=["child"], parent_id=parent_sid,
    )
    archive_dir = db_path.parent / "archive"
    # Move only the child to cold.
    _archive_one(db_path, archive_dir, child_sid)

    with pytest.raises(SessionHasChildrenError) as exc_info:
        Session.delete(db_path, parent_sid)
    assert child_sid in exc_info.value.children

    # Cold row + index entry for child both still there.
    reloaded = ArchiveIndex(archive_dir / "index.json")
    assert reloaded.get(child_sid) is not None


def test_delete_refuses_cold_session_with_cold_child(tmp_path: Path) -> None:
    """A cold parent with a cold child (other monthly bucket) is refused."""
    db_path = tmp_path / "session.db"
    parent_sid = _seed_session(db_path, tmp_path, texts=["parent"])
    child_sid = _seed_session(
        db_path, tmp_path, texts=["child"], parent_id=parent_sid,
    )
    archive_dir = db_path.parent / "archive"
    # Parent in June, child in July — different monthly buckets.
    _archive_one(db_path, archive_dir, parent_sid, archive_db="2026-06.db")
    _archive_one(db_path, archive_dir, child_sid, archive_db="2026-07.db")

    with pytest.raises(SessionHasChildrenError) as exc_info:
        Session.delete(db_path, parent_sid)
    assert child_sid in exc_info.value.children


def test_cold_session_children_helper_walks_all_dbs(tmp_path: Path) -> None:
    """``cold_session_children`` queries every YYYY-MM.db under archive_dir."""
    db_path = tmp_path / "session.db"
    parent_sid = _seed_session(db_path, tmp_path, texts=["parent"])
    june_child = _seed_session(
        db_path, tmp_path, texts=["june"], parent_id=parent_sid,
    )
    july_child = _seed_session(
        db_path, tmp_path, texts=["july"], parent_id=parent_sid,
    )
    archive_dir = db_path.parent / "archive"
    _archive_one(db_path, archive_dir, june_child, archive_db="2026-06.db")
    _archive_one(db_path, archive_dir, july_child, archive_db="2026-07.db")

    found = cold_session_children(
        archive_dir=archive_dir, session_id=parent_sid,
    )
    assert set(found) == {june_child, july_child}


def test_cold_session_children_opens_bounded_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection count stays bounded regardless of archive count (#127).

    ceil(N / batch_size) connections, where ``batch_size`` respects
    ``SQLITE_LIMIT_ATTACHED``. The load-bearing invariant is
    "connections < archives" — the per-batch constant doesn't matter
    for this test, only that opening 24 archives doesn't open 24
    connections.
    """
    db_path = tmp_path / "session.db"
    parent_sid = _seed_session(db_path, tmp_path, texts=["parent"])
    archive_dir = db_path.parent / "archive"
    archive_dir.mkdir()
    # 24 monthly DBs (2 years), each empty (no children of parent).
    # Pre-fix: 24 connections. Post-fix: bounded by batch size.
    import sqlite3

    from cothis.session import archive as archive_module
    for year in range(2024, 2026):
        for month in range(1, 13):
            cold = archive_dir / f"{year}-{month:02d}.db"
            conn = sqlite3.connect(cold)
            try:
                # Cold schema minimal — no rows needed for the perf path.
                conn.executescript(
                    "CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_id TEXT);"
                )
            finally:
                conn.close()

    connect_calls = 0
    real_connect = sqlite3.connect

    def counting_connect(path, *args, **kwargs):
        nonlocal connect_calls
        connect_calls += 1
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(archive_module.sqlite3, "connect", counting_connect)

    found = cold_session_children(
        archive_dir=archive_dir, session_id=parent_sid,
    )
    assert found == []
    # 24 archives pre-fix → 24 connections. Post-fix (batch=9) →
    # ceil(24/9) = 3. Allow generous headroom for any batch size
    # choice; the load-bearing invariant is "connections < archives".
    assert connect_calls < 24, (
        f"cold_session_children opened {connect_calls} connections for "
        f"24 archives; expected bounded < 24 (#127 regression)"
    )
