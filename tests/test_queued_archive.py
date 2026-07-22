"""Tests for queued UPDATE archival (Half B, #168).

When ``Session._deactivate_skill(name)`` is called, the session posts
an UPDATE task to the same queue that handles block writes. The
UPDATE marks all rows with ``skill=name`` as ``state='archived'``,
covering:

* Rows already flushed to SQLite (historical).
* Rows currently in-flight in the queue (queued before deactivate).
* Rows enqueued *after* deactivate (Half A in #167 marks these
  directly at enqueue time, but they're also covered idempotently
  by the UPDATE).

Runs on the consumer thread (single-writer invariant; no SQLite
concurrency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cothis.session import Session
from cothis.session.storage import BlockRow, SessionRow, Storage

if TYPE_CHECKING:
    from pathlib import Path


_FRONTMATTER_BLOCK = {
    "type": "tool_use",
    "id": "tu",
    "name": "load_skill",
    "input": {"name": "python"},
    "_cothis_skill": "python",
}


def _archive_skill_rows(
    storage: Storage, session_id: str, skill: str
) -> int:
    """Helper: count rows with ``skill=X`` (any state)."""
    rows = storage.load_blocks(session_id)
    return sum(1 for r in rows if r.skill == skill)


def _archived_skill_rows(
    storage: Storage, session_id: str, skill: str
) -> int:
    """Helper: count rows with ``skill=X, state='archived'``."""
    rows = storage.load_blocks(session_id)
    return sum(1 for r in rows if r.skill == skill and r.state == "archived")


# ---------------------------------------------------------------------
# Storage layer: archive_skill_blocks
# ---------------------------------------------------------------------


def test_storage_archive_skill_blocks_updates_matching_rows(
    tmp_path: Path,
) -> None:
    """``Storage.archive_skill_blocks`` UPDATEs all rows for (session, skill)."""
    db = tmp_path / "test.db"
    storage = Storage(db)
    sr = SessionRow(
        id="s1", parent_id=None, parent_seq=None, cwd="/x",
        cli_version="0.1.0", model="m", title="t",
        created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
    )
    rows = [
        BlockRow(
            session_id="s1", seq=0, msg_idx=0, block_idx=0,
            role="assistant", type="tool_use", ts="2026-01-01T00:00:00Z",
            content=None, signature=None, tool_id="t1", tool_name="load_skill",
            tool_input='{"name": "python"}', tool_use_id=None,
            tool_output=None, image_source=None, skill="python", state=None,
        ),
        BlockRow(
            session_id="s1", seq=1, msg_idx=1, block_idx=0,
            role="user", type="text", ts="2026-01-01T00:00:00Z",
            content="hi", signature=None, tool_id=None, tool_name=None,
            tool_input=None, tool_use_id=None,
            tool_output=None, image_source=None, skill=None, state=None,
        ),
        BlockRow(
            session_id="s1", seq=2, msg_idx=2, block_idx=0,
            role="user", type="tool_result", ts="2026-01-01T00:00:01Z",
            content="ok", signature=None, tool_id=None, tool_name=None,
            tool_input=None, tool_use_id="t1",
            tool_output="ok", image_source=None, skill="python", state=None,
        ),
    ]
    storage.write_atomic(sr, rows, "2026-01-01T00:00:02Z")

    updated = storage.archive_skill_blocks("s1", "python")
    assert updated == 2  # tool_use + tool_result; text row untouched.

    reloaded = storage.load_blocks("s1")
    states = {(r.skill, r.state) for r in reloaded}
    assert ("python", "archived") in states
    assert (None, None) in states  # text row kept state=None
    storage.close()


def test_storage_archive_skill_blocks_idempotent(tmp_path: Path) -> None:
    """Re-running archive on already-archived rows: no-op, no error."""
    db = tmp_path / "test.db"
    storage = Storage(db)
    sr = SessionRow(
        id="s1", parent_id=None, parent_seq=None, cwd="/x",
        cli_version="0.1.0", model="m", title="t",
        created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
    )
    rows = [BlockRow(
        session_id="s1", seq=0, msg_idx=0, block_idx=0,
        role="assistant", type="tool_use", ts="2026-01-01T00:00:00Z",
        content=None, signature=None, tool_id="t1", tool_name="load_skill",
        tool_input='{"name": "python"}', tool_use_id=None,
        tool_output=None, image_source=None, skill="python", state=None,
    )]
    storage.write_atomic(sr, rows, "2026-01-01T00:00:02Z")

    first = storage.archive_skill_blocks("s1", "python")
    second = storage.archive_skill_blocks("s1", "python")
    assert first == 1
    # Second call still returns count of matched rows (UPDATE is idempotent
    # on the state value; the count tells the caller what ran).
    assert second == 1
    storage.close()


def test_storage_archive_skill_blocks_unknown_skill_noop(tmp_path: Path) -> None:
    """No rows match the skill → UPDATE matches 0 rows, no error."""
    db = tmp_path / "test.db"
    storage = Storage(db)
    sr = SessionRow(
        id="s1", parent_id=None, parent_seq=None, cwd="/x",
        cli_version="0.1.0", model="m", title="t",
        created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
    )
    storage.write_atomic(sr, [], "2026-01-01T00:00:02Z")
    assert storage.archive_skill_blocks("s1", "nonexistent") == 0
    storage.close()


# ---------------------------------------------------------------------
# Session layer: _deactivate_skill posts queued UPDATE
# ---------------------------------------------------------------------


def test_deactivate_archives_already_flushed_rows(tmp_path: Path) -> None:
    """Half B: historical rows (already in SQLite) get UPDATEd.

    Loads → flush (write reaches disk) → deactivate → drain → all
    python rows now ``state='archived'``.
    """
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    s._activate_skill("python")
    s.append_message("assistant", [dict(_FRONTMATTER_BLOCK)])
    s.append_block(
        "user",
        {
            "type": "tool_result",
            "tool_use_id": "tu",
            "content": "ok",
            "_cothis_skill": "python",
        },
    )
    # Confirm pre-state: 2 python rows, state=None.
    pre = s._storage.load_blocks(s._session_id)
    assert sum(1 for r in pre if r.skill == "python" and r.state is None) == 2

    s._deactivate_skill("python")
    # In flush_sync=True mode the archive op ran inline. No drain needed.

    post = s._storage.load_blocks(s._session_id)
    archived = [r for r in post if r.skill == "python"]
    assert len(archived) == 2
    assert all(r.state == "archived" for r in archived)
    s.close()


def test_deactivate_covers_in_flight_rows(tmp_path: Path) -> None:
    """Half B: rows queued before deactivate are caught by the UPDATE.

    Uses async queue (flush_sync=False). Enqueue a block, then before
    the consumer drains it, call _deactivate_skill. The archive op
    lands AFTER the block's INSERT in queue order, so when both drain,
    the block ends up ``state='archived'``.
    """
    db_path = tmp_path / "db.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=False)
    session_id = s._session_id
    s._activate_skill("python")
    # Enqueue a python-tagged block. The consumer may drain it at any
    # time, but the queue is FIFO — the archive op posted by deactivate
    # runs strictly after.
    s.append_message("assistant", [dict(_FRONTMATTER_BLOCK)])
    s._deactivate_skill("python")
    s.close()  # waits for queue drain

    # Re-open storage after close (close() closed the connection).
    storage = Storage(db_path)
    rows = storage.load_blocks(session_id)
    python_rows = [r for r in rows if r.skill == "python"]
    assert len(python_rows) == 1
    assert python_rows[0].state == "archived"
    storage.close()


def test_deactivate_combines_half_a_and_half_b(tmp_path: Path) -> None:
    """End-to-end: load → deactivate → load epoch; both epochs archived.

    First epoch: pre-deactivate, queued + flushed normally. UPDATE
    catches them.
    Second epoch: post-deactivate, Half A marks them directly at
    enqueue time.
    """
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    s._activate_skill("python")
    # Epoch 1
    s.append_message("assistant", [dict(_FRONTMATTER_BLOCK)])
    s._deactivate_skill("python")
    # Epoch 2 (Half A handles directly)
    s.append_message("assistant", [dict(_FRONTMATTER_BLOCK)])

    rows = s._storage.load_blocks(s._session_id)
    python_rows = [r for r in rows if r.skill == "python"]
    assert len(python_rows) == 2
    assert all(r.state == "archived" for r in python_rows)
    s.close()


def test_deactivate_only_archives_named_skill(tmp_path: Path) -> None:
    """Archival is skill-specific: deactivating python leaves bash rows alone."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    s._activate_skill("python")
    s._activate_skill("bash")
    s.append_message("assistant", [{
        "type": "tool_use", "id": "p1", "name": "load_skill",
        "input": {"name": "python"}, "_cothis_skill": "python",
    }])
    s.append_message("assistant", [{
        "type": "tool_use", "id": "b1", "name": "load_skill",
        "input": {"name": "bash"}, "_cothis_skill": "bash",
    }])
    s._deactivate_skill("python")

    rows = s._storage.load_blocks(s._session_id)
    python = [r for r in rows if r.skill == "python"]
    bash = [r for r in rows if r.skill == "bash"]
    assert all(r.state == "archived" for r in python)
    assert all(r.state is None for r in bash)
    s.close()


def test_deactivate_repeat_epoch_idempotent(tmp_path: Path) -> None:
    """Half B repeat: deactivate twice doesn't double-archive or error."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    s._activate_skill("python")
    s.append_message("assistant", [dict(_FRONTMATTER_BLOCK)])
    s._deactivate_skill("python")
    s._deactivate_skill("python")  # repeat — no-op, no second queue op

    rows = s._storage.load_blocks(s._session_id)
    python = [r for r in rows if r.skill == "python"]
    assert len(python) == 1
    assert python[0].state == "archived"
    s.close()
