"""Tests for resume rebuild of ``active_skills`` (#71).

At ``Session.load``, the active set is rebuilt by scanning the
``load_skill`` / ``deactivate_skill`` ``tool_use`` sequence per skill
(most-recent wins; ``state`` is not consulted). Skills that vanished
from disk since the last session are dropped from the active set,
their tagged blocks archived via the queued UPDATE, and a warning is
logged — the session continues with the surviving skills.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from cothis.session import Session
from cothis.session.storage import BlockRow, SessionRow, Storage

if TYPE_CHECKING:
    from pathlib import Path


_LOAD = "load_skill"
_DEACTIVATE = "deactivate_skill"


def _now_iso() -> str:
    """Current UTC ISO timestamp (test sessions must look fresh or
    ``_run_startup_archival`` will archive them — 90 day threshold)."""
    return datetime.now(UTC).isoformat()


def _recent_ts(offset_seconds: int = 0) -> str:
    return (datetime.now(UTC) - timedelta(seconds=offset_seconds)).isoformat()


@pytest.fixture
def _all_skills_on_disk() -> object:
    """Make ``discover_skills`` find any name we ask about.

    Used by rebuild-only tests so the vanished-on-disk archival
    doesn't fire and pollute the active-set assertions.
    """
    from pathlib import Path as _Path

    from cothis.skills import Skill

    def _fake_discover(_cwd, **_kw):
        names = (
            "python", "bash", "gone", "also-gone", "other",
        )
        return [
            Skill(
                name=n, description=f"{n} d", body=f"{n} b",
                source=_Path(f"/tmp/{n}/SKILL.md"),
                deactivation="delete",
            )
            for n in names
        ]

    with patch("cothis.skills.discover_skills", side_effect=_fake_discover):
        yield


def _skill_tool_use(
    session_id: str, seq: int, msg_idx: int, skill: str, op: str,
    ts: str = _recent_ts(60),
) -> BlockRow:
    """Build a tool_use row for load_skill or deactivate_skill."""
    return BlockRow(
        session_id=session_id, seq=seq, msg_idx=msg_idx, block_idx=0,
        role="assistant", type="tool_use", ts=ts,
        content=None, signature=None, tool_id=f"t{seq}",
        tool_name=op, tool_input=json.dumps({"name": skill}),
        tool_use_id=None, tool_output=None, image_source=None,
        skill=skill, state=None,
    )


def _skill_tool_result(
    session_id: str, seq: int, msg_idx: int, tool_id: str, skill: str,
    ts: str = _recent_ts(30),
) -> BlockRow:
    """Build a tool_result row pairing with a skill tool_use.

    Required so the tool_use survives ``_rebuild_messages`` orphan
    truncation at resume time.
    """
    return BlockRow(
        session_id=session_id, seq=seq, msg_idx=msg_idx, block_idx=0,
        role="user", type="tool_result", ts=ts,
        content="ok", signature=None, tool_id=None, tool_name=None,
        tool_input=None, tool_use_id=tool_id,
        tool_output="ok", image_source=None,
        skill=skill, state=None,
    )


def _skill_op_pair(
    session_id: str, base_seq: int, base_msg: int, skill: str, op: str,
) -> list[BlockRow]:
    """Build a (tool_use, tool_result) pair for one skill operation."""
    tu = _skill_tool_use(session_id, base_seq, base_msg, skill, op)
    tr = _skill_tool_result(
        session_id, base_seq + 1, base_msg + 1, f"t{base_seq}", skill,
    )
    return [tu, tr]


def _write_session_with_rows(
    db_path: Path, session_id: str, rows: list[BlockRow],
) -> None:
    """Persist a session + rows so Session.load can read them."""
    storage = Storage(db_path)
    sr = SessionRow(
        id=session_id, parent_id=None, parent_seq=None, cwd=str(db_path.parent),
        cli_version="0.1.0", model="m", title="t",
        created_at=_recent_ts(60), updated_at=_recent_ts(30),
    )
    storage.write_atomic(sr, rows, _now_iso())
    storage.close()


# ---------------------------------------------------------------------
# _rebuild_active_skills_from_rows
# ---------------------------------------------------------------------


def test_rebuild_single_load_marks_skill_active(
    tmp_path: Path, _all_skills_on_disk: None,
) -> None:
    """One load_skill for 'python' → active after resume."""
    db_path = tmp_path / "db.db"
    sid = "abcdef0123456789abcdef0123456789"
    _write_session_with_rows(db_path, sid, [
        *_skill_op_pair(sid, 0, 0, "python", _LOAD),
    ])
    s = Session.load(db_path, sid, flush_sync=True)
    assert "python" in s.active_skills
    s.close()


def test_rebuild_load_then_deactivate_drops_skill(
    tmp_path: Path, _all_skills_on_disk: None,
) -> None:
    """load → deactivate: skill not in active set after resume."""
    db_path = tmp_path / "db.db"
    sid = "abcdef0123456789abcdef0123456789"
    _write_session_with_rows(db_path, sid, [
        *_skill_op_pair(sid, 0, 0, "python", _LOAD),
        *_skill_op_pair(sid, 2, 1, "python", _DEACTIVATE),
    ])
    s = Session.load(db_path, sid, flush_sync=True)
    assert "python" not in s.active_skills
    s.close()


def test_rebuild_repeated_epoch_most_recent_wins(
    tmp_path: Path, _all_skills_on_disk: None,
) -> None:
    """load → deactivate → load: skill active (most recent wins)."""
    db_path = tmp_path / "db.db"
    sid = "abcdef0123456789abcdef0123456789"
    _write_session_with_rows(db_path, sid, [
        *_skill_op_pair(sid, 0, 0, "python", _LOAD),
        *_skill_op_pair(sid, 2, 1, "python", _DEACTIVATE),
        *_skill_op_pair(sid, 4, 2, "python", _LOAD),
    ])
    s = Session.load(db_path, sid, flush_sync=True)
    assert "python" in s.active_skills
    s.close()


def test_rebuild_multiple_distinct_skills(
    tmp_path: Path, _all_skills_on_disk: None,
) -> None:
    """load python + load bash → both active after resume."""
    db_path = tmp_path / "db.db"
    sid = "abcdef0123456789abcdef0123456789"
    _write_session_with_rows(db_path, sid, [
        *_skill_op_pair(sid, 0, 0, "python", _LOAD),
        *_skill_op_pair(sid, 2, 1, "bash", _LOAD),
    ])
    s = Session.load(db_path, sid, flush_sync=True)
    assert "python" in s.active_skills
    assert "bash" in s.active_skills
    s.close()


def test_rebuild_no_skill_activity_empty(
    tmp_path: Path, _all_skills_on_disk: None,
) -> None:
    """No skill tool_use rows → empty active set."""
    db_path = tmp_path / "db.db"
    sid = "abcdef0123456789abcdef0123456789"
    storage = Storage(db_path)
    sr = SessionRow(
        id=sid, parent_id=None, parent_seq=None, cwd=str(tmp_path),
        cli_version="0.1.0", model="m", title="t",
        created_at=_recent_ts(60), updated_at=_recent_ts(30),
    )
    rows = [BlockRow(
        session_id=sid, seq=0, msg_idx=0, block_idx=0,
        role="user", type="text", ts=_recent_ts(60),
        content="hello", signature=None, tool_id=None, tool_name=None,
        tool_input=None, tool_use_id=None, tool_output=None,
        image_source=None, skill=None, state=None,
    )]
    storage.write_atomic(sr, rows, _now_iso())
    storage.close()
    s = Session.load(db_path, sid, flush_sync=True)
    assert s.active_skills == frozenset()
    s.close()


def test_rebuild_deactivated_then_resumed_stays_deactivated(
    tmp_path: Path, _all_skills_on_disk: None,
) -> None:
    """load → deactivate (no further load): not in active set."""
    db_path = tmp_path / "db.db"
    sid = "abcdef0123456789abcdef0123456789"
    _write_session_with_rows(db_path, sid, [
        *_skill_op_pair(sid, 0, 0, "python", _LOAD),
        *_skill_op_pair(sid, 2, 1, "python", _DEACTIVATE),
        # An unrelated second load for a different skill doesn't
        # bring python back.
        *_skill_op_pair(sid, 4, 2, "bash", _LOAD),
    ])
    s = Session.load(db_path, sid, flush_sync=True)
    assert "python" not in s.active_skills
    assert "bash" in s.active_skills
    s.close()


def test_rebuild_ignores_state_column(
    tmp_path: Path, _all_skills_on_disk: None,
) -> None:
    """Resume rebuild reads tool_use history regardless of state column.

    A deactivated skill's rows carry state='archived' (Half B), but
    the rebuild logic must still see the load → deactivate sequence
    to derive the correct active set.
    """
    db_path = tmp_path / "db.db"
    storage = Storage(db_path)
    sr = SessionRow(
        id="abcdef0123456789abcdef0123456789", parent_id=None, parent_seq=None, cwd=str(tmp_path),
        cli_version="0.1.0", model="m", title="t",
        created_at=_recent_ts(60), updated_at=_recent_ts(30),
    )
    # python: load (state=None) + deactivate (state='archived' from Half B)
    rows = [
        BlockRow(
            session_id="abcdef0123456789abcdef0123456789", seq=0, msg_idx=0, block_idx=0,
            role="assistant", type="tool_use", ts=_recent_ts(60),
            content=None, signature=None, tool_id="t0",
            tool_name=_LOAD, tool_input='{"name": "python"}',
            tool_use_id=None, tool_output=None, image_source=None,
            skill="python", state=None,
        ),
        BlockRow(
            session_id="abcdef0123456789abcdef0123456789", seq=1, msg_idx=1, block_idx=0,
            role="assistant", type="tool_use", ts=_recent_ts(60),
            content=None, signature=None, tool_id="t1",
            tool_name=_DEACTIVATE, tool_input='{"name": "python"}',
            tool_use_id=None, tool_output=None, image_source=None,
            skill="python", state=None,  # would be 'archived' in real flow
        ),
    ]
    storage.write_atomic(sr, rows, _now_iso())
    storage.close()
    s = Session.load(db_path, "abcdef0123456789abcdef0123456789", flush_sync=True)
    assert "python" not in s.active_skills
    s.close()


# ---------------------------------------------------------------------
# Vanished-on-disk archival
# ---------------------------------------------------------------------


def test_vanished_skill_dropped_and_archived(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Skill active at last save but no longer on disk → dropped + archived."""
    db_path = tmp_path / "db.db"
    sid = "abcdef0123456789abcdef0123456789"
    _write_session_with_rows(db_path, sid, [
        *_skill_op_pair(sid, 0, 0, "gone", _LOAD),
    ])
    # No SKILL.md for 'gone' on disk.
    with caplog.at_level(logging.WARNING, logger="cothis.session"):
        s = Session.load(db_path, sid, flush_sync=True)
    assert "gone" not in s.active_skills
    assert "gone" in s.archived_skills
    assert any(
        "gone" in r.message and (
            "vanished" in r.message.lower() or "archiv" in r.message.lower()
        )
        for r in caplog.records
    )
    s.close()


def test_surviving_skill_stays_active(tmp_path: Path) -> None:
    """Skill still on disk at resume → stays active, not archived."""
    # Place a SKILL.md for 'python' on disk.
    skill_dir = tmp_path / ".agents" / "skills" / "python"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: python\ndescription: d\n---\nbody\n", encoding="utf-8",
    )
    db_path = tmp_path / "db.db"
    sid = "abcdef0123456789abcdef0123456789"
    _write_session_with_rows(db_path, sid, [
        *_skill_op_pair(sid, 0, 0, "python", _LOAD),
    ])
    s = Session.load(db_path, sid, flush_sync=True, cwd=tmp_path)
    assert "python" in s.active_skills
    assert "python" not in s.archived_skills
    s.close()


def test_vanished_archival_propagates_to_storage(
    tmp_path: Path,
) -> None:
    """Vanished skill's blocks get state='archived' via Half B queued UPDATE."""
    db_path = tmp_path / "db.db"
    sid = "abcdef0123456789abcdef0123456789"
    _write_session_with_rows(db_path, sid, [
        *_skill_op_pair(sid, 0, 0, "gone", _LOAD),
    ])
    s = Session.load(db_path, sid, flush_sync=True)
    s.close()  # ensure queue drains

    # Re-open storage and verify state.
    storage = Storage(db_path)
    rows = storage.load_blocks(sid)
    storage.close()
    archived = [r for r in rows if r.skill == "gone"]
    assert len(archived) >= 1
    assert all(r.state == "archived" for r in archived)


def test_vanished_archival_skill_specific(tmp_path: Path) -> None:
    """Vanished skill is archived; surviving skill keeps its state."""
    skill_dir = tmp_path / ".agents" / "skills" / "python"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: python\ndescription: d\n---\nbody\n", encoding="utf-8",
    )
    db_path = tmp_path / "db.db"
    sid = "abcdef0123456789abcdef0123456789"
    _write_session_with_rows(db_path, sid, [
        *_skill_op_pair(sid, 0, 0, "python", _LOAD),
        *_skill_op_pair(sid, 2, 1, "gone", _LOAD),
    ])
    s = Session.load(db_path, sid, flush_sync=True, cwd=tmp_path)
    assert "python" in s.active_skills
    assert "gone" in s.archived_skills
    s.close()


def test_end_to_end_session_resume_round_trip(tmp_path: Path) -> None:
    """End-to-end: persist via real Session, load, verify active set."""
    db_path = tmp_path / "db.db"
    s1 = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s1._activate_skill("python")
    s1.append_message("assistant", [{
        "type": "tool_use", "id": "t1", "name": _LOAD,
        "input": {"name": "python"}, "_cothis_skill": "python",
    }])
    # Pair the tool_use with a tool_result so it survives _rebuild_messages
    # orphan-truncate at resume time.
    s1.append_block("user", {
        "type": "tool_result", "tool_use_id": "t1", "content": "loaded",
        "_cothis_skill": "python",
    })
    s1.close()

    # Place SKILL.md so python isn't 'vanished' on reload.
    skill_dir = tmp_path / ".agents" / "skills" / "python"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: python\ndescription: d\n---\nbody\n", encoding="utf-8",
    )

    s2 = Session.load(db_path, s1._session_id, flush_sync=True, cwd=tmp_path)
    assert "python" in s2.active_skills
    s2.close()
