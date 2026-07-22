"""Tests for ``deactivate_skill`` tool + in-memory archived set (#167).

Covers Half A of the two-half ``mark_archived`` design:

* ``Session._archived_skills`` runtime set.
* ``is_skill_archived`` + ``_deactivate_skill`` API.
* ``_block_to_row`` honours ``_cothis_state`` marker → ``BlockRow.state``.
* ``Session.append_message`` sets ``_cothis_state='archived'`` on blocks
  whose ``_cothis_skill`` is in the archived set when they're enqueued.
* ``deactivate_skill`` tool: unknown / not-active / repeat / happy path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from cothis.session import Session, _block_to_row
from cothis.session.storage import BlockRow
from cothis.skills import deactivate_skill, discover_skills

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------


def test_new_session_has_empty_archived_set(tmp_path: Path) -> None:
    """A fresh session has no archived skills."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    assert s.archived_skills == frozenset()
    assert not s.is_skill_archived("anything")
    s.close()


def test_deactivate_marks_skill_archived(tmp_path: Path) -> None:
    """``_deactivate_skill`` flips membership in the archived set."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    assert s._deactivate_skill("python") is True
    assert s.is_skill_archived("python")
    s.close()


def test_deactivate_is_idempotent(tmp_path: Path) -> None:
    """Repeat ``_deactivate_skill`` returns False; state unchanged."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    assert s._deactivate_skill("python") is True
    assert s._deactivate_skill("python") is False
    assert s.is_skill_archived("python")
    s.close()


def test_archived_independent_from_active(tmp_path: Path) -> None:
    """Archival and active sets are distinct; deactivating does not activate."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    s._activate_skill("python")
    s._deactivate_skill("python")
    # Both sets record the skill; the row-state read path picks archival.
    assert s.is_skill_active("python")
    assert s.is_skill_archived("python")
    s.close()


# ---------------------------------------------------------------------
# BlockRow mapping
# ---------------------------------------------------------------------


def test_block_to_row_reads_cothis_state() -> None:
    """``_block_to_row`` honours ``_cothis_state`` marker → ``BlockRow.state``."""
    block = {
        "type": "tool_use",
        "id": "t1",
        "name": "load_skill",
        "input": {"name": "python"},
        "_cothis_skill": "python",
        "_cothis_state": "archived",
    }
    row = _block_to_row(
        "s1", seq=0, msg_idx=0, block_idx=0,
        role="assistant", ts="2026-01-01T00:00:00Z",
        block=block,
    )
    assert row.state == "archived"
    assert row.skill == "python"


def test_block_to_row_state_none_when_no_marker() -> None:
    """No ``_cothis_state`` → ``BlockRow.state`` is None (default)."""
    block = {"type": "text", "text": "hi"}
    row = _block_to_row(
        "s1", seq=0, msg_idx=0, block_idx=0,
        role="user", ts="2026-01-01T00:00:00Z",
        block=block,
    )
    assert row.state is None


# ---------------------------------------------------------------------
# Session.append_message writes archived state
# ---------------------------------------------------------------------


def test_append_message_marks_blocks_for_archived_skill(tmp_path: Path) -> None:
    """Block with ``_cothis_skill=X`` enqueued after X archived → state='archived'."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    s._deactivate_skill("python")

    block = {
        "type": "tool_use",
        "id": "t1",
        "name": "load_skill",
        "input": {"name": "python"},
        "_cothis_skill": "python",
    }
    s.append_message("assistant", [block])

    rows = s._storage.load_blocks(s._session_id)
    assert len(rows) == 1
    assert rows[0].state == "archived"
    assert rows[0].skill == "python"
    s.close()


def test_append_message_leaves_unarchived_skill_blocks_alone(tmp_path: Path) -> None:
    """Block for a non-archived skill → ``state`` is None."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    # python is active (not archived)
    s._activate_skill("python")

    block = {
        "type": "tool_use",
        "id": "t1",
        "name": "load_skill",
        "input": {"name": "python"},
        "_cothis_skill": "python",
    }
    s.append_message("assistant", [block])

    rows = s._storage.load_blocks(s._session_id)
    assert rows[0].state is None
    s.close()


def test_append_message_no_skill_marker_unaffected(tmp_path: Path) -> None:
    """Block without ``_cothis_skill`` → ``state`` stays None even if some
    other skill is archived."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    s._deactivate_skill("python")

    s.append_message("user", [{"type": "text", "text": "hi"}])
    rows = s._storage.load_blocks(s._session_id)
    assert rows[0].state is None
    s.close()


def test_append_message_block_enqueued_before_archival_is_caught_by_half_b(
    tmp_path: Path,
) -> None:
    """A block enqueued before archival is caught by Half B's queued UPDATE.

    Half A (#167) only marks blocks directly at enqueue time, so the
    pre-archival enqueue leaves ``state=None``. The queued UPDATE posted
    by ``_deactivate_skill`` (#168) catches historical + in-flight rows
    regardless of when they were written.
    """
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    block = {
        "type": "tool_use",
        "id": "t1",
        "name": "load_skill",
        "input": {"name": "python"},
        "_cothis_skill": "python",
    }
    s.append_message("assistant", [block])

    # Before deactivate: Half A didn't mark (enqueue was before archival).
    pre = s._storage.load_blocks(s._session_id)
    assert pre[0].state is None

    # After deactivate: Half B catches the historical row.
    s._deactivate_skill("python")
    rows = s._storage.load_blocks(s._session_id)
    assert rows[0].state == "archived"
    s.close()


# ---------------------------------------------------------------------
# deactivate_skill tool
# ---------------------------------------------------------------------


class _FakeSession:
    """Minimal Session stand-in: tracks active + archived sets."""

    def __init__(self) -> None:
        self._active: set[str] = set()
        self._archived: set[str] = set()

    def is_skill_active(self, name: str) -> bool:
        return name in self._active

    def is_skill_archived(self, name: str) -> bool:
        return name in self._archived

    def _activate_skill(self, name: str) -> bool:
        if name in self._active:
            return False
        self._active.add(name)
        return True

    def _deactivate_skill(self, name: str) -> bool:
        if name in self._archived:
            return False
        self._archived.add(name)
        return True


def test_deactivate_skill_unknown_name_returns_error() -> None:
    """Unknown skill name → error listing available skills."""
    session = _FakeSession()
    with patch("cothis.skills.discover_skills", return_value=[]):
        result = deactivate_skill(name="python", _session=session)
    assert "unknown" in result.lower() or "error" in result.lower()


def test_deactivate_skill_not_active_returns_notice() -> None:
    """Known skill but not active → 'not active' notice, no state change."""
    session = _FakeSession()
    skill = _make_skill("python")
    with patch("cothis.skills.discover_skills", return_value=[skill]):
        result = deactivate_skill(name="python", _session=session)
    assert "not" in result.lower() and "active" in result.lower()
    assert not session.is_skill_archived("python")


def test_deactivate_skill_already_archived_is_noop() -> None:
    """Already-archived skill → safe no-op, idempotent."""
    session = _FakeSession()
    session._activate_skill("python")
    session._deactivate_skill("python")
    skill = _make_skill("python")
    with patch("cothis.skills.discover_skills", return_value=[skill]):
        result = deactivate_skill(name="python", _session=session)
    assert "already" in result.lower() or "no-op" in result.lower() or "archived" in result.lower()


def test_deactivate_skill_happy_path() -> None:
    """Active + not archived → deactivates, confirmation message."""
    session = _FakeSession()
    session._activate_skill("python")
    skill = _make_skill("python")
    with patch("cothis.skills.discover_skills", return_value=[skill]):
        result = deactivate_skill(name="python", _session=session)
    assert session.is_skill_archived("python")
    assert "python" in result
    assert "deactivat" in result.lower() or "archiv" in result.lower()


def test_deactivate_skill_no_session_returns_error() -> None:
    """No session attached → error (cannot record archival state)."""
    with patch("cothis.skills.discover_skills", return_value=[_make_skill("python")]):
        result = deactivate_skill(name="python", _session=None)
    assert "error" in result.lower() or "no session" in result.lower()


def test_deactivate_skill_handler_registered_as_tool() -> None:
    """``deactivate_skill`` carries skill_marker + session injection."""
    # The @tool decorator sets these attributes on the wrapper.
    assert getattr(deactivate_skill, "_skill_marker", False) is True
    assert getattr(deactivate_skill, "_inject_session", False) is True


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _make_skill(name: str) -> Any:
    """Build a minimal Skill instance for tests."""
    from pathlib import Path as _Path

    from cothis.skills import Skill
    return Skill(
        name=name,
        description=f"{name} description",
        body=f"{name} body",
        source=_Path(f"/tmp/{name}/SKILL.md"),
    )
