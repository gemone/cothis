"""Tests for ``/reload-skills`` vanished-archival (#74 final AC).

When ``/reload-skills`` re-runs discovery, any active skill that's no
longer on disk (vanished) is archived with a WARNING. Surviving
active skills keep their state. The archival runs through
``_deactivate_skill`` so it composes with Half A + Half B + the
in-memory walk.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import cothis.slash as slash_mod
from cothis.skills import register_slash_commands
from cothis.slash import SlashContext, dispatch

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    slash_mod._entries.clear()


@pytest.fixture
def _registered() -> None:
    register_slash_commands()


def _skill(name: str) -> object:
    """Build a minimal Skill-like object for tests."""
    from pathlib import Path as _Path

    from cothis.skills import Skill
    return Skill(
        name=name,
        description=f"{name} description",
        body=f"{name} body",
        source=_Path(f"/tmp/{name}/SKILL.md"),
        deactivation="delete",
    )


class _FakeSession:
    """Session stand-in: tracks active + archived sets + cwd."""

    def __init__(self, cwd: Path) -> None:
        self._cwd = cwd
        self._active: set[str] = set()
        self._archived: set[str] = set()

    @property
    def cwd(self) -> Path:
        return self._cwd

    @property
    def active_skills(self) -> frozenset[str]:
        return frozenset(self._active)

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


# ---------------------------------------------------------------------
# Vanished-skill archival
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vanished_active_skill_is_archived(
    tmp_path: Path, _registered: None, caplog: pytest.LogCaptureFixture,
) -> None:
    """An active skill missing from the new discovery → archived + WARNING."""
    session = _FakeSession(tmp_path)
    session._activate_skill("gone")
    # Discovery returns a different skill — "gone" has vanished.
    with patch("cothis.skills.discover_skills", return_value=[_skill("other")]):
        with caplog.at_level(logging.WARNING, logger="cothis.skills"):
            result = await dispatch(
                "/reload-skills", ctx=SlashContext(session=session),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            )
    assert session.is_skill_archived("gone")
    assert any(
        "gone" in r.message and (
            "vanished" in r.message.lower() or "archiv" in r.message.lower()
        )
        for r in caplog.records
    )
    assert result is not None
    assert "gone" in result


@pytest.mark.asyncio
async def test_surviving_active_skill_keeps_state(
    tmp_path: Path, _registered: None,
) -> None:
    """An active skill still in the new discovery → stays active, not archived."""
    session = _FakeSession(tmp_path)
    session._activate_skill("python")
    with patch("cothis.skills.discover_skills", return_value=[_skill("python")]):
        await dispatch(
            "/reload-skills", ctx=SlashContext(session=session),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )
    assert session.is_skill_active("python")
    assert not session.is_skill_archived("python")


@pytest.mark.asyncio
async def test_mixed_vanished_and_surviving(
    tmp_path: Path, _registered: None,
) -> None:
    """Multiple active skills: some vanish, some survive."""
    session = _FakeSession(tmp_path)
    session._activate_skill("python")
    session._activate_skill("gone")
    session._activate_skill("also-gone")
    with patch(
        "cothis.skills.discover_skills",
        return_value=[_skill("python")],  # only python survives
    ):
        result = await dispatch(
            "/reload-skills", ctx=SlashContext(session=session),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )
    assert session.is_skill_active("python")
    assert not session.is_skill_archived("python")
    assert session.is_skill_archived("gone")
    assert session.is_skill_archived("also-gone")
    # Summary mentions both vanished skills.
    assert result is not None
    assert "gone" in result
    assert "also-gone" in result


@pytest.mark.asyncio
async def test_no_active_skills_no_archival(
    tmp_path: Path, _registered: None, caplog: pytest.LogCaptureFixture,
) -> None:
    """No active skills → no archival, no warnings."""
    session = _FakeSession(tmp_path)
    with patch("cothis.skills.discover_skills", return_value=[_skill("python")]):
        with caplog.at_level(logging.WARNING, logger="cothis.skills"):
            result = await dispatch(
                "/reload-skills", ctx=SlashContext(session=session),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            )
    assert not session._archived
    assert result is not None
    assert "python" in result  # catalog summary still listed


@pytest.mark.asyncio
async def test_vanished_archival_composes_with_half_b(
    tmp_path: Path, _registered: None,
) -> None:
    """Vanished archival runs through _deactivate_skill (Half A+B+walk)."""
    # Use a real Session so the full mechanism fires.
    from cothis.session import Session
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    s._activate_skill("python")
    # Write a python-tagged block so Half B has something to archive.
    s.append_message("assistant", [{
        "type": "tool_use", "id": "t1", "name": "load_skill",
        "input": {"name": "python"}, "_cothis_skill": "python",
    }])

    # Discovery returns empty — python has vanished.
    with patch("cothis.skills.discover_skills", return_value=[]):
        await dispatch("/reload-skills", ctx=SlashContext(session=s))

    # _deactivate_skill fired: archival propagated to SQLite.
    assert s.is_skill_archived("python")
    rows = s._storage.load_blocks(s._session_id)
    python_rows = [r for r in rows if r.skill == "python"]
    assert len(python_rows) == 1
    assert python_rows[0].state == "archived"
    s.close()


@pytest.mark.asyncio
async def test_no_session_no_archival(
    _registered: None, caplog: pytest.LogCaptureFixture,
) -> None:
    """No session attached → handler still reports discovery, no archival."""
    with patch("cothis.skills.discover_skills", return_value=[_skill("python")]):
        with caplog.at_level(logging.WARNING, logger="cothis.skills"):
            result = await dispatch("/reload-skills", ctx=SlashContext())
    assert result is not None
    assert "python" in result
    # No vanished-archival warnings (no session to check against).
    assert not any(
        "vanished" in r.message.lower() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_vanished_archival_idempotent(
    tmp_path: Path, _registered: None,
) -> None:
    """Re-running reload after archival: the skill is already archived,
    no second WARNING burst."""
    session = _FakeSession(tmp_path)
    session._activate_skill("gone")
    # First reload: archive.
    with patch("cothis.skills.discover_skills", return_value=[]):
        await dispatch("/reload-skills", ctx=SlashContext(session=session))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    assert session.is_skill_archived("gone")

    # Second reload: gone is no longer active (archived), so it's not
    # in the active-vs-discovered diff. No further state change.
    with patch("cothis.skills.discover_skills", return_value=[]):
        await dispatch("/reload-skills", ctx=SlashContext(session=session))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    # Still archived (state unchanged).
    assert session.is_skill_archived("gone")
