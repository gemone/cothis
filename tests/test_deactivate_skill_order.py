"""Tests for ``deactivate_skill`` order-of-checks (#180).

A skill that was activated earlier but is no longer on disk must
still be deactivatable from session state. The catalog check
should fire only when the name is neither active nor archived
(truly unknown to the session).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from cothis.skills import deactivate_skill

if TYPE_CHECKING:
    from pathlib import Path


class _FakeSession:
    """Minimal Session stand-in."""

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


def _make_skill(name: str) -> Any:
    from pathlib import Path as _Path

    from cothis.skills import Skill
    return Skill(
        name=name, description=f"{name} d", body=f"{name} b",
        source=_Path(f"/tmp/{name}/SKILL.md"), deactivation="delete",
    )


# ---------------------------------------------------------------------
# The bug: active skill removed from disk can still be deactivated
# ---------------------------------------------------------------------


def test_active_skill_not_on_disk_can_be_deactivated() -> None:
    """Skill activated earlier, then removed from disk, still deactivates."""
    session = _FakeSession()
    session._activate_skill("temp")
    # Discovery returns empty — the skill is no longer on disk.
    with patch("cothis.skills.discover_skills", return_value=[]):
        result = deactivate_skill(name="temp", _session=session)
    assert session.is_skill_archived("temp")
    assert "archived" in result.lower() or "deactivat" in result.lower()


def test_active_skill_not_on_disk_does_not_return_unknown_error() -> None:
    """Regression: the misleading 'unknown skill' error must not fire."""
    session = _FakeSession()
    session._activate_skill("temp")
    with patch("cothis.skills.discover_skills", return_value=[]):
        result = deactivate_skill(name="temp", _session=session)
    assert "unknown skill" not in result.lower()


def test_active_skill_not_on_disk_archival_propagates_to_state() -> None:
    """Deactivation of vanished-from-disk active skill sets archived state."""
    session = _FakeSession()
    session._activate_skill("temp")
    with patch("cothis.skills.discover_skills", return_value=[]):
        deactivate_skill(name="temp", _session=session)
    assert session.is_skill_archived("temp")


# ---------------------------------------------------------------------
# Order of checks is unchanged for the other paths
# ---------------------------------------------------------------------


def test_already_archived_skill_returns_notice() -> None:
    """Already-archived skill → safe no-op (regardless of catalog)."""
    session = _FakeSession()
    session._activate_skill("temp")
    session._deactivate_skill("temp")
    with patch("cothis.skills.discover_skills", return_value=[]):
        result = deactivate_skill(name="temp", _session=session)
    assert "already" in result.lower() or "archived" in result.lower()


def test_unknown_skill_truly_unknown_returns_error() -> None:
    """Skill neither active nor archived nor in catalog → unknown error."""
    session = _FakeSession()
    with patch("cothis.skills.discover_skills", return_value=[]):
        result = deactivate_skill(name="never_heard", _session=session)
    assert "unknown" in result.lower() or "error" in result.lower()
    assert "never_heard" in result


def test_in_catalog_but_not_active_returns_not_active_notice() -> None:
    """Skill in catalog but never activated → 'not active' notice."""
    session = _FakeSession()
    skill = _make_skill("available")
    with patch("cothis.skills.discover_skills", return_value=[skill]):
        result = deactivate_skill(name="available", _session=session)
    assert "not" in result.lower() and "active" in result.lower()
    assert not session.is_skill_archived("available")


def test_no_session_returns_error() -> None:
    """No session attached → error (cannot record archival state)."""
    with patch("cothis.skills.discover_skills", return_value=[]):
        result = deactivate_skill(name="temp", _session=None)
    assert "error" in result.lower() or "no session" in result.lower()


def test_deactivate_skill_on_disk_normal_path() -> None:
    """Active + on-disk skill → normal deactivation (unchanged path)."""
    session = _FakeSession()
    session._activate_skill("python")
    skill = _make_skill("python")
    with patch("cothis.skills.discover_skills", return_value=[skill]):
        result = deactivate_skill(name="python", _session=session)
    assert session.is_skill_archived("python")
    assert "python" in result
