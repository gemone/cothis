"""Tests for ``deactivation: summarize`` declaration fallback (#170).

A skill's SKILL.md may declare ``deactivation: summarize`` (intent:
produce a summary before archiving). The Summarize strategy is
deferred — for now the declaration is parsed + recorded, and
``deactivate_skill`` falls back to Delete with a logged WARNING.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from cothis.skills import (
    Skill,
    _parse_skill_md,
    deactivate_skill,
    discover_skills,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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


def _make_skill(name: str, *, deactivation: str = "delete") -> Skill:
    """Build a Skill with the given deactivation declaration."""
    from pathlib import Path as _Path
    return Skill(
        name=name,
        description=f"{name} description",
        body=f"{name} body",
        source=_Path(f"/tmp/{name}/SKILL.md"),
        deactivation=deactivation,
    )


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------


def test_skill_default_deactivation_is_delete(tmp_path: Path) -> None:
    """SKILL.md with no ``deactivation:`` field defaults to 'delete'."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: my-skill\ndescription: d\n---\nbody\n",
        encoding="utf-8",
    )
    skill = _parse_skill_md(skill_dir / "SKILL.md")
    assert skill is not None
    assert skill.deactivation == "delete"


def test_skill_parses_deactivation_delete(tmp_path: Path) -> None:
    """Explicit ``deactivation: delete`` parses to 'delete'."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: skill\ndescription: d\ndeactivation: delete\n---\nbody\n",
        encoding="utf-8",
    )
    skill = _parse_skill_md(skill_dir / "SKILL.md")
    assert skill is not None
    assert skill.deactivation == "delete"


def test_skill_parses_deactivation_summarize(tmp_path: Path) -> None:
    """``deactivation: summarize`` parses to 'summarize'."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: skill\ndescription: d\ndeactivation: summarize\n---\nbody\n",
        encoding="utf-8",
    )
    skill = _parse_skill_md(skill_dir / "SKILL.md")
    assert skill is not None
    assert skill.deactivation == "summarize"


def test_skill_unknown_deactivation_normalizes_to_delete(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown value (e.g., 'rename') → default 'delete' + warning."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: skill\ndescription: d\ndeactivation: rename\n---\nbody\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="cothis.skills"):
        skill = _parse_skill_md(skill_dir / "SKILL.md")
    assert skill is not None
    assert skill.deactivation == "delete"
    assert any(
        "deactivation" in r.message.lower() and "rename" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------
# deactivate_skill fallback behavior
# ---------------------------------------------------------------------


def test_deactivate_skill_summarize_logs_warning_and_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``deactivation: summarize`` → WARNING + Delete fallback (state mutated)."""
    session = _FakeSession()
    session._activate_skill("python")
    skill = _make_skill("python", deactivation="summarize")
    with patch("cothis.skills.discover_skills", return_value=[skill]):
        with caplog.at_level(logging.WARNING, logger="cothis.skills"):
            result = deactivate_skill(name="python", _session=session)
    # Skill was deactivated (Delete fallback).
    assert session.is_skill_archived("python")
    # WARNING logged.
    assert any(
        "summarize" in r.message.lower() and "python" in r.message.lower()
        for r in caplog.records
    )
    # Result mentions archiving (Delete fallback completed).
    assert "archived" in result.lower() or "deactivat" in result.lower()


def test_deactivate_skill_delete_no_extra_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``deactivation: delete`` → no extra warning beyond the normal path."""
    session = _FakeSession()
    session._activate_skill("python")
    skill = _make_skill("python", deactivation="delete")
    with patch("cothis.skills.discover_skills", return_value=[skill]):
        with caplog.at_level(logging.WARNING, logger="cothis.skills"):
            result = deactivate_skill(name="python", _session=session)
    assert session.is_skill_archived("python")
    # No warning about summarize fallback.
    assert not any(
        "summarize" in r.message.lower() and "fallback" in r.message.lower()
        for r in caplog.records
    )


def test_deactivate_skill_default_no_extra_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No declaration → no summarize warning (default 'delete')."""
    session = _FakeSession()
    session._activate_skill("python")
    skill_with_default = _make_skill("python", deactivation="delete")
    with patch("cothis.skills.discover_skills", return_value=[skill_with_default]):
        with caplog.at_level(logging.WARNING, logger="cothis.skills"):
            deactivate_skill(name="python", _session=session)
    assert session.is_skill_archived("python")
    assert not any(
        "summarize" in r.message.lower() and "fallback" in r.message.lower()
        for r in caplog.records
    )


def test_deactivate_skill_summarize_result_mentions_fallback() -> None:
    """The return value tells the user about the fallback (so they notice)."""
    session = _FakeSession()
    session._activate_skill("python")
    skill = _make_skill("python", deactivation="summarize")
    with patch("cothis.skills.discover_skills", return_value=[skill]):
        result = deactivate_skill(name="python", _session=session)
    assert session.is_skill_archived("python")
    # Result should mention the skill + that summarize isn't implemented.
    assert "python" in result
    assert "summarize" in result.lower() or "fallback" in result.lower()


# ---------------------------------------------------------------------
# End-to-end via discover_skills
# ---------------------------------------------------------------------


def test_discover_skills_propagates_deactivation_field(
    tmp_path: Path,
) -> None:
    """End-to-end: deactivation field survives discover_skills."""
    skills_dir = tmp_path / ".agents" / "skills"
    summary_skill = skills_dir / "summary-skill"
    summary_skill.mkdir(parents=True)
    summary_skill.joinpath("SKILL.md").write_text(
        "---\nname: summary-skill\ndescription: d\ndeactivation: summarize\n"
        "---\nbody\n",
        encoding="utf-8",
    )
    delete_skill = skills_dir / "delete-skill"
    delete_skill.mkdir(parents=True)
    delete_skill.joinpath("SKILL.md").write_text(
        "---\nname: delete-skill\ndescription: d\n---\nbody\n",
        encoding="utf-8",
    )

    skills = discover_skills(
        tmp_path, cothis_home=tmp_path / "ch", user_agents=tmp_path / "ua",
    )
    by_name = {s.name: s for s in skills}
    assert by_name["summary-skill"].deactivation == "summarize"
    assert by_name["delete-skill"].deactivation == "delete"
