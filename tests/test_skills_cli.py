"""Tests for the ``cothis skills`` CLI subcommand (I13 / roadmap #4 follow-up)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from cothis.cli import app
from cothis.skills import _SKILL_FILE

if TYPE_CHECKING:
    from pathlib import Path


def _make_skill(skills_dir: Path, name: str, description: str = "Test skill.") -> None:
    """Write a real ``SKILL.md`` for ``name`` under ``skills_dir``.

    Mirrors the frontmatter format of ``_make_skill_on_disk`` in
    tests/test_load_skill.py so the real ``discover_skills`` code path
    parses it without mocking.
    """
    d = skills_dir / name
    d.mkdir(parents=True)
    d.joinpath(_SKILL_FILE).write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody.\n",
        encoding="utf-8",
    )


def test_skills_list_shows_project_skill(
    tmp_path: Path, monkeypatch: Any
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _make_skill(project / ".agents" / "skills", "deploy", "Deploy the service.")

    empty_home = tmp_path / "home"
    empty_home.mkdir()
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.delenv("COTHIS_HOME", raising=False)
    monkeypatch.chdir(project)

    runner = CliRunner()
    result = runner.invoke(app, ["skills"])
    assert result.exit_code == 0, result.output
    assert "deploy" in result.output
    assert "Deploy the service." in result.output
    assert "project" in result.output


def test_skills_list_empty(tmp_path: Path, monkeypatch: Any) -> None:
    project = tmp_path / "project"
    project.mkdir()
    empty_home = tmp_path / "home"
    empty_home.mkdir()
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.delenv("COTHIS_HOME", raising=False)
    monkeypatch.chdir(project)

    runner = CliRunner()
    result = runner.invoke(app, ["skills"])
    assert result.exit_code == 0, result.output
    assert "no skills found" in result.output
