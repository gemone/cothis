"""Tests for ``cothis.skills`` — discovery + catalog (#68)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cothis.skills import Skill, discover_skills, format_catalog

if TYPE_CHECKING:
    import pytest


def _make_skill(
    skills_dir: Path, name: str, *, description: str = "A skill.",
    body: str = "Instructions here.", skill_name: str | None = None,
) -> Path:
    """Create a skill directory + SKILL.md; return the directory."""
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    front_name = skill_name if skill_name is not None else name
    d.joinpath("SKILL.md").write_text(
        f"---\nname: {front_name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )
    return d


# ---------------------------------------------------------------------
# Layer shadowing
# ---------------------------------------------------------------------


def test_project_shadows_user_cothis(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Project layer wins over user-cothis on name collision."""
    project = tmp_path / "project"
    user_cothis = tmp_path / "cothis_home"
    user_agents = tmp_path / "user_agents"
    _make_skill(project / ".agents" / "skills", "deploy", description="proj")
    _make_skill(user_cothis / "skills", "deploy", description="user")

    with caplog.at_level(logging.WARNING, logger="cothis.skills"):
        skills = discover_skills(
            project, cothis_home=user_cothis, user_agents=user_agents,
        )
    by_name = {s.name: s for s in skills}
    assert by_name["deploy"].description == "proj"
    assert "shadow" in " ".join(r.message for r in caplog.records)


def test_three_layer_shadow_order(
    tmp_path: Path,
) -> None:
    """Project > user-cothis > user-agents precedence."""
    project = tmp_path / "project"
    cothis_home = tmp_path / "cothis_home"
    user_agents = tmp_path / "user_agents"
    _make_skill(project / ".agents" / "skills", "x", description="p")
    _make_skill(cothis_home / "skills", "x", description="c")
    _make_skill(user_agents / "skills", "x", description="a")

    skills = discover_skills(
        project, cothis_home=cothis_home, user_agents=user_agents,
    )
    assert len(skills) == 1
    assert skills[0].description == "p"


def test_no_shadow_different_names(
    tmp_path: Path,
) -> None:
    """Different skill names in different layers all load."""
    project = tmp_path / "project"
    cothis_home = tmp_path / "cothis_home"
    _make_skill(project / ".agents" / "skills", "alpha")
    _make_skill(cothis_home / "skills", "beta")
    skills = discover_skills(
        project, cothis_home=cothis_home,
        user_agents=tmp_path / "no_agents",
    )
    assert {s.name for s in skills} == {"alpha", "beta"}


# ---------------------------------------------------------------------
# Lenient parsing
# ---------------------------------------------------------------------


def test_missing_name_defaults_to_directory(
    tmp_path: Path,
) -> None:
    """SKILL.md without ``name:`` field loads with directory name."""
    skills_dir = tmp_path / ".agents" / "skills"
    d = skills_dir / "my-skill"
    d.mkdir(parents=True)
    d.joinpath("SKILL.md").write_text(
        "---\ndescription: has desc\n---\nbody\n", encoding="utf-8",
    )
    skills = discover_skills(
        tmp_path, cothis_home=tmp_path / "ch", user_agents=tmp_path / "ua",
    )
    assert len(skills) == 1
    assert skills[0].name == "my-skill"


def test_empty_description_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty ``description`` → skip + log."""
    skills_dir = tmp_path / ".agents" / "skills"
    d = skills_dir / "bad"
    d.mkdir(parents=True)
    d.joinpath("SKILL.md").write_text(
        "---\nname: bad\ndescription: \"\"\n---\nbody\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="cothis.skills"):
        skills = discover_skills(
            tmp_path, cothis_home=tmp_path / "ch", user_agents=tmp_path / "ua",
        )
    assert skills == []


def test_broken_yaml_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Broken YAML frontmatter → skip + log."""
    skills_dir = tmp_path / ".agents" / "skills"
    d = skills_dir / "broken"
    d.mkdir(parents=True)
    d.joinpath("SKILL.md").write_text(
        "---\nname: [unclosed\n---\nbody\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="cothis.skills"):
        skills = discover_skills(
            tmp_path, cothis_home=tmp_path / "ch", user_agents=tmp_path / "ua",
        )
    assert skills == []


def test_no_frontmatter_skipped(
    tmp_path: Path,
) -> None:
    """SKILL.md without ``---`` frontmatter → skip."""
    skills_dir = tmp_path / ".agents" / "skills"
    d = skills_dir / "nofm"
    d.mkdir(parents=True)
    d.joinpath("SKILL.md").write_text("just markdown\n", encoding="utf-8")
    skills = discover_skills(
        tmp_path, cothis_home=tmp_path / "ch", user_agents=tmp_path / "ua",
    )
    assert skills == []


def test_name_mismatch_warns_but_loads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """``name`` ≠ directory → warn + load with declared name."""
    _make_skill(
        tmp_path / ".agents" / "skills", "dir-name",
        skill_name="declared-name",
    )
    with caplog.at_level(logging.WARNING, logger="cothis.skills"):
        skills = discover_skills(
            tmp_path, cothis_home=tmp_path / "ch", user_agents=tmp_path / "ua",
        )
    assert len(skills) == 1
    assert skills[0].name == "declared-name"
    assert "≠ directory" in " ".join(r.message for r in caplog.records)


# ---------------------------------------------------------------------
# format_catalog
# ---------------------------------------------------------------------


def test_catalog_empty_returns_none() -> None:
    assert format_catalog([]) is None


def test_catalog_single_skill() -> None:
    skill = Skill(name="deploy", description="Deploy stuff", body="x", source=Path())
    out = format_catalog([skill])
    assert out is not None
    assert "<available_skills>" in out
    assert "deploy" in out
    assert "Deploy stuff" in out


def test_catalog_many_skills_sorted() -> None:
    """Catalog renders in the order given (discover_skills already sorts)."""
    skills = [
        Skill(name="alpha", description="a", body="", source=Path()),
        Skill(name="zebra", description="z", body="", source=Path()),
    ]
    out = format_catalog(skills)
    assert out is not None
    lines = out.strip().splitlines()
    assert "alpha" in lines[1]
    assert "zebra" in lines[2]


def test_catalog_is_pure_function() -> None:
    """Same input always produces same output; no side effects."""
    skills = [Skill(name="x", description="y", body="z", source=Path())]
    a = format_catalog(skills)
    b = format_catalog(skills)
    assert a == b


# ---------------------------------------------------------------------
# Integration: discover_skills end-to-end
# ---------------------------------------------------------------------


def test_discover_no_skills_returns_empty(tmp_path: Path) -> None:
    """No skills directory → empty list."""
    skills = discover_skills(
        tmp_path, cothis_home=tmp_path / "ch", user_agents=tmp_path / "ua",
    )
    assert skills == []


def test_discover_ignores_non_directory_entries(
    tmp_path: Path,
) -> None:
    """Files in skills/ are ignored; only directories count."""
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)
    skills_dir.joinpath("README.md").write_text("not a skill dir")
    skills = discover_skills(
        tmp_path, cothis_home=tmp_path / "ch", user_agents=tmp_path / "ua",
    )
    assert skills == []
