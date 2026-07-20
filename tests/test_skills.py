"""Tests for ``cothis.skills`` — discovery and catalog assembly (#30 slice 1).

Covers stories 1-7 of issue #30:

- **Discovery**: scan three layer dirs (``.agents/skills/``,
  ``~/.cothis/skills/``, ``~/.agents/skills/``) for ``SKILL.md``.
- **Parsing**: ``SKILL.md`` frontmatter → ``name`` / ``description``.
- **Layering**: project shadows user-cothis shadows user-agents.
- **Leniency**: missing ``name`` defaults to dir name; missing
  ``description`` or unparseable YAML skips with a logged warning;
  ``name`` ≠ dir name warns but loads.
- **Catalog**: ``format_catalog`` is a pure function of the discovered
  list; produces the ``<available_skills>`` block (or ``None`` when
  empty) with usage instructions + name+description rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import pytest

from cothis.skills import SkillRecord, discover_skills, format_catalog


def _write_skill(
    skills_dir: Path,
    dir_name: str,
    *,
    frontmatter: str,
    body: str = "skill body",
) -> Path:
    """Create ``skills_dir/dir_name/SKILL.md`` with the given frontmatter + body."""
    skill_dir = skills_dir / dir_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8"
    )
    return skill_dir


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------


def test_discover_finds_skill_md_in_project_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``SKILL.md`` in ``.agents/skills/`` is discovered."""
    project_skills = tmp_path / ".agents" / "skills"
    _write_skill(project_skills, "git-pr", frontmatter="name: git-pr\ndescription: Open PRs.")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path / "cothis-home"))

    skills = discover_skills()
    assert len(skills) == 1
    assert skills[0].name == "git-pr"
    assert skills[0].description == "Open PRs."
    assert skills[0].layer == "project"


def test_discover_finds_skills_in_user_cothis_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``~/.cothis/skills/`` is the user-cothis layer."""
    home = tmp_path / "home"
    cothis_home = home / ".cothis"
    user_skills = cothis_home / "skills"
    _write_skill(user_skills, "tdd", frontmatter="name: tdd\ndescription: Test-driven dev.")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COTHIS_HOME", str(cothis_home))
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)

    skills = discover_skills()
    assert len(skills) == 1
    assert skills[0].name == "tdd"
    assert skills[0].layer == "user-cothis"


def test_discover_finds_skills_in_user_agents_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``~/.agents/skills/`` is the user-agents layer (cross-tool convention)."""
    home = tmp_path / "home"
    user_skills = home / ".agents" / "skills"
    _write_skill(user_skills, "code-review", frontmatter="name: code-review\ndescription: Reviews.")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path / "cothis-home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)

    skills = discover_skills()
    assert len(skills) == 1
    assert skills[0].name == "code-review"
    assert skills[0].layer == "user-agents"


def test_project_shadows_user_layer_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """Same ``name`` in project + user → project wins; WARN names the shadow."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COTHIS_HOME", str(home / ".cothis"))
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    _write_skill(home / ".cothis" / "skills", "x", frontmatter="name: x\ndescription: user copy")
    _write_skill(project / ".agents" / "skills", "x", frontmatter="name: x\ndescription: project copy")

    with caplog.at_level("WARNING", logger="cothis.skills"):
        skills = discover_skills()
    assert len(skills) == 1
    assert skills[0].layer == "project"
    assert skills[0].description == "project copy"
    shadow_msgs = [r for r in caplog.records if "shadow" in r.message.lower()]
    assert len(shadow_msgs) == 1
    assert "x" in shadow_msgs[0].message


def test_missing_name_defaults_to_directory_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``name`` absent from frontmatter → directory name is used."""
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path / "ch"))
    _write_skill(
        project / ".agents" / "skills",
        "default-name",
        frontmatter="description: Has no name field.",
    )

    skills = discover_skills()
    assert len(skills) == 1
    assert skills[0].name == "default-name"


def test_name_directory_mismatch_warns_but_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """``name`` ≠ directory name warns once but the skill still loads."""
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path / "ch"))
    _write_skill(
        project / ".agents" / "skills",
        "dir-name",
        frontmatter="name: declared-name\ndescription: Mismatch.",
    )

    with caplog.at_level("WARNING", logger="cothis.skills"):
        skills = discover_skills()
    assert len(skills) == 1
    # The declared name wins (it's what the model would call).
    assert skills[0].name == "declared-name"
    mismatch_msgs = [r for r in caplog.records if "name" in r.message.lower() and "dir" in r.message.lower()]
    assert len(mismatch_msgs) == 1


def test_empty_description_is_skipped_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """``description`` empty / missing → skill skipped (catalog is for the model)."""
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path / "ch"))
    _write_skill(
        project / ".agents" / "skills",
        "no-desc",
        frontmatter="name: no-desc\ndescription:",
    )

    with caplog.at_level("WARNING", logger="cothis.skills"):
        skills = discover_skills()
    assert skills == []
    skip_msgs = [r for r in caplog.records if "no-desc" in r.message]
    assert len(skip_msgs) == 1


def test_unparseable_yaml_is_skipped_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """Broken YAML frontmatter → skill skipped; we never raise."""
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path / "ch"))
    skill_dir = project / ".agents" / "skills" / "broken"
    skill_dir.mkdir(parents=True)
    # Invalid YAML: bad indentation under a mapping.
    (skill_dir / "SKILL.md").write_text(
        "---\nname: broken\n  description: mis-indented\n---\nbody\n",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger="cothis.skills"):
        skills = discover_skills()
    assert skills == []
    skip_msgs = [r for r in caplog.records if "broken" in r.message]
    assert len(skip_msgs) == 1


def test_missing_skill_md_is_silently_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory under ``skills/`` without ``SKILL.md`` is skipped quietly.

    Lets users stage partial skills or hold assets without polluting the catalog.
    """
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agents" / "skills" / "draft").mkdir(parents=True)
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path / "ch"))

    assert discover_skills() == []


def test_discover_no_dirs_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No skill dirs at all → empty list, no error."""
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path / "ch"))
    monkeypatch.setenv("COTHIS_AGENTS_USER_GLOBAL", "0")

    assert discover_skills() == []


def test_discover_records_skill_md_path_and_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``SkillRecord`` carries the source path + body for ``load_skill`` to read later."""
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path / "ch"))
    _write_skill(
        project / ".agents" / "skills",
        "rich",
        frontmatter="name: rich\ndescription: rich body.",
        body="actual skill content",
    )

    skills = discover_skills()
    assert len(skills) == 1
    assert skills[0].path.name == "SKILL.md"
    assert skills[0].body.strip() == "actual skill content"


# ---------------------------------------------------------------------
# Catalog formatting (pure function)
# ---------------------------------------------------------------------


def test_format_catalog_returns_none_when_empty() -> None:
    """No skills discovered → no catalog block (system prompt stays compact)."""
    assert format_catalog([]) is None


def test_format_catalog_returns_tagged_block_with_usage_and_rows() -> None:
    """The catalog block lists each skill's name + description under a usage header."""
    skills = [
        SkillRecord(
            name="git-pr",
            description="Open PRs from branches.",
            path=Path("/x/SKILL.md"),
            body="",
            layer="project",
        ),
        SkillRecord(
            name="tdd",
            description="Drive features through tests.",
            path=Path("/y/SKILL.md"),
            body="",
            layer="user-cothis",
        ),
    ]
    out = format_catalog(skills)
    assert out is not None
    assert "<available_skills>" in out
    assert "</available_skills>" in out
    assert "load_skill" in out
    assert "deactive_skill" in out
    assert "git-pr" in out
    assert "Open PRs from branches." in out
    assert "tdd" in out


def test_format_catalog_is_pure_function() -> None:
    """Calling twice with the same input yields the same output; input list untouched."""
    skills = [
        SkillRecord(
            name="x",
            description="d",
            path=Path("/x/SKILL.md"),
            body="",
            layer="project",
        ),
    ]
    a = format_catalog(skills)
    b = format_catalog(skills)
    assert a == b
    # Input list is not mutated.
    assert len(skills) == 1
