"""Regression: the skill tools reach the model on every production path (#I12).

``cothis.skills`` ships two model-facing tools — ``load_skill`` and
``deactivate_skill`` — for the Agent to activate and retire skills at runtime.
They are registered as members of the builtin tool layer
(``cothis.tools.builtins.TOOLS``), so ``discover_tools`` surfaces them on all
three production Agent-construction sites (``ask``, ``acp``, ``--session``
worker) with no per-site wiring. These tests pin that registration end-to-end
and guard the reserved-name shadow contract a project tool still enjoys.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def test_skill_tools_reach_production_tool_map(tmp_path: Any) -> None:
    """``load_skill`` + ``deactivate_skill`` are in the builtin layer, so they
    flow through ``discover_tools`` into the Agent dispatch map and wire
    schemas — the regression that would have caught the registration gap.

    Empty user/project dirs are intentional: the builtin layer loads
    regardless of the scanned dirs, mirroring the production call
    ``discover_tools(_PROJECT_TOOLS_DIR, _user_tools_dir())`` for a project
    that declares no tools of its own.
    """
    from cothis.agent import Agent, _sanitize_tool_name
    from cothis.tools.core import discover_tools

    tools = discover_tools(tmp_path, tmp_path)
    names = {t.__name__ for t in tools}
    assert "load_skill" in names
    assert "deactivate_skill" in names

    # End-to-end through the construction path every production mode uses.
    agent = Agent(model="x", provider="openrouter", tools=tools)
    for skill in ("load_skill", "deactivate_skill"):
        assert _sanitize_tool_name(skill) in agent._tool_map
    schema_by_name = {s["name"]: s for s in (agent._tool_schemas() or [])}
    for skill in ("load_skill", "deactivate_skill"):
        assert skill in schema_by_name
        assert (schema_by_name[skill].get("description") or "").strip()


def test_project_tool_shadows_skill_builtin(
    tmp_path: Any, caplog: Any
) -> None:
    """A project-declared tool named ``load_skill`` shadows the skill builtin
    with the existing WARNING and wins — the reserved name is NOT silently
    hardened. Resolution direction matches the fs builtins today.
    """
    from cothis.tools.core import discover_tools

    # Project layer (highest precedence) declares a tool named ``load_skill``.
    (tmp_path / "override.py").write_text(
        'from cothis.tools import tool\n\n@tool("load_skill")\n'
        'def load_skill(name: str) -> str:\n    """Project override."""\n'
        '    return "project-wins"\n',
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="cothis.tools"):
        tools = discover_tools(tmp_path, tmp_path / "empty_user")
    by_name = {t.__name__: t for t in tools}
    assert "load_skill" in by_name
    # The project tool won: its source is the override file, not "builtins".
    # ``_source`` is set by ``load_tools_from_layer`` (absent on the bare
    # ``Tool`` protocol), so read it defensively — same pattern
    # ``discover_tools`` uses internally.
    src = getattr(by_name["load_skill"], "_source", None)
    assert src is not None
    assert "override.py" in str(src)
    # The existing shadow WARNING fired, naming the builtin being shadowed.
    shadow_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.name == "cothis.tools" and "shadows" in r.getMessage()
    ]
    assert any("load_skill" in m for m in shadow_msgs)


def test_shipped_fixture_skill_is_discoverable(tmp_path: Any) -> None:
    """The shipped sample skill under ``tests/fixtures`` parses cleanly and is
    reachable through the real three-layer discovery path (project layer). The
    other two layers point at empty dirs so the test is hermetic — it does not
    depend on skills installed in ``$COTHIS_HOME`` or ``~/.agents``."""
    from cothis.skills import discover_skills

    fixtures_cwd = Path(__file__).parent / "fixtures"
    skills = discover_skills(
        cwd=fixtures_cwd,
        cothis_home=tmp_path / "cothis_home",
        user_agents=tmp_path / "user_agents",
    )
    assert [s.name for s in skills] == ["git-commit"]
    skill = skills[0]
    assert skill.description.strip()
    assert skill.body.strip()
    assert skill.deactivation == "delete"
