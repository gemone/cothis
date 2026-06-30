"""Tests for ``cothis.cli`` formatting helpers and tool discovery.

``_format_tool_call`` is the only pure formatting function in the CLI module
today; it's worth locking down because its output format is what users read
to debug multi-step agent turns, and the ``repr`` convention (strings quoted,
numbers not) is a deliberate choice.

``_all_tools`` tests cover the two-layer discovery model (project-local +
user-global) and the cross-layer ceiling (raises until #10/#11 land).
"""

from __future__ import annotations

from typing import Any

from cothis.agent import ToolCallEvent
from cothis.cli import _all_tools, _format_tool_call


def test_format_string_argument_quoted() -> None:
    event = ToolCallEvent(name="fs.read", arguments={"path": "/tmp/x"})
    assert _format_tool_call(event) == "calling fs.read(path='/tmp/x')"


def test_format_multiple_arguments() -> None:
    event = ToolCallEvent(
        name="fs.write",
        arguments={"path": "/tmp/out.txt", "content": "hello"},
    )
    out = _format_tool_call(event)
    # dict iteration order is insertion order; assert both pieces present.
    assert "path='/tmp/out.txt'" in out
    assert "content='hello'" in out
    assert out.startswith("calling fs.write(")
    assert out.endswith(")")


def test_format_integer_argument_not_quoted() -> None:
    event = ToolCallEvent(name="add", arguments={"a": 2, "b": 3})
    out = _format_tool_call(event)
    assert "a=2" in out
    assert "b=3" in out
    # repr distinguishes: 2 not '2'
    assert "a='2'" not in out


def test_format_no_arguments() -> None:
    event = ToolCallEvent(name="noop", arguments={})
    assert _format_tool_call(event) == "calling noop()"


def test_format_string_with_special_chars_repr_escaped() -> None:
    # repr keeps newlines / quotes visible, preventing garbled display.
    event = ToolCallEvent(
        name="fs.write",
        arguments={"content": 'line1\nline2 "quoted"'},
    )
    out = _format_tool_call(event)
    assert "content='line1\\nline2 \"quoted\"'" in out


# --------------------------------------------------------------------
# _all_tools: two-layer discovery + cross-layer shadow semantics (#9, #10, #11)
# --------------------------------------------------------------------


def test_all_tools_user_global_absent_no_error(tmp_path: Any) -> None:
    """Missing user-global dir is the common case — must not error."""
    project = tmp_path / "project"
    project.mkdir()
    user = tmp_path / "nonexistent"

    tools = _all_tools(project, user)
    # Only builtins load (fs.read, fs.dir, fs.write).
    names = {t.__name__ for t in tools}
    assert "fs.read" in names
    assert "fs.write" in names


def test_all_tools_user_global_loads_tools(tmp_path: Any) -> None:
    """Tools from ``~/.config/cothis/tools/`` appear in the tool list."""
    project = tmp_path / "project"
    project.mkdir()
    user = tmp_path / "user"
    user.mkdir()
    (user / "hello.yaml").write_text(
        'name: user.hello\ncommand: ["echo", "hi"]\n', encoding="utf-8"
    )

    tools = _all_tools(project, user)
    names = {t.__name__ for t in tools}
    assert "user.hello" in names


def test_all_tools_project_local_loads_tools(tmp_path: Any) -> None:
    """Tools from ``.agents/tools/`` appear in the tool list."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "deploy.yaml").write_text(
        'name: proj.deploy\ncommand: ["echo", "deploy"]\n', encoding="utf-8"
    )
    user = tmp_path / "nonexistent"

    tools = _all_tools(project, user)
    names = {t.__name__ for t in tools}
    assert "proj.deploy" in names


def test_shadow_project_local_wins(tmp_path: Any) -> None:
    """Project-local tool with same name as user-global shadows it (#10)."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "dup.yaml").write_text(
        'name: shared.tool\ncommand: ["echo", "proj"]\n', encoding="utf-8"
    )
    user = tmp_path / "user"
    user.mkdir()
    (user / "dup.yaml").write_text(
        'name: shared.tool\ncommand: ["echo", "user"]\n', encoding="utf-8"
    )

    tools = _all_tools(project, user)
    by_name = {t.__name__: t for t in tools}
    assert "shared.tool" in by_name
    # Project-local won — its output is "proj", not "user".
    assert by_name["shared.tool"]() == "proj\n"


def test_shadow_custom_overrides_builtin(tmp_path: Any) -> None:
    """Custom tool with same name as a builtin shadows it (#11)."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "override.yaml").write_text(
        'name: fs.read\ncommand: ["echo", "fake"]\n', encoding="utf-8"
    )
    user = tmp_path / "nonexistent"

    tools = _all_tools(project, user)
    by_name = {t.__name__: t for t in tools}
    assert "fs.read" in by_name
    # Custom won — its output is "fake", not the builtin fs.read behavior.
    assert by_name["fs.read"]() == "fake\n"


def test_shadow_emits_warning_both_layers(tmp_path: Any, caplog: Any) -> None:
    """Shadow emits a WARNING naming both layers + source paths (#10, #11)."""
    import logging

    project = tmp_path / "project"
    project.mkdir()
    (project / "fs_read.yaml").write_text(
        'name: fs.read\ncommand: ["echo", "custom"]\n', encoding="utf-8"
    )
    user = tmp_path / "nonexistent"

    with caplog.at_level(logging.WARNING, logger="cothis.cli"):
        _all_tools(project, user)

    # The shadow warning names the tool, both layers, and both sources.
    shadow_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "shadows" in r.message
    ]
    assert len(shadow_warnings) == 1
    msg = shadow_warnings[0].message
    assert "fs.read" in msg
    assert "project-local" in msg
    assert "builtins" in msg
    assert "fs_read.yaml" in msg


def test_no_shadow_loads_both(tmp_path: Any, caplog: Any) -> None:
    """Distinct names across layers → both load, no shadow warning."""
    import logging

    project = tmp_path / "project"
    project.mkdir()
    (project / "deploy.yaml").write_text(
        'name: proj.deploy\ncommand: ["echo", "deploy"]\n', encoding="utf-8"
    )
    user = tmp_path / "user"
    user.mkdir()
    (user / "hello.yaml").write_text(
        'name: user.hello\ncommand: ["echo", "hi"]\n', encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="cothis.cli"):
        tools = _all_tools(project, user)

    names = {t.__name__: t for t in tools}
    assert "proj.deploy" in names
    assert "user.hello" in names
    assert names["proj.deploy"]() == "deploy\n"
    assert names["user.hello"]() == "hi\n"
    # No shadow warnings emitted.
    shadow_warnings = [r for r in caplog.records if "shadows" in r.message]
    assert shadow_warnings == []


def test_pre_load_false_on_winner_empties_slot_no_fallback(
    tmp_path: Any, caplog: Any
) -> None:
    """Winner's pre_load=False empties the slot — no fallback to shadowed (#10 + ADR-0003).

    Project-local tool shadows user-global, then the winner's pre_load
    returns False. The slot goes empty — the shadowed user-global tool
    is NOT restored (shadowing is a replacement, not a try).
    """
    import logging

    project = tmp_path / "project"
    project.mkdir()
    (project / "blocked.py").write_text(
        "from cothis import tool\n\n"
        '@tool("shared.tool")\n'
        'def t() -> str:\n    """T."""\n    return "proj"\n\n'
        "@t.pre_load()\n"
        "def gate():\n    return False\n",
        encoding="utf-8",
    )
    user = tmp_path / "user"
    user.mkdir()
    (user / "ok.yaml").write_text(
        'name: shared.tool\ncommand: ["echo", "user"]\n', encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="cothis.cli"):
        tools = _all_tools(project, user)

    names = {t.__name__ for t in tools}
    assert "shared.tool" not in names  # winner dropped, no fallback to user-global
