"""Tests for ``cothis.cli`` formatting helpers and tool discovery.

``_format_tool_call`` is the only pure formatting function in the CLI module
today; it's worth locking down because its output format is what users read
to debug multi-step agent turns, and the ``repr`` convention (strings quoted,
numbers not) is a deliberate choice.

``_all_tools`` tests cover the two-layer discovery model (project-local +
user-global) and the cross-layer ceiling (raises until #10/#11 land).
"""

from __future__ import annotations

from pathlib import Path
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
# _all_tools: two-layer discovery (issue #9)
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


def test_all_tools_cross_layer_conflict_raises(tmp_path: Any) -> None:
    """Same name in user-global and project-local raises (ceiling — #10 fixes this)."""
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

    import pytest

    with pytest.raises(ValueError, match="duplicate tool name.*shared.tool"):
        _all_tools(project, user)


def test_all_tools_builtin_conflict_raises(tmp_path: Any) -> None:
    """Custom tool shadowing a builtin raises (ceiling — #11 fixes this)."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "override.yaml").write_text(
        'name: fs.read\ncommand: ["echo", "fake"]\n', encoding="utf-8"
    )
    user = tmp_path / "nonexistent"

    import pytest

    with pytest.raises(ValueError, match="duplicate tool name.*fs.read"):
        _all_tools(project, user)


def test_all_tools_pre_load_false_drops_tool(tmp_path: Any) -> None:
    """A tool whose pre_load returns False is dropped from the final list."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "t.py").write_text(
        "from cothis import tool\n\n"
        '@tool("blocked.tool")\n'
        'def t() -> str:\n    """T."""\n    return "ok"\n\n'
        "@t.pre_load()\n"
        "def gate():\n    return False\n",
        encoding="utf-8",
    )
    user = tmp_path / "nonexistent"

    tools = _all_tools(project, user)
    names = {t.__name__ for t in tools}
    assert "blocked.tool" not in names
