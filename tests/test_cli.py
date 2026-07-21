"""Tests for ``cothis.cli`` formatting helpers and tool discovery.

``_format_tool_call`` is the only pure formatting function in the CLI module
today; it's worth locking down because its output format is what users read
to debug multi-step agent turns, and the ``repr`` convention (strings quoted,
numbers not) is a deliberate choice.

``discover_tools`` tests cover the two-layer discovery model (project-local +
user-global) and the cross-layer ceiling (raises until #10/#11 land).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from cothis.agent import ToolCallEvent
from cothis.cli import _format_tool_call
from cothis.tools import discover_tools


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


def test_discover_tools_emits_per_tool_debug_log(tmp_path: Any, caplog: Any) -> None:
    """Story 43: each loaded tool gets a DEBUG line naming its source.

    The WARNING summary stays (for shadow diagnostics); the per-tool DEBUG
    lines are the user-facing way to answer "why didn't my tool load?"
    without digging through shadow/gating WARNINGs.
    """
    import logging

    project = tmp_path / "project"
    project.mkdir()
    (project / "deploy.yaml").write_text(
        'name: proj.deploy\ncommand: ["echo", "deploy"]\n', encoding="utf-8"
    )
    user = tmp_path / "nonexistent"

    with caplog.at_level(logging.DEBUG, logger="cothis.tools"):
        tools = discover_tools(project, user)

    names = {t.__name__ for t in tools}
    assert "proj.deploy" in names

    debug_loaded = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and "loaded tool" in r.msg and "from" in r.msg
    ]
    # Each registered tool emitted one DEBUG line.
    debug_names = [r.getMessage() for r in debug_loaded]
    assert any("proj.deploy" in m and "deploy.yaml" in m for m in debug_names)
    assert any("fs.read" in m and "builtins" in m for m in debug_names)
    assert any("fs.write" in m and "builtins" in m for m in debug_names)


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
# discover_tools: two-layer discovery + cross-layer shadow semantics (#9, #10, #11)
# --------------------------------------------------------------------


def test_discover_tools_user_global_absent_no_error(tmp_path: Any) -> None:
    """Missing user-global dir is the common case — must not error."""
    project = tmp_path / "project"
    project.mkdir()
    user = tmp_path / "nonexistent"

    tools = discover_tools(project, user)
    # Only builtins load (fs.read, fs.dir, fs.write).
    names = {t.__name__ for t in tools}
    assert "fs.read" in names
    assert "fs.write" in names


def test_discover_tools_user_global_loads_tools(tmp_path: Any) -> None:
    """Tools from ``$COTHIS_HOME/tools/`` appear in the tool list."""
    project = tmp_path / "project"
    project.mkdir()
    user = tmp_path / "user"
    user.mkdir()
    (user / "hello.yaml").write_text(
        'name: user.hello\ncommand: ["echo", "hi"]\n', encoding="utf-8"
    )

    tools = discover_tools(project, user)
    names = {t.__name__ for t in tools}
    assert "user.hello" in names


def test_discover_tools_project_local_loads_tools(tmp_path: Any) -> None:
    """Tools from ``.agents/tools/`` appear in the tool list."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "deploy.yaml").write_text(
        'name: proj.deploy\ncommand: ["echo", "deploy"]\n', encoding="utf-8"
    )
    user = tmp_path / "nonexistent"

    tools = discover_tools(project, user)
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

    tools = discover_tools(project, user)
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

    tools = discover_tools(project, user)
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

    with caplog.at_level(logging.WARNING, logger="cothis.tools"):
        discover_tools(project, user)

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


def test_shadow_warning_names_both_file_paths(tmp_path: Any, caplog: Any) -> None:
    """user-global vs project-local shadow — warning names BOTH file paths.

    The builtin-case test (test_shadow_emits_warning_both_layers) can't
    distinguish the layer name from the source fallback (both contain
    "builtins"). This test uses two real file paths so a regression to
    single-path warnings would fail it.
    """
    import logging

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

    with caplog.at_level(logging.WARNING, logger="cothis.tools"):
        discover_tools(project, user)

    shadow_warnings = [r for r in caplog.records if "shadows" in r.message]
    assert len(shadow_warnings) == 1
    msg = shadow_warnings[0].message
    # Both layer names appear.
    assert "project-local" in msg
    assert "user-global" in msg
    # Both file paths appear (the winner's AND the shadowed tool's).
    assert str(project / "dup.yaml") in msg
    assert str(user / "dup.yaml") in msg


def test_chained_shadow_three_layers_two_warnings(tmp_path: Any, caplog: Any) -> None:
    """All three layers claim one name → two shadow warnings, project wins.

    Covers the chained-shadow path Copilot flagged: user-global shadows a
    builtin AND project-local shadows the user-global tool, both in one
    ``discover_tools`` call. Two distinct warnings must fire, and the final
    winner must be project-local.
    """
    import logging

    project = tmp_path / "project"
    project.mkdir()
    (project / "override.yaml").write_text(
        'name: fs.read\ncommand: ["echo", "proj"]\n', encoding="utf-8"
    )
    user = tmp_path / "user"
    user.mkdir()
    (user / "override.yaml").write_text(
        'name: fs.read\ncommand: ["echo", "user"]\n', encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="cothis.tools"):
        tools = discover_tools(project, user)

    # Two shadow warnings: user-global→builtins, then project-local→user-global.
    shadow_warnings = [r for r in caplog.records if "shadows" in r.message]
    assert len(shadow_warnings) == 2
    # Final winner is project-local (its output is "proj", not the builtin
    # behavior, not the user-global "user").
    by_name = {t.__name__: t for t in tools}
    assert by_name["fs.read"]() == "proj\n"


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

    with caplog.at_level(logging.WARNING, logger="cothis.tools"):
        tools = discover_tools(project, user)

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

    with caplog.at_level(logging.WARNING, logger="cothis"):
        tools = discover_tools(project, user)

    names = {t.__name__ for t in tools}
    assert "shared.tool" not in names  # winner dropped, no fallback to user-global
    # Observability (ADR-0003 + grilling #10): the pre_load=False skip must be
    # logged at WARNING so it's visible by default. Filter on the tools logger
    # (the skip is emitted from ``_run_load_hooks`` in tools.py, not cli.py).
    pre_load_skips = [
        r
        for r in caplog.records
        if r.name == "cothis.tools"
        and r.levelno == logging.WARNING
        and "pre_load callback returned False" in r.getMessage()
    ]
    assert len(pre_load_skips) == 1
    assert "shared.tool" in pre_load_skips[0].message


def test_shadowed_tool_load_hooks_never_fire(tmp_path: Any, monkeypatch: Any) -> None:
    """A shadowed tool's load hooks never fire (ADR-0003 Q3).

    The guarantee is structural — load hooks run in the post-merge loop on
    winners only, so a shadowed loser never reaches ``_run_load_hooks``.
    This test pins the negative case: if a regression re-added hook calls
    to the loader, the loser's ``after_load`` side effect would happen.
    The loser is a Python tool (YAML tools can't register hooks); the
    winner is a YAML tool shadowing it — format is never a layer (Q1).
    """
    marker = tmp_path / "loser_hook_fired"
    monkeypatch.setenv("COTHIS_TEST_HOOK_MARKER", str(marker))

    project = tmp_path / "project"
    project.mkdir()
    # Winner: project-local YAML tool shadows the user-global Python tool.
    (project / "winner.yaml").write_text(
        'name: shared.tool\ncommand: ["echo", "proj"]\n', encoding="utf-8"
    )
    user = tmp_path / "user"
    user.mkdir()
    # Loser: user-global Python tool with a side-effecting after_load hook.
    # If its hook fires, it touches the marker file.
    (user / "loser.py").write_text(
        "import os\n\n"
        "from cothis import tool\n\n"
        '@tool("shared.tool")\n'
        'def t() -> str:\n    """T."""\n    return "user"\n\n'
        "@t.after_load()\n"
        "def mark():\n"
        '    path = os.environ.get("COTHIS_TEST_HOOK_MARKER")\n'
        "    if path:\n"
        '        open(path, "w").close()\n',
        encoding="utf-8",
    )

    tools = discover_tools(project, user)

    # Winner is registered (project-local YAML), loser is not.
    by_name = {t.__name__: t for t in tools}
    assert by_name["shared.tool"]() == "proj\n"
    # The loser's after_load hook must NOT have fired — no marker file.
    assert not marker.exists()


def test_cothis_home_env_var_overrides_default(monkeypatch: Any) -> None:
    """``COTHIS_HOME`` overrides the default ``~/.cothis`` for user tools.

    No ``importlib.reload`` needed: ``_cothis_home()`` / ``_user_tools_dir()``
    read the env lazily per call (#66), so monkeypatch's env restore is
    sufficient.
    """
    from pathlib import Path

    from cothis.cli import _cothis_home, _user_tools_dir

    monkeypatch.setenv("COTHIS_HOME", "/custom/cothis-home")
    assert _cothis_home() == Path("/custom/cothis-home")
    assert _user_tools_dir() == Path("/custom/cothis-home/tools")


def test_cothis_home_defaults_to_home_dotcothis(monkeypatch: Any) -> None:
    """Without ``COTHIS_HOME``, the default is ``~/.cothis``."""
    from pathlib import Path

    from cothis.cli import _cothis_home, _user_tools_dir

    monkeypatch.delenv("COTHIS_HOME", raising=False)
    assert _cothis_home() == Path.home() / ".cothis"
    assert _user_tools_dir() == Path.home() / ".cothis" / "tools"


def test_cothis_home_picks_up_late_env_change(
    monkeypatch: Any,
) -> None:
    """Changing ``COTHIS_HOME`` after import is reflected without reload (#66).

    The pre-#66 constants froze the value at import; a wrapper script
    that set the env after import got the stale path. The lazy
    functions read the env on every call.
    """
    from pathlib import Path

    from cothis.cli import _cothis_home

    monkeypatch.setenv("COTHIS_HOME", "/first")
    assert _cothis_home() == Path("/first")
    monkeypatch.setenv("COTHIS_HOME", "/second")
    assert _cothis_home() == Path("/second")


def test_main_keyboard_interrupt_exits_130(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C during ``app()`` surfaces as ``SystemExit(130)`` — the POSIX
    convention for SIGINT (128 + 2). No ``Error:`` line on stderr."""
    import cothis.cli as cli_mod

    def raise_kbi(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli_mod, "app", raise_kbi)
    monkeypatch.setattr(cli_mod, "_debug", False)
    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main()
    assert exc_info.value.code == 130


def test_main_keyboard_interrupt_with_debug_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under ``--debug`` a Ctrl-C must surface the traceback rather than
    be silently swallowed — developers want to see where the interrupt
    landed."""
    import cothis.cli as cli_mod

    def raise_kbi(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli_mod, "app", raise_kbi)
    monkeypatch.setattr(cli_mod, "_debug", True)
    with pytest.raises(KeyboardInterrupt):
        cli_mod.main()


def test_main_generic_exception_still_error_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-``KeyboardInterrupt`` exceptions surface as ``Error: <msg>`` on
    stderr with exit code 1."""
    import cothis.cli as cli_mod

    def raise_value_error(*args: Any, **kwargs: Any) -> None:
        raise ValueError("genuine crash")

    monkeypatch.setattr(cli_mod, "app", raise_value_error)
    monkeypatch.setattr(cli_mod, "_debug", False)
    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main()
    assert exc_info.value.code == 1
