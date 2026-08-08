"""Tests for ``cothis.cli`` formatting helpers and tool discovery.

``_format_tool_call`` is the only pure formatting function in the CLI module
today; it's worth locking down because its output format is what users read
to debug multi-step agent turns, and the ``repr`` convention (strings quoted,
numbers not) is a deliberate choice.

``discover_tools`` tests cover the two-layer discovery model (project-local +
user-global) and the cross-layer ceiling (raises until #10/#11 land).
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from cothis.supervisor import Supervisor

from cothis.agent import ToolCallEvent
from cothis.cli import _format_tool_call
from cothis.tools import discover_tools


def test_format_string_argument_quoted() -> None:
    event = ToolCallEvent(
        name="fs.read", arguments={"path": "/tmp/x"}, call_id="tu_test",
    )
    assert _format_tool_call(event) == "calling fs.read(path='/tmp/x')"


def test_format_multiple_arguments() -> None:
    event = ToolCallEvent(
        name="fs.create",
        arguments={"path": "/tmp/out.txt", "content": "hello"},
        call_id="tu_test",
    )
    out = _format_tool_call(event)
    # dict iteration order is insertion order; assert both pieces present.
    assert "path='/tmp/out.txt'" in out
    assert "content='hello'" in out
    assert out.startswith("calling fs.create(")
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
    assert any("fs.create" in m and "builtins" in m for m in debug_names)


def test_format_integer_argument_not_quoted() -> None:
    event = ToolCallEvent(
        name="add", arguments={"a": 2, "b": 3}, call_id="tu_test",
    )
    out = _format_tool_call(event)
    assert "a=2" in out
    assert "b=3" in out
    # repr distinguishes: 2 not '2'
    assert "a='2'" not in out


def test_format_no_arguments() -> None:
    event = ToolCallEvent(name="noop", arguments={}, call_id="tu_test")
    assert _format_tool_call(event) == "calling noop()"


def test_format_string_with_special_chars_repr_escaped() -> None:
    # repr keeps newlines / quotes visible, preventing garbled display.
    event = ToolCallEvent(
        name="fs.create",
        arguments={"content": 'line1\nline2 "quoted"'},
        call_id="tu_test",
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
    # Only builtins load (fs.read, fs.list, fs.create).
    names = {t.__name__ for t in tools}
    assert "fs.read" in names
    assert "fs.create" in names


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
    import asyncio
    assert asyncio.run(by_name["shared.tool"]()) == "proj\n"


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
    import asyncio
    assert asyncio.run(by_name["fs.read"]()) == "fake\n"


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
    import asyncio
    assert asyncio.run(by_name["fs.read"]()) == "proj\n"


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
    import asyncio
    assert asyncio.run(names["proj.deploy"]()) == "deploy\n"
    assert asyncio.run(names["user.hello"]()) == "hi\n"
    # No shadow warnings emitted.
    shadow_warnings = [r for r in caplog.records if "shadows" in r.message]
    assert shadow_warnings == []


def test_pre_load_false_on_winner_empties_slot_no_fallback(
    tmp_path: Any, caplog: Any
) -> None:
    """Winner's pre_load=False empties the slot — no fallback to shadowed (#10).

    Project-local tool shadows user-global, then the winner's pre_load
    returns False. The slot goes empty — the shadowed user-global tool
    is NOT restored (shadowing is a replacement, not a try).
    """
    import logging

    project = tmp_path / "project"
    project.mkdir()
    (project / "blocked.py").write_text(
        "from cothis.tools import tool\n\n"
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
    # Observability (grilling #10): the pre_load=False skip must be
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
    """A shadowed tool's load hooks never fire.

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
        "from cothis.tools import tool\n\n"
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
    import asyncio
    assert asyncio.run(by_name["shared.tool"]()) == "proj\n"
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

    ``_cothis_home`` reads ``$COTHIS_HOME`` on every call; a wrapper
    script that sets the env after import sees the new path without
    ``importlib.reload``.
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


def test_tui_command_dispatches_to_cothis_tui_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #237: ``cothis tui`` launches the Textual app via ``cothis.tui.run``.

    The test monkeypatches ``cothis.tui.run`` so the actual Textual event
    loop isn't started (which would block the test). With the TUI wiring (#234),
    ``tui`` passes a ``CothisApp`` subclass instance (``_DrivenCothisApp``)
    into ``run`` rather than letting ``run`` construct a bare ``CothisApp``.
    The test verifies the dispatch + that an app instance is passed.
    """
    import cothis.cli as cli_mod
    import cothis.tui as tui_mod

    captured: list[object] = []

    def fake_run(app: object | None = None) -> None:
        captured.append(app)

    monkeypatch.setattr(tui_mod, "run", fake_run)
    # Supervisor opens a SQLite DB at ``$COTHIS_HOME/supervisor.db`` by
    # default. Stub its construction so the test doesn't touch the
    # filesystem.
    monkeypatch.setattr(
        "cothis.supervisor.Supervisor",
        lambda *a, **kw: MagicMock(),
    )

    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["tui"])
    assert result.exit_code == 0, f"tui command failed: {result.output}"
    assert len(captured) == 1, (
        f"cothis.tui.run was not called once; got {captured}"
    )
    assert captured[0] is not None, (
        "tui command should pass a CothisApp subclass instance, not None"
    )


def test_launch_tui_app_closes_supervisor_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#403: ``Supervisor.close()`` runs when the TUI exits.

    Without this, spawned workers are orphaned (reparented to init), still
    holding session locks — a later ``cothis chat --resume`` raises
    ``SessionLockedError`` until the orphan is manually killed.
    """
    import cothis.cli as cli_mod
    import cothis.tui as tui_mod

    monkeypatch.setattr(tui_mod, "run", lambda app=None: None)

    closed: list[bool] = []
    fake_sup = MagicMock()
    fake_sup.close = lambda: closed.append(True)
    monkeypatch.setattr("cothis.supervisor.Supervisor", lambda *a, **kw: fake_sup)

    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["tui"])
    assert result.exit_code == 0
    assert closed == [True], (
        "Supervisor.close() should be called after run_tui returns; "
        f"close was called {len(closed)} time(s)"
    )


def test_driven_cothis_app_on_worktree_pick_spawns_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """AC #234: ``_DrivenCothisApp.on_worktree_pick`` spawns a session bound to the picked cwd.

    The driven app overrides the logging stub with the real spawn
    recipe: Session.new → Supervisor.spawn_worker → schedule
    attach_session_ws. This test stubs Session + Supervisor so the
    spawn args are verifiable without a real subprocess.
    """
    import cothis.cli as cli_mod

    # Stub Session.new to avoid touching the filesystem.
    sent_session_rows: list[dict] = []

    class _FakeSession:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id

        def append_message(self, role, content):  # noqa: ANN001
            sent_session_rows.append({"role": role, "content": content})

        def close(self) -> None:
            pass

    def fake_session_new(db_path, *, cwd, model, flush_sync):  # noqa: ANN001
        return _FakeSession(session_id="deadbeef" * 4)

    monkeypatch.setattr(
        "cothis.session.Session.new", fake_session_new,
    )

    # Fake Supervisor that records spawn_worker args + returns a handle-like obj.
    spawn_calls: list[dict] = []

    class _FakeHandle:
        ws_url = "ws://127.0.0.1:9999/agent"
        token = "fake-bearer-token"

    class _FakeSupervisor:
        def spawn_worker(self, sid, *, model, provider, cwd, sessions_dir=None, extra_env=None):  # noqa: ANN001
            spawn_calls.append({
                "sid": sid,
                "model": model,
                "provider": provider,
                "cwd": cwd,
                "extra_env": extra_env,
            })
            return _FakeHandle()

    # Stub ``asyncio.create_task`` so the scheduled attach doesn't run.
    scheduled: list = []
    monkeypatch.setattr(
        "asyncio.create_task", lambda coro: scheduled.append(coro),
    )

    sup = _FakeSupervisor()
    # ty needs a cast to accept the fake as the Supervisor parameter —
    # the fake quacks like Supervisor for the spawn_worker call only.
    from typing import cast

    app = cli_mod._DrivenCothisApp.build(
        supervisor=cast("Supervisor", sup),
        model="test-model",
        provider="test-provider",
        provider_env={"TEST_API_KEY": "val"},
    )

    # Stub ``attach_session_ws`` so the scheduled task doesn't try to use
    # real WS infrastructure. It's enough that ``on_worktree_pick`` schedules it.
    attach_calls: list = []
    setattr(app, "attach_session_ws", lambda sid, ws_url, token, db_path=None: attach_calls.append((sid, ws_url, token, db_path)))

    cwd = tmp_path / "worktree-feat"
    cwd.mkdir()
    app.on_worktree_pick(str(cwd))

    # Session row seeded with a placeholder user message naming the cwd.
    assert len(sent_session_rows) == 1
    assert sent_session_rows[0]["role"] == "user"
    assert "worktree-feat" in sent_session_rows[0]["content"][0]["text"]

    # spawn_worker called once with the right args.
    assert len(spawn_calls) == 1
    call = spawn_calls[0]
    assert call["sid"] == "deadbeef" * 4
    assert call["model"] == "test-model"
    assert call["provider"] == "test-provider"
    assert call["cwd"] == cwd
    assert call["extra_env"] == {"TEST_API_KEY": "val"}

    # attach_session_ws scheduled (create_task was stubbed; verify the coro
    # is what ``on_worktree_pick`` would have scheduled).
    assert len(scheduled) == 1

    # attach_session_ws carried the worktree DB path so replay-on-attach
    # fires.
    assert len(attach_calls) == 1
    assert attach_calls[0][3] is not None


@pytest.mark.asyncio
async def test_reattach_on_restart_swallows_and_logs_attach_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """#398 review: a WS-connect failure during re-attach is logged, not lost.

    Pre-fix the ``on_restart`` lambda scheduled a bare ``attach_session_ws``
    task; a raise there became an un-retrieved task exception and the
    re-attach vanished silently — exactly the failure the recovery path
    must survive. ``_reattach_on_restart`` catches and logs it instead.
    """
    import logging as _logging
    from typing import cast

    import cothis.cli as cli_mod
    from cothis.supervisor import Supervisor

    sup = MagicMock()
    app = cli_mod._DrivenCothisApp.build(
        supervisor=cast("Supervisor", sup),
        model="m", provider="p", provider_env={},
    )

    def _boom(sid, ws_url, token):  # noqa: ANN001
        raise OSError("WS connect refused")

    # ``setattr`` (not direct assignment) mirrors the existing spawn test —
    # the real attribute is a bound method on ``CothisApp``.
    setattr(app, "attach_session_ws", _boom)

    with caplog.at_level(_logging.INFO, logger="cothis.cli"):
        # Must not raise — that is the whole point of the wrapper. The
        # method lives on the inner ``_App`` subclass, not ``CothisApp``.
        await app._reattach_on_restart(  # type: ignore
            "c" * 32, "ws://127.0.0.1:1/agent", "t",
        )

    messages = [r.message for r in caplog.records]
    assert any("re-attaching" in m for m in messages), (
        f"expected an info 're-attaching' signal; got {messages}"
    )
    assert any("re-attach failed" in m for m in messages), (
        f"expected a 're-attach failed' warning; got {messages}"
    )


@pytest.mark.asyncio
async def test_on_unmount_cancels_monitor_task() -> None:
    """#398 review: ``on_unmount`` cancels the stashed monitor task.

    Without this, the event loop closes on a pending ``asyncio.sleep``
    inside ``monitor_worker_health`` and asyncio logs
    "Task was destroyed but it is pending!".
    """
    from typing import cast

    import cothis.cli as cli_mod
    from cothis.supervisor import Supervisor

    sup = MagicMock()
    app = cli_mod._DrivenCothisApp.build(
        supervisor=cast("Supervisor", sup),
        model="m", provider="p", provider_env={},
    )

    long_task = asyncio.create_task(asyncio.sleep(1000))
    setattr(app, "_monitor_task", long_task)

    # ``on_unmount`` lives on the inner ``_App`` subclass, not ``CothisApp``.
    await app.on_unmount()  # type: ignore

    assert long_task.cancelled(), "on_unmount should cancel the monitor task"


def test_driven_cothis_app_on_worktree_pick_rolls_back_on_spawn_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#390: a failed ``spawn_worker`` rolls back the just-created session.

    No ``session-*.db`` is left behind (no ghost surfacing in
    ``cothis history``), and the failure is logged. Uses real
    ``Session.new`` (creates a real db file) so the rollback's file
    removal is exercised end-to-end, not just the row delete.
    """
    import logging
    from typing import cast

    import cothis.cli as cli_mod

    # on_worktree_pick resolves the db itself (_resolve_db_path(cwd=worktree));
    # pin COTHIS_HOME so that lands at tmp_path/agents.db.
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))
    db_path = cli_mod._resolve_db_path()

    class _FailingSupervisor:
        def spawn_worker(self, *a: object, **kw: object) -> None:
            raise RuntimeError("worker exec failed (simulated)")

    app = cli_mod._DrivenCothisApp.build(
        supervisor=cast("Supervisor", _FailingSupervisor()),
        model="m",
        provider="p",
        provider_env={},
    )

    with caplog.at_level(logging.ERROR, logger="cothis.cli"):
        # Must NOT raise — the rollback catches the spawn failure.
        app.on_worktree_pick(str(tmp_path))

    # AC #1: the session row was deleted (no ghost in the shared db).
    # The db file still exists (other sessions may live in it), but the
    # rolled-back session must not be queryable.
    from cothis.session import Session
    try:
        Session.peek_messages(db_path, "0" * 32)
        raise AssertionError("peek should raise KeyError for unknown id")
    except KeyError:
        pass  # correct — the db has no sessions after rollback
    # AC #2: the failure + rollback is logged.
    assert any(
        "rolled back" in r.getMessage() and "spawn failed" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_tui_created_session_visible_to_cli_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#400: a session created via ``on_worktree_pick`` is visible to
    ``cothis history`` (``Session.list_visible`` on the shared db).

    Previously the TUI stored sessions in ``<cwd>/.cothis/sessions/`` per-
    session files — divergent from ``_resolve_db_path()`` (``agents.db`` or
    ``.agents/sessions/session.db``). A TUI-created session was invisible to
    ``cothis history`` / ``delete`` / ``archive`` / ``resume``.
    """
    from typing import cast

    import cothis.cli as cli_mod
    from cothis.session import Session

    # on_worktree_pick resolves the db itself (_resolve_db_path(cwd=worktree));
    # pin COTHIS_HOME so the TUI-created session lands at tmp_path/agents.db,
    # the same db ``cothis history`` reads.
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))
    db_path = cli_mod._resolve_db_path()

    class _FakeHandle:
        ws_url = "ws://fake"
        token = "tok"

    class _FakeSupervisor:
        def spawn_worker(self, sid, **kw):  # noqa: ANN001, ANN003
            return _FakeHandle()

    # Stub asyncio.create_task so the WS-attach doesn't need a running loop.
    monkeypatch.setattr("asyncio.create_task", lambda coro: None)

    app = cli_mod._DrivenCothisApp.build(
        supervisor=cast("Supervisor", _FakeSupervisor()),
        model="m",
        provider="p",
        provider_env={},
    )
    app.on_worktree_pick(str(tmp_path))

    # The session must be in the SAME db that ``cothis history`` reads
    # (``_resolve_db_path`` → ``Session.list_visible``).
    sessions = Session.list_visible(db_path, tmp_path)
    assert len(sessions) == 1, (
        f"TUI-created session should be visible to list_visible; got {sessions}"
    )


@pytest.mark.asyncio
async def test_on_worktree_pick_real_worker_finds_session_default_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#402 acceptance #2: a REAL worker finds the session on_worktree_pick made.

    Not mocked: a real ``Supervisor`` spawns a real ``cothis worker``
    subprocess; the worker resolves its db via ``_resolve_db_path()`` and
    must load the session the TUI just created. The mocked-Supervisor tests
    above can't catch the TUI-writes-here / worker-reads-there db mismatch —
    only a real subprocess exercises the worker's own db resolution. Default
    storage mode (``$COTHIS_HOME/agents.db``).
    """
    import asyncio
    import json

    import websockets

    import cothis.cli as cli_mod
    from cothis.session import Session
    from cothis.supervisor import Supervisor

    worktree = tmp_path / "worktree-feat"
    worktree.mkdir()
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    # on_worktree_pick schedules attach_session_ws via create_task; close the
    # coro without running it — the WS attach is not what this test exercises.
    monkeypatch.setattr(asyncio, "create_task", lambda coro: coro.close())

    sup = Supervisor(tmp_path / "sup.db")
    try:
        app = cli_mod._DrivenCothisApp.build(
            supervisor=sup,
            model="openai/gpt-oss-120b",
            provider="openrouter",
            provider_env={"OPENROUTER_API_KEY": "test-dummy-not-used"},
        )
        # A raise here means the worker exited before bind — it could not
        # find the session: the exact #402 failure mode.
        app.on_worktree_pick(str(worktree))

        # Spawn registered exactly one running worker for the created session.
        assert len(sup._workers) == 1, sup._workers
        sid, handle = next(iter(sup._workers.items()))
        assert handle.status == "running"
        assert handle.ws_url.startswith("ws://127.0.0.1:")

        # Real WS round-trip proves the bind landed and the worker is live.
        async with websockets.connect(
            handle.ws_url,
            additional_headers={"Authorization": f"Bearer {handle.token}"},
        ) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            assert json.loads(raw) == {"type": "pong"}

        # The session persisted (not rolled back) in the shared db.
        msgs = Session.peek_messages(cli_mod._resolve_db_path(), sid)
        assert msgs, "TUI-created session should persist after a successful spawn"
    finally:
        sup.close()


@pytest.mark.asyncio
async def test_on_worktree_pick_real_worker_finds_session_project_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#402: the create-session flow also works in project storage mode.

    Project mode (``COTHIS_SESSIONS_TYPE=project``) resolves the db
    cwd-relative (``<cwd>/.agents/sessions/session.db``). The worker runs in
    the picked worktree, so on_worktree_pick must create the session in the
    *worktree's* db — not the TUI launch dir — or the worker resolves a
    different file and exits before bind. Regression guard for the
    worktree-cwd divergence (the TUI launch dir here != the worktree).
    """
    import asyncio
    import json

    import websockets

    import cothis.cli as cli_mod
    from cothis.session import Session
    from cothis.supervisor import Supervisor

    launch = tmp_path / "launch"
    launch.mkdir()
    worktree = tmp_path / "wt-proj"
    worktree.mkdir()
    monkeypatch.setenv("COTHIS_SESSIONS_TYPE", "project")
    monkeypatch.setenv("COTHIS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    # TUI launch dir != worktree: without the worktree-relative fix,
    # on_worktree_pick would resolve the launch-dir db while the worker
    # resolves the worktree db → mismatch.
    monkeypatch.chdir(launch)
    monkeypatch.setattr(asyncio, "create_task", lambda coro: coro.close())

    sup = Supervisor(tmp_path / "sup.db")
    try:
        app = cli_mod._DrivenCothisApp.build(
            supervisor=sup,
            model="openai/gpt-oss-120b",
            provider="openrouter",
            provider_env={"OPENROUTER_API_KEY": "test-dummy-not-used"},
        )
        app.on_worktree_pick(str(worktree))  # must not raise

        assert len(sup._workers) == 1, sup._workers
        sid, handle = next(iter(sup._workers.items()))
        assert handle.status == "running"

        async with websockets.connect(
            handle.ws_url,
            additional_headers={"Authorization": f"Bearer {handle.token}"},
        ) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            assert json.loads(raw) == {"type": "pong"}

        # Session lives in the WORKTREE's db (where the worker reads), not
        # the launch dir's.
        wt_db = worktree / ".agents" / "sessions" / "session.db"
        assert wt_db.exists(), f"session db should be in the worktree; got {wt_db}"
        assert Session.peek_messages(wt_db, sid), "session missing from worktree db"
    finally:
        sup.close()


def test_check_resume_exists_rejects_bogus_id(tmp_path: Path) -> None:
    """#394: a well-formed-but-nonexistent resume id is rejected before the TUI launches.

    Both the TUI default path (``chat``) and the ``tui`` command must probe
    existence (not just format) so a misspelt/obsolete id fails fast with
    "session … not found" instead of launching the TUI + spawning a doomed
    worker.
    """
    import typer

    from cothis.cli import _check_resume_exists

    db_path = tmp_path / "session.db"
    bogus = "0" * 32  # well-formed hex, nonexistent

    try:
        _check_resume_exists(db_path, bogus)
        raise AssertionError("should have raised BadParameter")
    except typer.BadParameter as exc:
        assert "not found" in str(exc)
        assert bogus in str(exc)


@pytest.mark.parametrize(
    "argv",
    [
        ["chat", "--resume", "0" * 32],
        ["tui", "--resume", "0" * 32],
    ],
    ids=["chat", "tui"],
)
def test_resume_bogus_id_rejected_before_tui_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
) -> None:
    """#394 AC #1+#2: both TUI entrypoints reject a bogus resume id pre-launch.

    ``chat`` (default TUI path) and ``tui`` gate on existence, not just
    format. A well-formed-but-nonexistent id exits non-zero with the legacy
    "session … not found" message and never reaches ``_launch_tui_app`` /
    ``Supervisor.spawn_worker`` (both stubbed to fail if touched).
    """
    import cothis.cli as cli_mod

    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))

    def fail_if_reached(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not launch the TUI for a bogus resume id")

    monkeypatch.setattr(cli_mod, "_launch_tui_app", fail_if_reached)
    monkeypatch.setattr(
        "cothis.supervisor.Supervisor.spawn_worker", fail_if_reached,
    )

    from typer.testing import CliRunner

    result = CliRunner().invoke(cli_mod.app, argv)
    assert result.exit_code != 0, f"expected failure, got: {result.output}"
    assert "not found" in result.output, result.output
    assert "0" * 32 in result.output, result.output


def test_check_resume_exists_rejects_foreign_cwd(tmp_path: Path) -> None:
    """#394 review: the probe enforces the cwd visibility filter.

    ``Session.load(cwd=...)`` treats a session whose cwd is neither the
    current cwd nor an ancestor as not-found (KeyError) — the same parity
    the legacy REPL gets from ``Session.load``. The seeded cwd is a sibling
    of the test process's cwd, so it is never an ancestor-or-equal
    regardless of where the suite runs.
    """
    from pathlib import Path

    import typer

    from cothis.cli import _check_resume_exists
    from cothis.session import Session

    db_path = tmp_path / "session.db"
    foreign_cwd = Path.cwd().parent / "cothis-foreign-cwd"
    s = Session.new(db_path, cwd=foreign_cwd, model="m", flush_sync=True)
    s.append_message("user", [{"type": "text", "text": "hi"}])
    sid = s.session_id
    s.close()

    try:
        _check_resume_exists(db_path, sid)
        raise AssertionError("should have raised BadParameter")
    except typer.BadParameter as exc:
        assert "not found" in str(exc)


def test_check_resume_exists_accepts_archived_session(tmp_path: Path) -> None:
    """#394 review: the probe accepts cold/archived sessions via the #384 fallback.

    ``Session.load`` falls back to the archive index on a hot miss, so an
    archived session id must pass the probe (return ``None``) instead of
    being reported as "not found".
    """
    from pathlib import Path

    from cothis.cli import _check_resume_exists
    from cothis.session import Session
    from cothis.session.archive import ArchiveIndex, archive_session

    db_path = tmp_path / "session.db"
    s = Session.new(db_path, cwd=Path.cwd(), model="m", flush_sync=True)
    s.append_message("user", [{"type": "text", "text": "hi"}])
    sid = s.session_id
    s.close()

    archive_dir = tmp_path / "archive"
    archive_session(
        hot_db_path=db_path,
        archive_dir=archive_dir,
        session_id=sid,
        archive_db_name="2026-07.db",
        archived_at="2026-07-20T00:00:00+00:00",
        index=ArchiveIndex(archive_dir / "index.json"),
    )

    assert _check_resume_exists(db_path, sid) is None


def test_launch_tui_app_passes_resume_to_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #237 follow-up: ``_launch_tui_app`` forwards ``resume`` to ``build()``.

    The TUI's ``on_mount`` auto-spawns a worker for the resumed session
    when ``resume_session_id`` is set. This test verifies the wiring:
    ``_launch_tui_app(resume=<id>)`` → ``build(resume_session_id=<id>)``.
    """
    import cothis.cli as cli_mod

    captured: dict = {}

    def fake_build(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    def fake_run(app: object | None = None) -> None:
        pass

    monkeypatch.setattr(cli_mod._DrivenCothisApp, "build", fake_build)
    monkeypatch.setattr("cothis.supervisor.Supervisor", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("cothis.tui.run", fake_run)

    session_id = "a" * 32
    cli_mod._launch_tui_app(model="m", provider="p", resume=session_id)

    assert captured.get("resume_session_id") == session_id, (
        f"expected resume_session_id={session_id!r}; got "
        f"{captured.get('resume_session_id')!r}"
    )


def test_launch_tui_app_without_resume_passes_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``resume``, ``build()`` receives ``resume_session_id=None``."""
    import cothis.cli as cli_mod

    captured: dict = {}

    def fake_build(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    def fake_run(app: object | None = None) -> None:
        pass

    monkeypatch.setattr(cli_mod._DrivenCothisApp, "build", fake_build)
    monkeypatch.setattr("cothis.supervisor.Supervisor", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("cothis.tui.run", fake_run)

    cli_mod._launch_tui_app(model="m", provider="p")

    assert captured.get("resume_session_id") is None


# ---------------------------------------------------------------------
# Chat → TUI dispatch routing (#237)
#
# ``chat`` defaults to the TUI; ``--legacy`` and ``--skill`` fall back
# to the REPL. These tests verify the routing without launching either
# path (both are stubbed).
# ---------------------------------------------------------------------


def test_chat_defaults_to_tui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #237: ``cothis chat`` (no flags) routes to ``_launch_tui_app``."""
    import cothis.cli as cli_mod

    captured: list[dict] = []

    def fake_launch(model: str, provider: str, resume: str | None = None) -> None:
        captured.append({"model": model, "provider": provider, "resume": resume})

    monkeypatch.setattr(cli_mod, "_launch_tui_app", fake_launch)

    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["chat"])
    assert result.exit_code == 0, f"chat command failed: {result.output}"
    assert len(captured) == 1, (
        f"expected _launch_tui_app called once; got {captured}"
    )
    assert captured[0]["resume"] is None


def test_chat_legacy_runs_repl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #237: ``cothis chat --legacy`` bypasses the TUI, runs the REPL."""
    import asyncio

    import cothis.cli as cli_mod

    tui_called: list[bool] = []
    asyncio_called: list[bool] = []

    def fake_launch(*args: object, **kwargs: object) -> None:
        tui_called.append(True)

    def fake_run(coro: Any) -> None:
        asyncio_called.append(True)
        coro.close()

    monkeypatch.setattr(cli_mod, "_launch_tui_app", fake_launch)
    monkeypatch.setattr(asyncio, "run", fake_run)

    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["chat", "--legacy"])
    assert result.exit_code == 0, f"chat --legacy failed: {result.output}"
    assert tui_called == [], "TUI should NOT be called with --legacy"
    assert len(asyncio_called) == 1, "legacy REPL (asyncio.run) should be called"


def test_chat_legacy_with_resume_runs_repl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #237: ``chat --legacy --resume <id>`` runs the legacy REPL, not the TUI.

    ``--legacy`` takes precedence over ``--resume`` — the user explicitly
    asked for the old REPL, so the TUI path is skipped even though
    ``--resume`` is supported in the TUI (#365).
    """
    import asyncio

    import cothis.cli as cli_mod

    tui_called: list[bool] = []
    asyncio_called: list[bool] = []

    def fake_launch(*args: object, **kwargs: object) -> None:
        tui_called.append(True)

    def fake_run(coro: Any) -> None:
        asyncio_called.append(True)
        coro.close()

    monkeypatch.setattr(cli_mod, "_launch_tui_app", fake_launch)
    monkeypatch.setattr(asyncio, "run", fake_run)

    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["chat", "--legacy", "--resume", "a" * 32])
    assert result.exit_code == 0, f"chat --legacy --resume failed: {result.output}"
    assert tui_called == [], "TUI should NOT be called with --legacy"
    assert len(asyncio_called) == 1, "legacy REPL should be called"


def test_chat_skill_falls_back_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #237: ``cothis chat --skill foo`` falls back to legacy REPL."""
    import asyncio

    import cothis.cli as cli_mod

    tui_called: list[bool] = []
    asyncio_called: list[bool] = []

    def fake_launch(*args: object, **kwargs: object) -> None:
        tui_called.append(True)

    def fake_run(coro: Any) -> None:
        asyncio_called.append(True)
        coro.close()

    monkeypatch.setattr(cli_mod, "_launch_tui_app", fake_launch)
    monkeypatch.setattr(asyncio, "run", fake_run)

    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["chat", "--skill", "foo"])
    assert result.exit_code == 0, f"chat --skill failed: {result.output}"
    assert tui_called == [], "TUI should NOT be called with --skill"
    assert len(asyncio_called) == 1, "legacy REPL should be called for --skill"


@pytest.mark.asyncio
async def test_worker_session_preactivates_persisted_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#415: the worker agent build preactivates the persisted skill selection.

    Mocks Agent/SessionWorker/Session.load so ``_worker_session`` runs without
    a real DB or WS server; asserts ``preactivate_skills`` reads the saved set.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    import cothis.cli as cli_mod
    import cothis.worker as worker_mod
    from cothis.skills import save_skill_selection

    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))
    save_skill_selection({"alpha", "beta"})
    # Make the persisted names discoverable so the worker's availability
    # filter (mirroring the TUI menu's ``saved & set(skills)``) keeps them.
    from types import SimpleNamespace

    monkeypatch.setattr(
        "cothis.skills.discover_skills",
        lambda _cwd: [SimpleNamespace(name="alpha"), SimpleNamespace(name="beta")],
    )

    # No real DB needed — Session.load is mocked.
    monkeypatch.setattr(
        cli_mod.Session, "load", lambda *a, **k: MagicMock(name="loaded"),
    )
    # Capture the Agent(...) build kwargs.
    captured: dict = {}
    mock_agent = MagicMock()
    mock_agent.attach_session = MagicMock()
    mock_agent.aclose = AsyncMock()
    monkeypatch.setattr(
        cli_mod, "Agent", lambda **kw: (captured.update(kw), mock_agent)[1],
    )
    # SessionWorker: serve_forever exits immediately via CancelledError.
    mock_worker = MagicMock()
    mock_worker.start = AsyncMock(return_value="ws://127.0.0.1:1/agent")
    mock_worker.token = "tok"
    mock_worker.serve_forever = AsyncMock(side_effect=asyncio.CancelledError)
    mock_worker.stop = AsyncMock()
    monkeypatch.setattr(worker_mod, "SessionWorker", lambda agent: mock_worker)

    await cli_mod._worker_session(
        session="a" * 32, model="m", provider="p",
        max_iterations=1, max_tokens=None,
    )

    assert captured.get("preactivate_skills") == ["alpha", "beta"], (
        f"worker should preactivate the persisted selection; got "
        f"{captured.get('preactivate_skills')}"
    )


@pytest.mark.asyncio
async def test_worker_session_preactivation_filters_unavailable_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#416 review: an unavailable persisted skill must not crash the worker.

    A project-scoped skill chosen in project A, or one uninstalled between
    sessions, is absent from ``discover_skills`` at worker start. Passing it
    to ``Agent(preactivate_skills=...)`` would crash ``_run_preactivation``
    with "Unknown skill" (agent.py) before the worker serves. The worker
    filters the persisted set against available skills — mirroring the TUI's
    menu-open filter (``saved & set(skills)``).
    """
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import cothis.cli as cli_mod
    import cothis.worker as worker_mod
    from cothis.skills import save_skill_selection

    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))
    # "ghost" is persisted but not discoverable in this project.
    save_skill_selection({"alpha", "beta", "ghost"})
    monkeypatch.setattr(
        "cothis.skills.discover_skills",
        lambda _cwd: [SimpleNamespace(name="alpha"), SimpleNamespace(name="beta")],
    )

    monkeypatch.setattr(
        cli_mod.Session, "load", lambda *a, **k: MagicMock(name="loaded"),
    )
    captured: dict = {}
    mock_agent = MagicMock()
    mock_agent.attach_session = MagicMock()
    mock_agent.aclose = AsyncMock()
    monkeypatch.setattr(
        cli_mod, "Agent", lambda **kw: (captured.update(kw), mock_agent)[1],
    )
    mock_worker = MagicMock()
    mock_worker.start = AsyncMock(return_value="ws://127.0.0.1:1/agent")
    mock_worker.token = "tok"
    mock_worker.serve_forever = AsyncMock(side_effect=asyncio.CancelledError)
    mock_worker.stop = AsyncMock()
    monkeypatch.setattr(worker_mod, "SessionWorker", lambda agent: mock_worker)

    await cli_mod._worker_session(
        session="a" * 32, model="m", provider="p",
        max_iterations=1, max_tokens=None,
    )

    assert captured.get("preactivate_skills") == ["alpha", "beta"], (
        f"unavailable persisted skill 'ghost' should be filtered out; got "
        f"{captured.get('preactivate_skills')}"
    )


# --- compaction CLI flags (--summary-model / --min-retained-turns) ---------
#
# These pin the two operator knobs. They mirror the
# ``--max-tokens`` / ``COTHIS_MAX_TOKENS`` idiom. ``--min-retained-turns``
# carries ``envvar=COTHIS_MIN_RETAINED_TURNS`` (typer reads it); ``--summary-model``
# deliberately has NO envvar= so the env read stays inside the agent's
# ``resolve_summary_model`` (the env-var behaviour preserved byte-for-byte).


def _ask_capture_agent(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
) -> tuple[dict, Any]:
    """Invoke ``cothis ask`` capturing the ``Agent(...)`` build kwargs.

    ``Agent`` is replaced by a spy; ``asyncio.run`` is short-circuited (the
    coro is closed) so the mock agent never actually runs. Returns
    ``(captured_kwargs, runner_result)``.
    """
    import cothis.cli as cli_mod

    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)

    captured: dict = {}
    mock_agent = MagicMock()
    monkeypatch.setattr(
        cli_mod, "Agent", lambda **kw: (captured.update(kw), mock_agent)[1],
    )

    def fake_run(coro: Any) -> None:
        coro.close()

    monkeypatch.setattr(asyncio, "run", fake_run)

    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["ask", *argv])
    return captured, result


def test_ask_threads_compaction_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ask --summary-model X --min-retained-turns 7`` reaches the Agent ctor."""
    captured, result = _ask_capture_agent(
        monkeypatch, ["--summary-model", "X", "--min-retained-turns", "7", "hi"]
    )
    assert result.exit_code == 0, f"ask failed: {result.output}"
    assert captured["summary_model"] == "X"
    assert captured["min_retained_turns"] == 7


def test_ask_min_retained_turns_envvar_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``COTHIS_MIN_RETAINED_TURNS`` feeds ``--min-retained-turns`` via typer envvar."""
    captured, result = _ask_capture_agent(
        monkeypatch, ["hi"], env={"COTHIS_MIN_RETAINED_TURNS": "9"}
    )
    assert result.exit_code == 0, f"ask failed: {result.output}"
    assert captured["min_retained_turns"] == 9


def test_ask_min_retained_turns_flag_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag > env precedence for ``--min-retained-turns`` (mirrors --max-tokens)."""
    captured, result = _ask_capture_agent(
        monkeypatch,
        ["--min-retained-turns", "8", "hi"],
        env={"COTHIS_MIN_RETAINED_TURNS": "5"},
    )
    assert result.exit_code == 0, f"ask failed: {result.output}"
    assert captured["min_retained_turns"] == 8


def test_ask_summary_model_env_not_read_by_typer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--summary-model`` has NO ``envvar=`` on purpose.

    With ``COTHIS_SUMMARY_MODEL`` set and no ``--summary-model`` flag, the
    Agent ctor receives ``summary_model=None`` so the env read is delegated
    to ``resolve_summary_model`` inside the agent (the env-var behaviour
    preserved byte-for-byte). This pins the no-envvar= design.
    """
    captured, result = _ask_capture_agent(
        monkeypatch, ["hi"], env={"COTHIS_SUMMARY_MODEL": "openai/gpt-4o"}
    )
    assert result.exit_code == 0, f"ask failed: {result.output}"
    assert captured["summary_model"] is None


@pytest.mark.asyncio
async def test_chat_session_threads_compaction_flags_to_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_chat_session`` (chat's ``--legacy`` REPL path) forwards the compaction
    flags into the in-process ``Agent(...)`` ctor.

    The TUI default path is intentionally NOT asserted here — it spawns a
    worker subprocess that does not yet carry the flags (deferred follow-up).
    """
    from unittest.mock import AsyncMock

    import cothis.cli as cli_mod

    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))
    # Session.new: no real DB; a closeable MagicMock is enough.
    mock_session = MagicMock()
    mock_session.close = MagicMock()
    monkeypatch.setattr(cli_mod.Session, "new", lambda *a, **k: mock_session)

    captured: dict = {}
    mock_agent = MagicMock()
    mock_agent.attach_session = MagicMock()
    mock_agent.aclose = AsyncMock()
    monkeypatch.setattr(
        cli_mod, "Agent", lambda **kw: (captured.update(kw), mock_agent)[1],
    )

    # PromptSession.prompt_async -> EOFError on first call so the REPL loop
    # breaks immediately without reading stdin (Python 3.14 parses the
    # ``except EOFError, KeyboardInterrupt:`` tuple form, so EOFError is
    # caught and the loop breaks).
    class _EOFSession:
        async def prompt_async(self, *_a: object, **_k: object) -> str:
            raise EOFError

    monkeypatch.setattr(
        "prompt_toolkit.shortcuts.PromptSession",
        lambda *a, **k: _EOFSession(),
    )

    await cli_mod._chat_session(
        model="m",
        provider="p",
        max_iterations=1,
        max_tokens=None,
        summary_model="openai/gpt-4o",
        min_retained_turns=3,
    )

    assert captured["summary_model"] == "openai/gpt-4o"
    assert captured["min_retained_turns"] == 3

