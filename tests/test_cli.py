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
    loop isn't started (which would block the test). After slice E (#234),
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
    """AC #234 slice E: ``_DrivenCothisApp.on_worktree_pick`` spawns a session bound to the picked cwd.

    The driven app overrides slice D's logging stub with the real spawn
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
        def spawn_worker(self, sid, *, model, provider, cwd, sessions_dir, extra_env):  # noqa: ANN001
            spawn_calls.append({
                "sid": sid,
                "model": model,
                "provider": provider,
                "cwd": cwd,
                "sessions_dir": sessions_dir,
                "extra_env": extra_env,
            })
            return _FakeHandle()

    # Stub ``asyncio.create_task`` so the scheduled attach doesn't run.
    scheduled: list = []
    monkeypatch.setattr(
        "asyncio.create_task", lambda coro: scheduled.append(coro),
    )

    sessions_dir = tmp_path / "sessions"
    sup = _FakeSupervisor()
    # ty needs a cast to accept the fake as the Supervisor parameter —
    # the fake quacks like Supervisor for the spawn_worker call only.
    from typing import cast

    app = cli_mod._DrivenCothisApp.build(
        supervisor=cast("Supervisor", sup),
        sessions_dir=sessions_dir,
        model="test-model",
        provider="test-provider",
        provider_env={"TEST_API_KEY": "val"},
    )

    # Stub ``attach_session_ws`` so the scheduled task doesn't try to use
    # real WS infrastructure. It's enough that ``on_worktree_pick`` schedules it.
    attach_calls: list = []
    setattr(app, "attach_session_ws", lambda sid, ws_url, token: attach_calls.append((sid, ws_url, token)))

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
    assert call["sessions_dir"] == sessions_dir
    assert call["extra_env"] == {"TEST_API_KEY": "val"}

    # attach_session_ws scheduled (create_task was stubbed; verify the coro
    # is what ``on_worktree_pick`` would have scheduled).
    assert len(scheduled) == 1


def test_driven_cothis_app_on_worktree_pick_rolls_back_on_spawn_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
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

    sessions_dir = tmp_path / "sessions"

    class _FailingSupervisor:
        def spawn_worker(self, *a: object, **kw: object) -> None:
            raise RuntimeError("worker exec failed (simulated)")

    app = cli_mod._DrivenCothisApp.build(
        supervisor=cast("Supervisor", _FailingSupervisor()),
        sessions_dir=sessions_dir,
        model="m",
        provider="p",
        provider_env={},
    )

    with caplog.at_level(logging.ERROR, logger="cothis.cli"):
        # Must NOT raise — the rollback catches the spawn failure.
        app.on_worktree_pick(str(tmp_path))

    # AC #1: no ghost session-*.db left behind.
    assert list(sessions_dir.glob("session-*.db")) == []
    # AC #2: the failure + rollback is logged.
    assert any(
        "rolled back" in r.getMessage() and "spawn failed" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


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
