"""Tests for YAML-driven shell-template tools.

These cover the ``tools/xxx.yaml`` format: a tool is declared in YAML with
a ``name``, a ``command`` template, and (later) typed ``args``. Each YAML
tool compiles down to a Python callable the Agent registers by name and
dispatches by calling with keyword args matching the declared arg names.

Shell mode (cothis): ``command`` is rendered via ``string.Template`` and
executed with ``shell=True``. This is required for the template-string
substitution the user asked for and supports pipes/redirection down the
road. Ceiling: shell=True is injection-permissive by design — a YAML tool
is trusted the same way a hand-written Python tool is. Upgrade path: if
untrusted YAML ever needs to be supported, add an ``argv`` mode that
splits the command into a list and uses ``shell=False``.
"""

from __future__ import annotations

from typing import cast

import pytest

from cothis.tools import (
    _eval_if,
    _render_command,
    _resolve_shell,
    _ShellTool,
    load_tools_from_dir,
    load_yaml_tools,
    preview,
)


def _shell_tool(yaml_text: str) -> _ShellTool:
    """Load a YAML tool and cast to ``_ShellTool`` for attribute access.

    ``load_yaml_tools`` returns ``list[Tool]`` (the dispatch protocol), but
    every YAML-compiled tool is concretely a ``_ShellTool`` carrying
    ``__cothis_schema__`` / ``__signature__``. ``cast`` tells ty that without
    a per-line type-ignore — the cast is sound because the loader only ever
    produces ``_ShellTool`` instances.
    """
    return cast("_ShellTool", load_yaml_tools(yaml_text)[0])


def test_yaml_no_args_runs_command() -> None:
    """A no-arg YAML tool runs its command verbatim and returns stdout.

    Tracer bullet: proves the YAML -> callable -> subprocess path works
    end to end. No placeholders, no arg plumbing — just ``command`` as a
    literal string handed to the shell.
    """
    yaml_text = """
name: hello
command: echo hello
"""
    tools = load_yaml_tools(yaml_text)
    assert len(tools) == 1
    tool = tools[0]
    assert tool.__name__ == "hello"
    result = tool()
    assert result == "hello\n"


def test_yaml_description_becomes_docstring() -> None:
    """``description:`` flows to the tool's docstring.

    any-llm's ``callable_to_tool`` raises ``ValueError`` if ``__doc__`` is
    empty, so the loader MUST set a docstring or every YAML tool crashes
    at tool-prepare time (observed in production: ``Function hello must
    have a docstring``).
    """
    yaml_text = """
name: hello
description: Say hello.
command: echo hello
"""
    tool = load_yaml_tools(yaml_text)[0]
    assert tool.__doc__ == "Say hello."


def test_yaml_missing_description_gets_default_docstring() -> None:
    """A YAML tool without ``description:`` still gets a non-empty docstring.

    Without this, ``callable_to_tool`` crashes even though the tool is
    otherwise well-formed. The default is derived from the command so a
    user can still tell tools apart by their docstring in the schema.
    """
    yaml_text = """
name: hello
command: echo hello
"""
    tool = load_yaml_tools(yaml_text)[0]
    assert tool.__doc__
    assert "echo hello" in tool.__doc__


def test_yaml_arg_substituted_into_command() -> None:
    """A declared arg fills its ``{name}`` placeholder in the command.

    Tracer bullet for the args slice: a single ``int`` arg ``offset``
    declared in YAML becomes a kwarg on the compiled callable, and its
    value is substituted into ``command`` before the shell runs. We use
    ``echo`` (not ``date``) so the test does not depend on a real shell
    builtin's argument parsing — only the template substitution matters
    here; the ``date -d`` smoke test lives in ``.agents/tools/`` and is
    exercised by the end-to-end runnable check.
    """
    yaml_text = """
name: echo_offset
description: Echo back the offset value.
command: echo offset is {offset}
args:
  - name: offset
    type: int
    description: Number to echo.
"""
    tool = load_yaml_tools(yaml_text)[0]
    result = tool(offset=7)
    assert result == "offset is 7\n"


def test_yaml_arg_list_substituted_space_separated() -> None:
    """A list-typed arg renders as a space-separated sequence.

    This is the slice the user asked for (``args 列表传入``): a value
    like ``[1, 2, 3]`` becomes ``"1 2 3"`` in the rendered command, so
    multi-value arguments (packages, ids, …) flow into shell commands
    the same way they would on a hand-typed command line.
    """
    yaml_text = """
name: echo_many
description: Echo multiple ids space-separated.
command: echo ids are {ids}
args:
  - name: ids
    type: list
    description: Ids to echo.
"""
    tool = load_yaml_tools(yaml_text)[0]
    result = tool(ids=[1, 2, 3])
    assert result == "ids are 1 2 3\n"


def test_yaml_nonzero_exit_returns_error_with_stderr() -> None:
    """Stories 18/19: a non-zero exit surfaces as an error string, not a raise.

    The PRD specifies ``stdout on success / Error: exit code N: <stderr>``
    on failure. The agent must see the failure to recover; an unhandled
    exception would crash the ReAct loop mid-turn. stderr is included so
    the model can act on what went wrong.
    """
    yaml_text = """
name: failer
description: A tool that fails.
command: echo boom >&2; exit 3
"""
    tool = load_yaml_tools(yaml_text)[0]
    result = tool()
    assert "exit code 3" in result
    assert "boom" in result


def test_yaml_zero_exit_returns_stdout() -> None:
    """Stories 18/19: a zero exit returns stdout, not an error string."""
    yaml_text = """
name: ok
description: A tool that succeeds.
command: echo good
"""
    tool = load_yaml_tools(yaml_text)[0]
    assert tool() == "good\n"


def test_yaml_steps_selects_matching_if(monkeypatch) -> None:
    """A ``command:`` list with per-branch ``if:`` selects the matching OS.

    Tracer bullet for the conditional-list syntax: each branch carries an
    ``if:`` GitHub-Actions expression over ``runner.os``; the loader picks
    the first whose predicate is true under the current context. We force
    ``sys.platform == 'linux'`` so the test is deterministic and does not
    depend on the host it runs on.
    """
    monkeypatch.setattr("sys.platform", "linux")
    yaml_text = """
name: greeting
description: Greet based on OS.
command:
  - if: runner.os == 'Linux'
    run: echo from-linux
  - if: runner.os == 'Windows'
    run: echo from-windows
"""
    tool = load_yaml_tools(yaml_text)[0]
    assert tool() == "from-linux\n"


def test_yaml_steps_unknown_os_loads_with_error(monkeypatch) -> None:
    """No matching ``if:`` and no ``default`` → ``preview`` raises (load skips).

    The load path silently skips registration (returns ``[]``); ``preview``
    is a diagnostic tool so it surfaces the gap as a ``ValueError`` naming
    the platform. Both behaviours are pinned here.
    """
    monkeypatch.setattr("sys.platform", "linux")
    yaml_text = """
name: windows-only
description: Only makes sense on Windows.
command:
  - if: runner.os == 'Windows'
    run: echo hi
"""
    # Load path: silently skipped, not registered.
    assert load_yaml_tools(yaml_text) == []
    # Preview path: diagnostic, raises.
    import pytest

    with pytest.raises(ValueError, match="no matching branch"):
        preview(yaml_text)


def test_load_tools_from_dir_finds_yaml_files(tmp_path) -> None:
    """``.agents/tools/*.yaml`` discovery: every YAML file in the dir loads.

    Verifies the directory-loading path end to end against a real temp
    dir with two YAML files on disk. Non-YAML files in the same dir are
    ignored (the loader globs ``*.yaml`` / ``*.yml`` only).
    """
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "hello.yaml").write_text(
        "name: hello\ncommand: echo hello\n", encoding="utf-8"
    )
    (tools_dir / "bye.yml").write_text(
        "name: bye\ncommand: echo bye\n", encoding="utf-8"
    )
    # Non-YAML files must not be picked up.
    (tools_dir / "README.md").write_text("not a tool", encoding="utf-8")

    tools = load_tools_from_dir(tools_dir)
    assert len(tools) == 2
    by_name = {t.__name__: t for t in tools}
    assert set(by_name) == {"hello", "bye"}
    assert by_name["hello"]() == "hello\n"
    assert by_name["bye"]() == "bye\n"


def test_load_tools_from_dir_finds_nested_yaml(tmp_path) -> None:
    """Discovery walks subdirectories so ``tools/date/current.yaml`` loads.

    Nested directories are how related tools are grouped (all ``date.*``
    tools live under ``date/``); non-recursive globs would silently drop
    them. The glob is recursive (``**/*.yaml``), but tool ``name`` still
    comes from each file's ``name:`` field, not from the path.
    """
    date_dir = tmp_path / "date"
    date_dir.mkdir()
    (date_dir / "current.yaml").write_text(
        "name: date.current\ncommand: echo today\n", encoding="utf-8"
    )
    # Also confirm top-level files still load alongside nested ones.
    (tmp_path / "flat.yaml").write_text(
        "name: flat\ncommand: echo flat\n", encoding="utf-8"
    )

    tools = load_tools_from_dir(tmp_path)
    names = {t.__name__ for t in tools}
    assert names == {"date.current", "flat"}


# ----------------------------------------------------------------------
# preview() — inspect the rendered command without dispatching.
#
# These tests assert the *shell content* directly (the string that would
# be passed to ``subprocess.run``), so platform branches and arg
# substitution can be verified without spawning a process. This is the
# only way to test the Windows/pwsh branch on a Linux CI host.
# ----------------------------------------------------------------------


def test_preview_linux_branch_renders_command(monkeypatch) -> None:
    """On Linux, ``preview`` returns the selected branch's command, rendered.

    No subprocess is spawned — the returned ``cmd`` is exactly the string
    that would be handed to ``subprocess.run``. Shell is ``None`` (POSIX
    default ``sh``).
    """
    monkeypatch.setattr("sys.platform", "linux")
    yaml_text = """
name: date.calculate
description: Calculate a date offset.
args:
  - name: offset
    type: int
command:
  - if: runner.os == 'Linux' || runner.os == 'macOS'
    run: date -d "{offset} days" +%Y-%m-%d
  - if: runner.os == 'Windows'
    shell: pwsh
    run: (Get-Date).AddDays({offset}).ToString('yyyy-MM-dd')
"""
    shell, cmd = preview(yaml_text, offset=5)
    assert cmd == 'date -d "5 days" +%Y-%m-%d'
    assert shell is None  # POSIX default


def test_preview_windows_branch_no_subprocess(monkeypatch) -> None:
    """The Windows branch is selected and its command rendered, no subprocess.

    ``_os="Windows"`` forces the Windows branch regardless of the host
    OS — no need to monkeypatch ``sys.platform``. We still monkeypatch
    ``shutil.which`` so the resolved shell is deterministic regardless
    of whether pwsh is actually installed.
    """
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    yaml_text = """
name: date.calculate
description: Calculate a date offset.
args:
  - name: offset
    type: int
command:
  - if: runner.os == 'Linux' || runner.os == 'macOS'
    run: date -d "{offset} days"
  - if: runner.os == 'Windows'
    shell: pwsh
    run: (Get-Date).AddDays({offset}).ToString('yyyy-MM-dd')
"""
    shell, cmd = preview(yaml_text, _os="Windows", offset=-7)
    assert cmd == "(Get-Date).AddDays(-7).ToString('yyyy-MM-dd')"
    assert shell == "/fake/pwsh"


def test_preview_linux_branch_override_from_windows_host(monkeypatch) -> None:
    """``_os`` override selects a branch the host would not normally select.

    We force ``sys.platform`` to Windows but request the Linux branch via
    ``_os="Linux"`` — the override wins. This is the verification affordance
    for a matrix CI: one host (e.g. windows-latest) previews every branch.
    """
    monkeypatch.setattr("sys.platform", "win32")
    yaml_text = """
name: cross
description: Cross-platform.
command:
  - if: runner.os == 'Linux'
    run: echo from-linux
  - if: runner.os == 'Windows'
    run: echo from-windows
"""
    _, cmd = preview(yaml_text, _os="Linux")
    assert cmd == "echo from-linux"


def test_preview_list_arg_rendered_space_separated(monkeypatch) -> None:
    """``preview`` honours the same list→space-join rendering as dispatch."""
    monkeypatch.setattr("sys.platform", "linux")
    yaml_text = """
name: echo_many
description: Echo many.
command: echo ids {ids}
args:
  - name: ids
    type: list
"""
    _, cmd = preview(yaml_text, ids=[1, 2, 3])
    assert cmd == "echo ids 1 2 3"


# ----------------------------------------------------------------------
# Per-arg descriptions reaching the LLM schema.
#
# ``_build_tool_schema`` exists because any-llm's ``callable_to_tool``
# drops per-parameter ``description`` fields — the whole PRD win is that
# the rich text a YAML author writes actually reaches the model. If this
# silently regresses, the model gets generic "Parameter X of type Y"
# descriptions and picks worse arguments. No external signal — the tool
# still *runs* — so only a direct schema assertion catches the drift.
# ----------------------------------------------------------------------


def test_yaml_arg_description_reaches_schema() -> None:
    """A YAML arg's ``description:`` is carried verbatim into ``__cothis_schema__``.

    This is the core PRD promise: bypassing any-llm's lossy
    ``callable_to_tool`` so per-arg descriptions survive. We assert the
    exact description text appears on the property — not just that the
    property exists — because a regression that kept the property but
    dropped the description text would be the silent breakage.
    """
    yaml_text = """
name: calc
description: Calculate something.
command: echo {n}
args:
  - name: n
    type: int
    description: The number to calculate with.
"""
    schema = _shell_tool(yaml_text).__cothis_schema__
    assert schema is not None
    assert schema["function"]["description"] == "Calculate something."
    assert schema["function"]["parameters"]["properties"]["n"]["description"] == (
        "The number to calculate with."
    )
    assert schema["function"]["parameters"]["properties"]["n"]["type"] == "integer"
    assert schema["function"]["parameters"]["required"] == ["n"]


def test_yaml_arg_without_description_omits_field() -> None:
    """An arg with no ``description:`` yields a property with no description key.

    Pins the other half of the schema contract: absence is preserved as
    absence, not as an empty string or a generic placeholder. A provider
    seeing ``"description": ""`` might behave differently than seeing no
    key at all.
    """
    yaml_text = """
name: plain
description: Plain.
command: echo {n}
args:
  - name: n
    type: int
"""
    schema = _shell_tool(yaml_text).__cothis_schema__
    assert schema is not None
    prop = schema["function"]["parameters"]["properties"]["n"]
    assert "description" not in prop
    assert prop["type"] == "integer"


# ----------------------------------------------------------------------
# _render_command — the pure substitution function.
#
# Pins the real brace/placeholder behaviour. The earlier docstring claimed
# "literal braces survive and missing placeholders stay as-is" — that was
# false: ``string.Template`` requires ``${...}`` syntax, so the renderer
# rewrites ``{`` → ``${`` up front, and an undeclared ``{x}`` becomes
# ``${x}`` in the output (not ``{x}``). A user writing a command with a
# literal bash ``${VAR}`` or brace expansion would get mangled output, so
# the test documents the actual ceiling.
# ----------------------------------------------------------------------


def test_render_command_undeclared_placeholder_becomes_dollar_brace() -> None:
    """An undeclared ``{x}`` is rewritten to ``${x}``, not left as ``{x}``.

    This pins reality against the old (false) docstring claim. The renderer
    pre-converts ``{`` to ``${`` for ``string.Template``; ``safe_substitute``
    then leaves unknown ``${x}`` in place. So a literal brace in a command
    is NOT a way to pass ``{`` through — it becomes ``${``. Ceiling named
    in the corrected ``_render_command`` docstring.
    """
    assert _render_command("echo {x}", [], {}) == "echo ${x}"


def test_render_command_declared_arg_substituted() -> None:
    """A declared, provided arg substitutes its value; scalars are ``str()``-ed."""
    assert _render_command("echo {n}", ["n"], {"n": 7}) == "echo 7"


def test_render_command_list_space_joined() -> None:
    """A list value is space-joined into the placeholder."""
    assert _render_command("echo {ids}", ["ids"], {"ids": [1, 2, 3]}) == "echo 1 2 3"


def test_render_command_empty_list_renders_empty() -> None:
    """An empty list renders as the empty string (join of nothing).

    Edge case: ``" ".join([]) == ""``, so ``echo {ids}`` with ``ids=[]``
    yields ``echo `` (trailing space). Pinned so a future change to empty
    rendering (e.g. dropping the placeholder) is a conscious decision.
    """
    assert _render_command("echo {ids}", ["ids"], {"ids": []}) == "echo "


# ----------------------------------------------------------------------
# _eval_if — the GA ``if:`` recursive-descent evaluator.
#
# A parser with precedence rules (``||`` < ``&&`` < comparison) is textbook
# non-trivial logic: a subtle change to the grammar can silently flip which
# platform branch a tool selects, and the only outward signal is a wrong
# command on one OS. Each operator gets its own check so a precedence
# regression is localised.
# ----------------------------------------------------------------------


def test_eval_if_and_both_true() -> None:
    ctx = {"os": "Linux", "arch": "X64"}
    assert _eval_if("runner.os == 'Linux' && runner.arch == 'X64'", ctx) is True


def test_eval_if_and_short_circuits_on_false() -> None:
    # Left false → whole ``&&`` false, regardless of the right operand.
    ctx = {"os": "Windows", "arch": "X64"}
    assert _eval_if("runner.os == 'Linux' && runner.arch == 'X64'", ctx) is False


def test_eval_if_not_equal() -> None:
    ctx = {"os": "Windows"}
    assert _eval_if("runner.os != 'Linux'", ctx) is True
    assert _eval_if("runner.os != 'Windows'", ctx) is False


def test_eval_if_parens_group_over_precedence() -> None:
    # Without parens ``&&`` binds tighter than ``||``; parens force the OR
    # to evaluate first. This is the one place precedence is observable.
    ctx = {"os": "macOS", "arch": "ARM64"}
    expr = "(runner.os == 'Linux' || runner.os == 'macOS') && runner.arch == 'ARM64'"
    assert _eval_if(expr, ctx) is True
    # Same ctx but arch mismatch → the parenthesised OR is true but AND fails.
    ctx_x64 = {"os": "macOS", "arch": "X64"}
    assert _eval_if(expr, ctx_x64) is False


def test_eval_if_runner_arch_lookup() -> None:
    """``runner.arch`` resolves against the context's ``arch`` key."""
    ctx = {"os": "Linux", "arch": "ARM64"}
    assert _eval_if("runner.arch == 'ARM64'", ctx) is True
    assert _eval_if("runner.arch == 'X64'", ctx) is False


def test_eval_if_case_insensitive_string_compare() -> None:
    """GA ``==`` on strings is case-insensitive (``'linux'`` matches ``'Linux'``).

    GA's documented semantics; if this regresses to Python ``==``, every
    YAML author who lowercases the OS literal gets a silently wrong branch.
    """
    ctx = {"os": "Linux"}
    assert _eval_if("runner.os == 'linux'", ctx) is True
    assert _eval_if("runner.os == 'LINUX'", ctx) is True


def test_eval_if_rejects_unsupported_identifier() -> None:
    """Only ``runner.os`` / ``runner.arch`` resolve; anything else raises.

    Failures in ``if:`` must be loud (the loader/preview raises), not silent
    (treat as false and pick the wrong branch).
    """
    with pytest.raises(ValueError, match="unsupported identifier"):
        _eval_if("github.ref == 'main'", {"os": "Linux"})


def test_eval_if_rejects_trailing_tokens() -> None:
    """Malformed expressions raise rather than partially evaluate."""
    with pytest.raises(ValueError, match="trailing tokens"):
        _eval_if("runner.os == 'Linux' extra", {"os": "Linux"})


# ----------------------------------------------------------------------
# _resolve_shell — mapping a YAML ``shell:`` name to a subprocess path.
#
# Untested per the standards review. ``None`` → ``None`` is the POSIX/CMD
# default path; a named shell resolves via ``shutil.which``. The unresolved
# case (name not on PATH) returns the name as-is — which surfaces as a
# ``FileNotFoundError`` at dispatch time. That is a known ceiling (the
# review's hard finding), pinned here so the behaviour is documented in
# code, not just in prose.
# ----------------------------------------------------------------------


def test_resolve_shell_none_yields_none() -> None:
    """``shell:`` absent → ``None`` → subprocess uses its platform default."""
    assert _resolve_shell(None) is None


def test_resolve_shell_named_resolves_via_which(monkeypatch) -> None:
    """A named shell is resolved through ``shutil.which`` (PATH-driven)."""
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert _resolve_shell("bash") == "/usr/bin/bash"


def test_resolve_shell_unresolved_returns_name_as_is(monkeypatch) -> None:
    """A name ``shutil.which`` can't find is returned unchanged.

    cothis ceiling: this name is then passed to ``subprocess.run(executable=...)``
    and raises ``FileNotFoundError`` at dispatch time (caught by Agent's
    tool-error boundary and surfaced to the model). Pinned so a future
    change to skip-on-missing is a conscious decision, not silent.
    """
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert _resolve_shell("pwsh") == "pwsh"


# ----------------------------------------------------------------------
# Malformed-YAML error paths (story 24).
#
# A missing required field or an empty file must surface as a clear,
# actionable error naming the field (and the file, when loaded from disk),
# not a bare ``KeyError`` / ``TypeError``. Previously these raised
# ``KeyError: 'name'`` or ``TypeError: 'NoneType' object is not subscriptable``
# — errors the LLM/user cannot act on. ``_require`` centralises the fix.
# ----------------------------------------------------------------------


def test_load_yaml_tools_missing_name_names_field() -> None:
    """A spec without ``name:`` raises ``ValueError`` naming the field."""
    with pytest.raises(ValueError, match="must define 'name'"):
        load_yaml_tools("command: echo hi\n")


def test_load_yaml_tools_empty_file_raises_valueerror() -> None:
    """An empty YAML file (``safe_load`` → ``None``) raises, not ``TypeError``.

    ``None["name"]`` used to raise ``TypeError`` — not actionable. Now
    ``_require`` sees a non-dict spec and names the problem.
    """
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_yaml_tools("")


def test_load_yaml_tools_arg_missing_name_names_field() -> None:
    """An arg entry without ``name:`` raises naming the tool + field."""
    yaml_text = """
name: t
command: echo {x}
args:
  - type: int
"""
    with pytest.raises(ValueError, match="must define 'name'"):
        load_yaml_tools(yaml_text)


def test_load_tools_from_dir_error_names_the_file(tmp_path) -> None:
    """Loading from disk includes the file path in the error (story 24).

    The whole point of threading ``source`` through: a startup failure must
    point at the offending file, not just the field, so the user can find
    and fix it. We assert the filename appears in the message.
    """
    bad = tmp_path / "broken.yaml"
    bad.write_text("command: echo hi\n", encoding="utf-8")  # no name:
    with pytest.raises(ValueError, match="broken.yaml"):
        load_tools_from_dir(tmp_path)


# ----------------------------------------------------------------------
# ``command:`` list-shape validation.
#
# The rename from ``steps:`` to ``command:`` overloaded the field: it now
# accepts either a string or a list of branches. Each malformation (wrong
# type, empty list, branch missing ``run:``) has its own actionable error
# — a YAML author who typo's ``cmd:`` instead of ``run:`` should hear
# about that, not a cryptic ``KeyError`` deep in the renderer.
# ----------------------------------------------------------------------


def test_command_wrong_type_raises() -> None:
    """A non-str, non-list ``command:`` is rejected with a type-naming error."""
    yaml_text = """
name: t
command: 42
"""
    with pytest.raises(ValueError, match="must be a string or a list"):
        load_yaml_tools(yaml_text)


def test_command_empty_list_raises() -> None:
    """An empty ``command: []`` is rejected — a tool with no command is useless."""
    yaml_text = """
name: t
command: []
"""
    with pytest.raises(ValueError, match="list is empty"):
        load_yaml_tools(yaml_text)


def test_command_branch_missing_run_raises() -> None:
    """A list branch without ``run:`` names the branch index in the error.

    Common typo: writing ``cmd:`` or ``command:`` (the old field name) inside
    a branch instead of ``run:``. The error points at the offending branch.
    """
    yaml_text = """
name: t
command:
  - if: runner.os == 'Linux'
    cmd: echo hi
"""
    with pytest.raises(ValueError, match="branch #0 must be a mapping with a 'run'"):
        load_yaml_tools(yaml_text)


# ----------------------------------------------------------------------
# YAML schema discipline: type coercion + unknown-field rejection.
#
# These pin the ``extra="forbid"`` + type-stringify fixes. Without them a
# typo (``shel:``) or a non-string scalar (``name: 42``) was silently
# swallowed, and an unknown ``type: float`` silently became ``"string"`` in
# the emitted schema — the apify-MCP #738 class of bug (schema pollution
# with no signal). Each check makes one malformation loud.
# ----------------------------------------------------------------------


def test_unknown_top_level_field_rejected() -> None:
    """A typo'd top-level key is rejected, not silently ignored."""
    yaml_text = "name: t\ncommand: echo hi\nfrobnicate: yes\n"
    with pytest.raises(ValueError, match="unknown field.*frobnicate"):
        load_yaml_tools(yaml_text)


def test_unknown_arg_field_rejected() -> None:
    """An arg entry with an unknown key is rejected, naming the key."""
    yaml_text = """
name: t
command: echo {x}
args:
  - name: x
    type: int
    bogus: yes
"""
    with pytest.raises(ValueError, match="unknown field.*bogus"):
        load_yaml_tools(yaml_text)


def test_unknown_branch_field_rejected() -> None:
    """A command branch with an unknown key is rejected, naming branch + key."""
    yaml_text = """
name: t
command:
  - if: runner.os == 'Linux'
    run: echo hi
    bogus: yes
"""
    with pytest.raises(ValueError, match="command branch #0.*unknown field.*bogus"):
        load_yaml_tools(yaml_text)


def test_legacy_steps_field_rejected() -> None:
    """The removed ``steps:`` field is now unknown and rejected.

    After the rename to ``command:``, a leftover ``steps:`` is a stale-format
    marker — reject it so a migrated file doesn't silently lose its branches.
    """
    yaml_text = """
name: t
command: echo hi
steps:
  - run: echo bye
"""
    with pytest.raises(ValueError, match="unknown field.*steps"):
        load_yaml_tools(yaml_text)


def test_unknown_arg_type_rejected() -> None:
    """An unknown ``type:`` (e.g. float) is rejected, listing the legal types.

    Previously ``type: float`` silently became ``"string"`` in the emitted
    schema — the LLM saw a string arg when the author meant a number. Now it
    fails at load time listing {bool,int,list,str}.
    """
    yaml_text = """
name: t
command: echo {x}
args:
  - name: x
    type: float
"""
    with pytest.raises(ValueError, match="unknown type 'float'.*allowed"):
        load_yaml_tools(yaml_text)


def test_name_non_string_is_stringified() -> None:
    """A numeric ``name:`` is coerced to a string for ``__name__``.

    ``__name__`` must be a str (Agent keys its tool map by it; the schema's
    ``function.name`` is a string). YAML unquoted ``42`` parses as int; we
    stringify rather than reject so authors aren't forced to over-quote.
    """
    tool = load_yaml_tools("name: 42\ncommand: echo hi\n")[0]
    assert tool.__name__ == "42"
    assert isinstance(tool.__name__, str)


def test_description_non_string_is_stringified() -> None:
    """A numeric ``description:`` is coerced to a string for ``__doc__``.

    ``__doc__`` flows into the tool schema's description field, which must be
    a string; a numeric scalar is stringified rather than rejected.
    """
    tool = load_yaml_tools("name: t\ndescription: 123\ncommand: echo hi\n")[0]
    assert tool.__doc__ == "123"
    assert isinstance(tool.__doc__, str)


# ----------------------------------------------------------------------
# ``has_shell`` / ``has_exe`` predicate functions in ``if:`` expressions.
#
# These gate a branch on whether a binary is actually on PATH, not just on
# the OS. The canonical use: a PowerShell branch that should only register
# when ``pwsh`` is installed, not merely when the OS is Windows. Both names
# map to ``shutil.which`` — the intent differs ("run under this shell" vs
# "this executable exists") but the mechanism is identical.
# ----------------------------------------------------------------------


def test_has_shell_matches_when_binary_present(monkeypatch) -> None:
    """``has_shell('pwsh')`` is true when ``shutil.which`` finds pwsh."""
    monkeypatch.setattr(
        "shutil.which", lambda name: f"/usr/bin/{name}" if name == "pwsh" else None
    )
    assert _eval_if("has_shell('pwsh')", {"os": "Linux"}) is True


def test_has_shell_false_when_binary_absent(monkeypatch) -> None:
    """``has_shell('pwsh')`` is false when ``shutil.which`` returns None."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert _eval_if("has_shell('pwsh')", {"os": "Windows"}) is False


def test_has_exe_same_mechanism_as_has_shell(monkeypatch) -> None:
    """``has_exe`` is the same ``shutil.which`` check under a different name."""
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/git" if name == "git" else None
    )
    assert _eval_if("has_exe('git')", {"os": "Linux"}) is True
    assert _eval_if("has_exe('svn')", {"os": "Linux"}) is False


def test_has_shell_combinable_with_runner_os(monkeypatch) -> None:
    """``has_shell`` composes with ``runner.os`` via ``&&`` / ``||``.

    The whole point of putting predicates in the expression language (rather
    than as branch-level fields): they combine. A Windows+pwsh branch is
    ``runner.os == 'Windows' && has_shell('pwsh')``.
    """
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/pwsh" if name == "pwsh" else None
    )
    ctx = {"os": "Windows"}
    assert _eval_if("runner.os == 'Windows' && has_shell('pwsh')", ctx) is True
    assert _eval_if("runner.os == 'Linux' && has_shell('pwsh')", ctx) is False
    assert _eval_if("has_shell('pwsh') || runner.os == 'Linux'", ctx) is True


def test_has_shell_branch_registers_only_when_available(monkeypatch) -> None:
    """A ``has_shell``-gated branch is skipped (not registered) when absent.

    End-to-end: the tool has a pwsh branch and a fallback. When pwsh is not
    on PATH, the pwsh branch is not selected; the default (or next matching
    branch) wins. When no branch matches and there's no default, the tool
    is silently not registered.
    """
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: None)  # nothing on PATH
    yaml_text = """
name: ps_tool
description: PowerShell-only.
command:
  - if: has_shell('pwsh')
    shell: pwsh
    run: Get-Date
"""
    # No default, no match → not registered.
    assert load_yaml_tools(yaml_text) == []


def test_has_shell_branch_selected_when_available(monkeypatch) -> None:
    """When the gated binary IS available, the branch registers and runs."""
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/pwsh" if name == "pwsh" else None
    )
    yaml_text = """
name: ps_tool
description: PowerShell-only.
command:
  - if: has_shell('pwsh')
    shell: pwsh
    run: echo found
"""
    # Registered (not skipped) and the branch's run is selected.
    assert len(load_yaml_tools(yaml_text)) == 1
    _, cmd = preview(yaml_text)
    assert cmd == "echo found"


def test_unknown_function_in_if_raises(monkeypatch) -> None:
    """An unsupported function name raises, naming what's available.

    Uses a valid single-string-arg call shape (``always('x')``) so the parse
    reaches function dispatch and the name check fires — not the arg-shape
    check. ``always`` is a real GA function we deliberately don't support.
    """
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(ValueError, match="unsupported function"):
        _eval_if("always('x')", {"os": "Linux"})


# ----------------------------------------------------------------------
# ``default`` branch — the explicit fallback.
# ----------------------------------------------------------------------


def test_default_branch_selected_when_no_if_matches(monkeypatch) -> None:
    """The ``default: true`` branch wins when all ``if:`` branches fail."""
    monkeypatch.setattr("sys.platform", "linux")
    yaml_text = """
name: fallback
description: Has a default.
command:
  - if: runner.os == 'Windows'
    run: echo windows
  - default: true
    run: echo fallback
"""
    assert load_yaml_tools(yaml_text)[0]() == "fallback\n"


def test_default_branch_not_selected_when_if_matches(monkeypatch) -> None:
    """An ``if:`` match takes priority over ``default``."""
    monkeypatch.setattr("sys.platform", "linux")
    yaml_text = """
name: prioritised
description: If wins over default.
command:
  - if: runner.os == 'Linux'
    run: echo linux
  - default: true
    run: echo fallback
"""
    assert load_yaml_tools(yaml_text)[0]() == "linux\n"


def test_default_and_if_on_same_branch_raises() -> None:
    """A branch cannot carry both ``if:`` and ``default`` — ambiguous."""
    yaml_text = """
name: t
command:
  - if: runner.os == 'Linux'
    default: true
    run: echo hi
"""
    with pytest.raises(ValueError, match="cannot have both 'if' and 'default'"):
        load_yaml_tools(yaml_text)


def test_multiple_defaults_raise() -> None:
    """Two ``default`` branches are a configuration error."""
    yaml_text = """
name: t
command:
  - default: true
    run: echo one
  - default: true
    run: echo two
"""
    with pytest.raises(ValueError, match="multiple 'default' branches"):
        load_yaml_tools(yaml_text)
