"""Tests for the YAML tool loader (``cothis.tools``).

Covers the silent-breakage surfaces of the YAML tool format:

- **Type-driven execution**: ``command:`` as a list (argv mode, ``shell=False``)
  vs as a string (shell mode, ``shell=True`` with required ``shell:``).
- **Placeholder rendering**: ``{arg}`` substitution via ``str.format_map``,
  including ``{{`` escaping and load-time rejection of undeclared placeholders.
- **Platform selection**: ``platforms:`` map (``linux``/``macos``/``unix``/
  ``windows``) overriding top-level defaults; ``unix`` covering linux+macOS.
- **Executable gating**: argv[0] / ``shell:`` must be on PATH or the tool is
  silently not registered.
- **Args schema discipline**: per-arg descriptions reaching the LLM schema;
  declared-but-unreferenced args dropped with a warning; per-platform args
  merging (branch overrides same-named, inherits the rest).
- **Malformed-YAML errors**: every malformation surfaces an actionable error
  naming the field and file, never a bare ``KeyError``.
- **preview()**: the verification surface for asserting rendered command
  content without spawning a subprocess.

Tests spawn short-lived subprocesses (``echo``) but never touch the network.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import pytest

from cothis.tools.core import load_tools_from_layer
from cothis.tools.yaml import (
    _all_placeholders,
    _current_platform,
    _extract_field_names,
    _merge_arg_specs,
    _shell_argv,
    _ShellTool,
    load_yaml_tools,
    preview,
)

if TYPE_CHECKING:
    from pathlib import Path


def _shell_tool(yaml_text: str) -> _ShellTool:
    """Load a YAML tool and cast to ``_ShellTool`` for attribute access.

    ``load_yaml_tools`` returns ``list[Tool]`` (the dispatch protocol), but
    every YAML-compiled tool is concretely a ``_ShellTool`` carrying
    ``__cothis_schema__`` / ``__signature__``. ``cast`` tells ty that without
    a per-line type-ignore — the cast is sound because the loader only ever
    produces ``_ShellTool`` instances.
    """
    return cast("_ShellTool", load_yaml_tools(yaml_text)[0])


# ====================================================================
# argv mode (command: as a list)
# ====================================================================


def test_argv_no_args_runs_command() -> None:
    """A no-arg argv tool runs its command verbatim and returns stdout."""
    import asyncio

    tool = load_yaml_tools('name: hi\ncommand: ["echo", "hello"]\n')[0]
    assert tool.__name__ == "hi"
    assert asyncio.run(tool()) == "hello\n"


def test_argv_arg_substituted_into_element() -> None:
    """A declared arg fills its ``{name}`` placeholder in one argv element."""
    yaml_text = """
name: echo_n
description: Echo a number.
command: ["echo", "{n}"]
args:
  - name: n
    type: int
    description: The number.
"""
    import asyncio
    assert asyncio.run(load_yaml_tools(yaml_text)[0](n=7)) == "7\n"


def test_argv_spaces_in_value_safe_without_quote() -> None:
    """An arg value containing spaces stays as one argv item — no shell split.

    This is the core safety win of argv mode over shell mode: ``"my file"``
    is passed to ``execve`` as a single argument, not split on the space by
    a shell. No ``shlex.quote`` is needed.
    """
    yaml_text = """
name: cat_file
description: cat a file.
command: ["echo", "{path}"]
args:
  - name: path
    type: str
"""
    import asyncio
    assert asyncio.run(load_yaml_tools(yaml_text)[0](path="my file")) == "my file\n"


def test_argv_empty_list_rejected() -> None:
    """An empty argv list is rejected — a tool with no command is useless."""
    with pytest.raises(ValueError, match="'command' list is empty"):
        load_yaml_tools("name: t\ncommand: []\n")


# ====================================================================
# shell mode (command: as a string + shell:)
# ====================================================================


def test_shell_runs_via_declared_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A string command runs under the declared ``shell:`` interpreter."""
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    yaml_text = """
name: sh_echo
description: echo via shell.
shell: bash
command: "echo from-bash"
"""
    # Note: we monkeypatched shutil.which, so the shell resolves to a fake
    # path; the tool registers. Actual execution here uses echo (a builtin),
    # but we only assert the rendered command via preview (no real bash needed).
    cmd, shell = preview(yaml_text)
    assert cmd == "echo from-bash"
    assert shell == "bash"


def test_shell_auto_selected_when_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A string command without ``shell:`` auto-selects the OS default (story 16).

    POSIX → ``sh``, Windows → ``cmd``. Explicit ``shell:`` still overrides.
    """
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    yaml_text = """
name: t
command: echo hi
"""
    tool = _shell_tool(yaml_text)
    expected = "cmd" if _current_platform() == "windows" else "sh"
    assert tool._block.shell == expected


def test_shell_pipe_supported_in_string_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shell mode supports pipes / ``&&`` / redirection — that's its purpose."""
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    yaml_text = """
name: piped
description: pipe.
shell: bash
command: "echo foo | wc -c"
"""
    cmd, _ = preview(yaml_text)
    assert cmd == "echo foo | wc -c"


# ====================================================================
# Placeholder rendering (Python format-string semantics)
# ====================================================================


def test_placeholder_substitutes_declared_arg() -> None:
    """``{name}`` renders to the arg's value in both argv and shell modes."""
    assert "echo {x}".format_map({"x": "hi"}) == "echo hi"
    assert "{a}/{b}".format_map({"a": "x", "b": "y"}) == "x/y"


def test_placeholder_format_spec_honoured() -> None:
    """Python format specs work (``{n:03d}`` zero-pads), full ``str.format``.

    Free win from using ``str.format_map`` instead of a custom renderer:
    specs / conversions / complex templates all just work. The int value is
    passed through as-is (not pre-stringified) so the spec can apply.
    """
    yaml_text = """
name: t
shell: bash
command: "echo {n:03d}"
args:
  - name: n
    type: int
"""
    cmd, _ = preview(yaml_text, n=5)
    assert cmd == "echo 005"


def test_placeholder_conversion_honoured() -> None:
    """Python conversions work (``{p!r}`` adds repr quotes)."""
    yaml_text = """
name: t
shell: bash
command: "echo {p!r}"
args:
  - name: p
    type: str
"""
    cmd, _ = preview(yaml_text, p="hi")
    assert cmd == "echo 'hi'"


def test_shell_variable_escaped_with_double_brace() -> None:
    """Shell ``${HOME}`` must be escaped as ``${{HOME}}``.

    Because ``command:`` is a Python format-string, every literal ``{`` or
    ``}`` must be doubled. A bare ``${HOME}`` would work (``$`` isn't a
    format char), but ``${HOME}`` specifically contains ``{...}`` which
    ``str.format`` would try to interpret as a field — hence the escape.
    """
    cmd, _ = preview('name: t\nshell: bash\ncommand: "echo ${{HOME}}"\n')
    assert cmd == "echo ${HOME}"


def test_brace_expansion_escaped() -> None:
    """Bash brace expansion ``{a,b}`` must be escaped as ``{{a,b}}``."""
    cmd, _ = preview('name: t\nshell: bash\ncommand: "echo {{a,b}}"\n')
    assert cmd == "echo {a,b}"


def test_command_undeclared_placeholder_rejected_at_load() -> None:
    """A command referencing an undeclared arg is a load-time ``ValueError``.

    Surface the typo at startup, not as ``{name}`` residue reaching the shell
    (where bash might swallow it as an empty variable — silent breakage).
    """
    yaml_text = """
name: t
shell: bash
command: "echo {nope}"
"""
    with pytest.raises(ValueError, match="undeclared placeholder"):
        load_yaml_tools(yaml_text)


def test_extract_field_names_handles_specs_and_conversions() -> None:
    """``_extract_field_names`` returns the base name, ignoring spec/conversion."""
    assert _extract_field_names("echo {a} {b:03d} {c!r}") == {"a", "b", "c"}
    assert _extract_field_names("echo {a.b}") == {"a"}  # attr access → base name
    assert _extract_field_names("echo {a[0]}") == {"a"}  # index → base name


def test_all_placeholders_extracts_names() -> None:
    """``_all_placeholders`` finds every field name in argv and shell commands."""
    assert _all_placeholders(["echo", "{a}", "{b:03d}"]) == {"a", "b"}
    assert _all_placeholders("echo {a} {b!r}") == {"a", "b"}
    assert _all_placeholders(["echo"]) == set()


def test_list_arg_rendered_space_separated() -> None:
    """A list-valued arg is space-joined into the placeholder (shell mode).

    Note: lists are pre-joined (not passed to ``format_map`` as a list) so
    they render as ``"1 2 3"`` rather than ``"[1, 2, 3]"``.
    """
    yaml_text = """
name: echo_ids
description: echo ids.
shell: bash
command: "echo {ids}"
args:
  - name: ids
    type: list
"""
    cmd, _ = preview(yaml_text, ids=[1, 2, 3])
    assert cmd == "echo 1 2 3"


def test_shell_value_with_metacharacters_quoted() -> None:
    """Shell mode: a value containing metacharacters is ``shlex.quote``-d.

    Story 22: a value with spaces or shell metacharacters must not be able to
    break or inject into the command. ``shlex.quote`` wraps it in single
    quotes so the shell treats it as one literal token.
    """
    yaml_text = """
name: grep_file
shell: bash
command: "echo {pattern}"
args:
  - name: pattern
    type: str
"""
    cmd, _ = preview(yaml_text, pattern="foo; rm -rf /")
    assert cmd == "echo 'foo; rm -rf /'"


def test_shell_list_elements_quoted_individually() -> None:
    """Shell mode: list elements are quoted individually then space-joined.

    Each element is a separate shell token — quoting the joined string would
    turn multiple arguments into one quoted blob.
    """
    yaml_text = """
name: echo_args
shell: bash
command: "echo {args}"
args:
  - name: args
    type: list
"""
    cmd, _ = preview(yaml_text, args=["a b", "c"])
    assert cmd == "echo 'a b' c"


def test_shell_value_without_metacharacters_not_quoted() -> None:
    """A plain alphanumeric value needs no quoting — ``shlex.quote`` passes it through."""
    yaml_text = """
name: t
shell: bash
command: "echo {name}"
args:
  - name: name
    type: str
"""
    cmd, _ = preview(yaml_text, name="hello")
    assert cmd == "echo hello"


def test_argv_value_with_metacharacters_not_quoted() -> None:
    """Argv mode is inherently safe (``shell=False``) — no quoting applied."""
    yaml_text = """
name: t
command: ["echo", "{val}"]
args:
  - name: val
    type: str
"""
    cmd, _ = preview(yaml_text, val="foo; rm -rf /")
    assert cmd == ["echo", "foo; rm -rf /"]


def test_bool_arg_with_to_flag_renders_when_true() -> None:
    """A bool arg with ``to: --flag`` renders the flag when true (story 12)."""
    yaml_text = """
name: uv_add
shell: bash
command: "uv add {pkg} {is_dev}"
args:
  - name: pkg
    type: str
  - name: is_dev
    type: bool
    to: --dev
"""
    cmd, _ = preview(yaml_text, pkg="requests", is_dev=True)
    assert cmd == "uv add requests --dev"


def test_bool_arg_with_to_flag_renders_empty_when_false() -> None:
    """A bool arg with ``to: --flag`` renders nothing when false (story 12)."""
    yaml_text = """
name: uv_add
shell: bash
command: "uv add {pkg} {is_dev}"
args:
  - name: pkg
    type: str
  - name: is_dev
    type: bool
    to: --dev
"""
    cmd, _ = preview(yaml_text, pkg="requests", is_dev=False)
    assert cmd == "uv add requests "


def test_bool_flag_ignored_for_non_bool_value() -> None:
    """``to:`` only fires for bool values; a string value passes through."""
    yaml_text = """
name: t
shell: bash
command: "echo {val}"
args:
  - name: val
    type: str
    to: --should-not-appear
"""
    cmd, _ = preview(yaml_text, val="hello")
    assert cmd == "echo hello"


# ====================================================================
# Platform selection (platforms: map)
# ====================================================================


def test_platform_overrides_command_for_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The current platform's branch overrides the top-level command."""
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "linux")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    yaml_text = """
name: t
description: t.
command: ["echo", "default"]
platforms:
  linux:
    command: ["echo", "from-linux"]
"""
    import asyncio
    assert asyncio.run(load_yaml_tools(yaml_text)[0]()) == "from-linux\n"


def test_unix_covers_linux_and_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    """``unix:`` matches both linux and macOS when no exact key exists."""
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    yaml_text = """
name: t
description: t.
command: ["echo", "default"]
platforms:
  unix:
    command: ["echo", "from-unix"]
"""
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "linux")
    import asyncio
    assert asyncio.run(load_yaml_tools(yaml_text)[0]()) == "from-unix\n"
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "macos")
    import asyncio
    assert asyncio.run(load_yaml_tools(yaml_text)[0]()) == "from-unix\n"


def test_exact_platform_wins_over_unix(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exact key (``linux:``) takes precedence over the ``unix:`` fallback."""
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "linux")
    yaml_text = """
name: t
command: ["echo", "default"]
platforms:
  unix:
    command: ["echo", "from-unix"]
  linux:
    command: ["echo", "from-linux"]
"""
    import asyncio
    assert asyncio.run(load_yaml_tools(yaml_text)[0]()) == "from-linux\n"


def test_platform_inherits_top_level_command_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A platform entry without ``command:`` inherits the top-level command."""
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "linux")
    yaml_text = """
name: t
command: ["echo", "inherited"]
platforms:
  linux: {}
"""
    import asyncio
    assert asyncio.run(load_yaml_tools(yaml_text)[0]()) == "inherited\n"


def test_unknown_platform_key_rejected() -> None:
    """Only ``linux``/``macos``/``unix``/``windows`` are valid platform keys."""
    yaml_text = """
name: t
command: ["echo", "hi"]
platforms:
  freebsd:
    command: ["echo", "nope"]
"""
    with pytest.raises(ValueError, match="unknown field.*freebsd"):
        load_yaml_tools(yaml_text)


# ====================================================================
# Executable gating (argv[0] / shell:)
# ====================================================================


def test_argv_executable_missing_silently_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """argv[0] not on PATH → tool is not registered (returns empty list)."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    yaml_text = 'name: t\ncommand: ["nonexistent-bin-xyz", "arg"]\n'
    assert load_yaml_tools(yaml_text) == []


def test_shell_missing_silently_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared shell not on PATH → tool is not registered."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    yaml_text = """
name: t
shell: nonexistent-shell-xyz
command: "echo hi"
"""
    assert load_yaml_tools(yaml_text) == []


# ====================================================================
# Args schema discipline
# ====================================================================


def test_arg_description_reaches_schema() -> None:
    """A YAML arg's ``description:`` is carried into ``__cothis_schema__``.

    This is the core PRD promise: bypassing any-llm's lossy
    ``callable_to_tool`` so per-arg descriptions survive.
    """
    yaml_text = """
name: calc
description: Calculate something.
command: ["echo", "{n}"]
args:
  - name: n
    type: int
    description: The number to calculate with.
"""
    schema = _shell_tool(yaml_text).__cothis_schema__
    assert schema is not None
    assert schema["description"] == "Calculate something."
    assert schema["input_schema"]["properties"]["n"]["description"] == (
        "The number to calculate with."
    )
    assert schema["input_schema"]["properties"]["n"]["type"] == "integer"
    assert schema["input_schema"]["required"] == ["n"]


def test_declared_but_unreferenced_arg_dropped_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An arg declared but not referenced by the command is dropped; warns.

    Point 1: prune the schema to what the command actually uses. The drop
    is silent in the schema but emits a ``UserWarning`` so the author hears
    about the dead declaration.
    """
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    yaml_text = """
name: t
command: ["echo", "{used}"]
args:
  - name: used
    type: str
  - name: unused
    type: str
"""
    with pytest.warns(UserWarning, match="undeclared arg.*unused|unused"):
        tool = _shell_tool(yaml_text)
    schema = tool.__cothis_schema__
    assert schema is not None
    props = schema["input_schema"]["properties"]
    assert "used" in props
    assert "unused" not in props


def test_per_platform_args_merge_override_same_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A platform branch's args override same-named top-level args, inherit rest.

    Point 1 (ii): branch ``args`` is an override, not a replacement. Same-named
    args take the branch's definition; other top-level args are inherited.
    """
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "linux")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    yaml_text = """
name: t
command: ["echo", "{a}", "{b}"]
args:
  - name: a
    type: str
    description: top-level a.
  - name: b
    type: str
    description: top-level b.
platforms:
  linux:
    args:
      - name: a
        type: str
        description: linux-specific a.
"""
    schema = _shell_tool(yaml_text).__cothis_schema__
    assert schema is not None
    props = schema["input_schema"]["properties"]
    assert props["a"]["description"] == "linux-specific a."
    assert props["b"]["description"] == "top-level b."


def test_merge_arg_specs_order_and_override() -> None:
    """``_merge_arg_specs`` keeps base order, appends new, overrides same-named."""
    base = [{"name": "a", "type": "str"}, {"name": "b", "type": "str"}]
    override = [{"name": "b", "type": "int"}, {"name": "c", "type": "str"}]
    merged = _merge_arg_specs(base, override)
    assert [m["name"] for m in merged] == ["a", "b", "c"]
    assert merged[1]["type"] == "int"  # b overridden


def test_unknown_arg_type_rejected() -> None:
    """An unknown ``type:`` is rejected, listing the legal types."""
    yaml_text = """
name: t
command: ["echo", "{x}"]
args:
  - name: x
    type: float
"""
    with pytest.raises(ValueError, match="unknown type 'float'"):
        load_yaml_tools(yaml_text)


def test_name_and_description_stringified() -> None:
    """Non-string ``name`` / ``description`` scalars are coerced to strings."""
    tool = load_yaml_tools("name: 42\ncommand: ['echo', 'hi']\n")[0]
    assert tool.__name__ == "42"
    assert isinstance(tool.__name__, str)
    tool2 = load_yaml_tools("name: t\ndescription: 123\ncommand: ['echo', 'hi']\n")[0]
    assert tool2.__doc__ == "123"
    assert isinstance(tool2.__doc__, str)


# ====================================================================
# Error / schema discipline (extra="forbid")
# ====================================================================


def test_unknown_top_level_field_rejected() -> None:
    """A typo'd top-level key is rejected, not silently ignored."""
    with pytest.raises(ValueError, match="unknown field.*frobnicate"):
        load_yaml_tools("name: t\ncommand: ['echo', 'hi']\nfrobnicate: yes\n")


def test_unknown_arg_field_rejected() -> None:
    """An arg entry with an unknown key is rejected, naming the key."""
    yaml_text = """
name: t
command: ["echo", "{x}"]
args:
  - name: x
    type: int
    bogus: yes
"""
    with pytest.raises(ValueError, match="unknown field.*bogus"):
        load_yaml_tools(yaml_text)


def test_missing_name_names_field() -> None:
    """A spec without ``name:`` raises ``ValueError`` naming the field."""
    with pytest.raises(ValueError, match="must define 'name'"):
        load_yaml_tools('command: ["echo", "hi"]\n')


def test_empty_file_raises_valueerror() -> None:
    """An empty YAML file raises, not ``TypeError`` (``None["name"]``)."""
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_yaml_tools("")


def test_command_wrong_type_rejected() -> None:
    """A non-str, non-list ``command:`` is rejected with a type-naming error."""
    with pytest.raises(ValueError, match="must be a string.*or a list"):
        load_yaml_tools("name: t\ncommand: 42\n")


def test_command_missing_rejected() -> None:
    """Missing ``command:`` is rejected (required field)."""
    with pytest.raises(ValueError, match="must define 'command'"):
        load_yaml_tools("name: t\n")


def test_platform_entry_null_rejected() -> None:
    """A ``platforms:`` entry that is null (not a mapping) is rejected.

    Previously ``platforms: {linux: null}`` would crash with ``TypeError``
    deep in ``_parse_command_block`` (``None.get('command')``) — not an
    actionable error. Now it names the offending platform key.
    """
    yaml_text = 'name: t\ncommand: ["echo", "hi"]\nplatforms:\n  linux: null\n'
    with pytest.raises(ValueError, match="platforms.linux.*must be a YAML mapping"):
        load_yaml_tools(yaml_text)


def test_argv_zeroth_element_with_placeholder_rejected() -> None:
    """argv[0] (the executable) must not contain ``{placeholder}``.

    Gating runs at load time, before args are available — a placeholder in
    argv[0] can never be resolved via ``shutil.which``, so the tool would be
    silently skipped. Reject it loudly so the author knows the executable
    must be a literal.
    """
    yaml_text = """
name: t
command: ["{exe}", "--version"]
args:
  - name: exe
    type: str
"""
    with pytest.raises(ValueError, match=r"argv\[0\].*must not contain placeholders"):
        load_yaml_tools(yaml_text)


def test_shell_field_with_argv_command_rejected() -> None:
    """``shell:`` with a list ``command:`` is meaningless and rejected.

    argv mode bypasses the shell entirely (``shell=False``), so declaring
    ``shell:`` is a misconception (the author probably wanted shell mode).
    Reject it rather than silently ignoring ``shell:``.
    """
    yaml_text = """
name: t
shell: bash
command: ["echo", "hi"]
"""
    with pytest.raises(ValueError, match=r"``shell:`` is meaningless with a list"):
        load_yaml_tools(yaml_text)


def test_preview_inherits_all_compile_checks() -> None:
    """``preview`` rejects every malformation ``load_yaml_tools`` does.

    Both paths share ``_compile``, so the invariants are enforced by
    construction — this test guards against a future refactor moving a
    check back into ``load_yaml_tools`` only. Covers the two cases preview
    previously missed (argv[0] placeholder, undeclared placeholder) plus
    shell-with-list (argv mode + ``shell:`` is meaningless).
    """
    # argv[0] placeholder — previously load-only; preview must now reject.
    argv0_placeholder = """
name: t
command: ["{exe}", "--version"]
args:
  - name: exe
    type: str
"""
    with pytest.raises(ValueError, match=r"argv\[0\].*must not contain placeholders"):
        preview(argv0_placeholder)

    # Undeclared placeholder — previously load-only; preview must now reject.
    undeclared = 'name: t\nshell: bash\ncommand: "echo {typo}"\n'
    with pytest.raises(ValueError, match="undeclared placeholder"):
        preview(undeclared)

    # String without shell — auto-selects OS default (story 16), no error.
    cmd, _ = preview('name: t\ncommand: "echo hi"\n')
    assert cmd == "echo hi"

    # List WITH shell — both still reject.
    with pytest.raises(ValueError, match="meaningless with a list"):
        preview('name: t\nshell: bash\ncommand: ["echo", "hi"]\n')


# ====================================================================
# load_tools_from_layer
# ====================================================================


def test_load_tools_from_layer_finds_yaml_files(tmp_path: Path) -> None:
    """``.agents/tools/*.yaml`` discovery: every YAML file in the dir loads."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "hello.yaml").write_text(
        'name: hello\ncommand: ["echo", "hello"]\n', encoding="utf-8"
    )
    (tools_dir / "bye.yml").write_text(
        'name: bye\ncommand: ["echo", "bye"]\n', encoding="utf-8"
    )
    (tools_dir / "README.md").write_text("not a tool", encoding="utf-8")

    tools = load_tools_from_layer(tools_dir)
    by_name = {t.__name__: t for t in tools}
    assert set(by_name) == {"hello", "bye"}
    import asyncio
    assert asyncio.run(by_name["hello"]()) == "hello\n"
    assert asyncio.run(by_name["bye"]()) == "bye\n"


def test_load_tools_from_layer_finds_nested_yaml(tmp_path: Path) -> None:
    """Discovery walks subdirectories so ``tools/date/current.yaml`` loads."""
    date_dir = tmp_path / "date"
    date_dir.mkdir()
    (date_dir / "current.yaml").write_text(
        'name: date.current\ncommand: ["echo", "today"]\n', encoding="utf-8"
    )
    (tmp_path / "flat.yaml").write_text(
        'name: flat\ncommand: ["echo", "flat"]\n', encoding="utf-8"
    )

    names = {t.__name__ for t in load_tools_from_layer(tmp_path)}
    assert names == {"date.current", "flat"}


def test_load_tools_from_layer_error_names_the_file(tmp_path: Path) -> None:
    """Loading from disk includes the file path in the error."""
    bad = tmp_path / "broken.yaml"
    bad.write_text('command: ["echo", "hi"]\n', encoding="utf-8")  # no name:
    with pytest.raises(ValueError, match="broken.yaml"):
        load_tools_from_layer(tmp_path)


def test_load_tools_from_layer_missing_dir_returns_empty(tmp_path: Path) -> None:
    """A missing directory yields ``[]``, not an error."""
    assert load_tools_from_layer(tmp_path / "nonexistent") == []


# ====================================================================
# preview() — verification surface
# ====================================================================


def test_preview_argv_returns_rendered_list() -> None:
    """``preview`` of an argv command returns the rendered list + ``None`` shell."""
    yaml_text = """
name: t
command: ["echo", "{x}"]
args:
  - name: x
    type: str
"""
    cmd, shell = preview(yaml_text, x="hi")
    assert cmd == ["echo", "hi"]
    assert shell is None


def test_preview_shell_returns_rendered_string_and_shell() -> None:
    """``preview`` of a shell command returns the rendered string + interpreter."""
    yaml_text = """
name: t
shell: bash
command: "echo {x}"
args:
  - name: x
    type: str
"""
    cmd, shell = preview(yaml_text, x="hi")
    assert cmd == "echo hi"
    assert shell == "bash"


def test_preview_platform_override_forces_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_platform="windows"`` forces the windows branch regardless of host."""
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    yaml_text = """
name: t
command: ["echo", "default"]
platforms:
  windows:
    shell: pwsh
    command: "echo from-windows"
"""
    cmd, shell = preview(yaml_text, _platform="windows")
    assert cmd == "echo from-windows"
    assert shell == "pwsh"


def test_preview_default_description_carries_through() -> None:
    """A tool without ``description:`` gets a default derived from its name."""
    tool = load_yaml_tools('name: bare\ncommand: ["echo", "hi"]\n')[0]
    assert tool.__doc__ == "Shell tool: bare"


def test_nonzero_exit_returns_error_with_stderr() -> None:
    """Stories 18/19: a non-zero exit surfaces as an error string, not a raise.

    The agent must see the failure to recover; an unhandled exception would
    crash the ReAct loop mid-turn.
    """
    tool = load_yaml_tools(
        'name: failer\ncommand: ["sh", "-c", "echo boom >&2; exit 3"]\n'
    )[0]
    import asyncio
    result = asyncio.run(tool())
    assert "exit code 3" in result
    assert "boom" in result


# ====================================================================
# Follow-up batch: preview _platform propagation, cmd quoting,
# stdout+stderr merge, name normalisation (Copilot A/B/C + story 11/21)
# ====================================================================


def test_preview_platform_override_propagates_to_shell_auto_select() -> None:
    """Copilot A: ``preview(_platform="windows")`` on a POSIX host
    auto-selects ``cmd`` (which is then rejected — #61 Option B), not
    ``sh``. The platform override reaches shell auto-selection, not just
    command-branch selection. Since ``cmd`` is now banned, the
    observable behaviour is that the windows branch raises at compile
    time; the linux branch still picks ``sh`` cleanly."""
    yaml_text = 'name: t\ncommand: "echo hi"\n'  # no shell: → auto-select
    # Windows auto-select now picks cmd, which Option B rejects.
    with pytest.raises(ValueError, match="'shell: cmd'"):
        preview(yaml_text, _platform="windows")
    cmd, shell_name = preview(yaml_text, _platform="linux")
    assert shell_name == "sh"
    assert cmd == "echo hi"


def test_cmd_shell_quoting_uses_list2cmdline_not_shlex() -> None:
    """Copilot B: ``_shell_quote`` for the ``cmd`` interpreter uses
    ``subprocess.list2cmdline`` (double-quoted), not ``shlex.quote``
    (single-quoted). ``shlex.quote`` would silently fail on ``cmd.exe``
    because cmd doesn't treat single quotes as quoting.

    The YAML pipeline rejects ``shell: cmd`` before render (#61 Option
    B), so ``_shell_quote``'s cmd branch is now unreachable from
    ``preview`` / ``load_yaml_tools``. This test exercises it directly
    to pin the quoting shape the retained defensive code produces — the
    branch stays as ceiling documentation + a defence-in-depth path for
    any future caller that builds a ``CommandBlock`` with ``shell=cmd``
    without going through ``_compile``.
    """
    from cothis.tools.yaml import _shell_quote
    assert _shell_quote("foo & bar", "cmd") == '"foo & bar"'
    # Sanity: a no-metachar value round-trips unchanged.
    assert _shell_quote("plain", "cmd") == "plain"


def test_shell_tool_name_with_spaces_normalised_to_dashes() -> None:
    """Story 11/21联动: ``name: uv add`` normalises to ``uv-add``.
    Provider function-name rules converge on ``[A-Za-z0-9_.-]``; spaces
    would break routing. Spaces → dashes (readable), logged at WARNING."""
    import logging

    yaml_text = """
name: uv add
command: ["echo", "hi"]
"""
    tool = load_yaml_tools(yaml_text)[0]
    assert tool.__name__ == "uv-add"


def test_shell_tool_name_with_spaces_emits_warning(caplog: Any) -> None:
    """The space→dash normalisation is observable: a WARNING fires naming
    the original and normalised forms so the author can fix it deliberately."""
    yaml_text = 'name: "my tool"\ncommand: ["echo", "hi"]\n'
    with caplog.at_level(logging.WARNING, logger="cothis.tools"):
        tool = load_yaml_tools(yaml_text)
    assert tool[0].__name__ == "my-tool"
    normalise_warnings = [r for r in caplog.records if "normalised" in r.message]
    assert len(normalise_warnings) == 1
    assert "my tool" in normalise_warnings[0].message
    assert "my-tool" in normalise_warnings[0].message


def test_shell_tool_name_strips_other_special_chars() -> None:
    """Characters outside ``[A-Za-z0-9_.-]`` (and not in ``_NAME_REPLACEMENTS``)
    are stripped, not replaced. ``"a!b"`` → ``"ab"``, not ``"a-b"``.
    Covers the strip arm distinct from the space/slash/colon→dash arm."""
    yaml_text = 'name: "a!b@c"\ncommand: ["echo", "hi"]\n'
    tool = load_yaml_tools(yaml_text)[0]
    assert tool.__name__ == "abc"


def test_shell_tool_name_empty_after_normalisation_rejected() -> None:
    """A name that normalises to empty (all special chars) raises — a tool
    with no callable name is unusable."""
    yaml_text = 'name: "!!"\ncommand: ["echo", "hi"]\n'
    with pytest.raises(ValueError, match="normalises to empty"):
        load_yaml_tools(yaml_text)


def test_shell_tool_name_with_alnum_unchanged() -> None:
    """A normal ``fs.read``-style name passes through unchanged."""
    yaml_text = 'name: fs.read\ncommand: ["echo", "hi"]\n'
    tool = load_yaml_tools(yaml_text)[0]
    assert tool.__name__ == "fs.read"


# --- finding #1: argv empty-element drop -------------------------------


def test_argv_bool_false_drops_empty_element() -> None:
    """In argv mode, a bool ``to:`` flag rendered false (→ ``""``) is dropped
    from the argv list so it never reaches the subprocess as an empty
    positional. ``uv add requests ''`` must not happen."""
    yaml_text = """
name: uv_add
command: ["uv", "add", "{pkg}", "{is_dev}"]
args:
  - name: pkg
    type: str
  - name: is_dev
    type: bool
    to: --dev
"""
    cmd, _ = preview(yaml_text, pkg="requests", is_dev=False)
    assert cmd == ["uv", "add", "requests"]


def test_argv_bool_true_keeps_flag_element() -> None:
    """In argv mode, a bool ``to:`` flag rendered true keeps the flag."""
    yaml_text = """
name: uv_add
command: ["uv", "add", "{pkg}", "{is_dev}"]
args:
  - name: pkg
    type: str
  - name: is_dev
    type: bool
    to: --dev
"""
    cmd, _ = preview(yaml_text, pkg="requests", is_dev=True)
    assert cmd == ["uv", "add", "requests", "--dev"]


# --- finding #3: non-ASCII tool name rejection --------------------------


def test_non_ascii_cjk_name_rejected() -> None:
    """A CJK name (``部署``) is stripped to empty and rejected — ``isalnum``
    alone admits non-ASCII, which passes load then fails at the provider API."""
    yaml_text = 'name: "部署"\ncommand: ["echo", "hi"]\n'
    with pytest.raises(ValueError, match="normalises to empty"):
        load_yaml_tools(yaml_text)


def test_non_ascii_accented_name_stripped() -> None:
    """Accented Latin chars are stripped (not passed through)."""
    yaml_text = 'name: "déploy"\ncommand: ["echo", "hi"]\n'
    tool = load_yaml_tools(yaml_text)[0]
    assert tool.__name__ == "dploy"


# ====================================================================
# Async dispatch — _ShellTool.__call__ must not block the loop (#90)
# ====================================================================


@pytest.mark.asyncio
async def test_shell_tool_does_not_block_event_loop() -> None:
    """A long subprocess must not freeze concurrent async tasks (#90).

    ``_ShellTool.__call__`` runs ``subprocess.run`` via
    ``asyncio.to_thread`` so the event loop stays responsive during
    the full subprocess lifetime.
    """
    import asyncio
    import time

    tool = load_yaml_tools('name: sleep1\ncommand: ["sleep", "1"]\n')[0]
    ticks: list[float] = []
    start = time.perf_counter()

    async def ticker() -> None:
        for _ in range(6):
            await asyncio.sleep(0.2)
            ticks.append(round(time.perf_counter() - start, 2))

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0.1)  # let the ticker start
    # mimics agent.py:1087 — tool() returns a coroutine; agent awaits.
    result = tool()
    if hasattr(result, "__await__"):
        await result
    await task

    # The ticker fires roughly every 0.2s. With the fix, several
    # ticks land *during* the 1s sleep (before t=1.0). Without the
    # fix the loop blocks for the full second and the first tick
    # lands at ~1.1s — i.e. zero ticks before t=1.0.
    ticks_during_sleep = [t for t in ticks if t < 1.0]
    assert len(ticks_during_sleep) >= 2, (
        f"loop stalled during subprocess: ticks={ticks}"
    )


# ====================================================================
# cmd shell rejection — Option B (#61, GH#139)
# ====================================================================


def test_cmd_shell_with_string_arg_rejected_at_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``shell: cmd`` raises ``ValueError`` at compile time (#61, GH#139).

    cmd.exe quoting is partial defence — ``&`` / ``|`` / ``%`` in arg
    values are live metacharacters ``list2cmdline`` does not neutralise,
    and a load-time WARNING (PR #137, Option A) cannot prevent the
    injection. Option B closes the ceiling permanently: ``_compile``
    rejects ``shell: cmd`` outright, naming the tool, the source, and
    both safe alternatives (``shell: pwsh`` / argv mode).
    """
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "windows")
    yaml_text = (
        "name: git-branch\n"
        "command: git branch {name}\n"
        "shell: cmd\n"
        "args:\n"
        "  - name: name\n"
        "    type: str\n"
    )
    with pytest.raises(ValueError) as excinfo:
        load_yaml_tools(yaml_text)
    msg = str(excinfo.value)
    # Error names the tool + source + both safe alternatives.
    assert "git-branch" in msg
    assert "'shell: cmd'" in msg
    assert "pwsh" in msg
    assert "argv" in msg


def test_cmd_shell_rejection_names_source(tmp_path: Path) -> None:
    """The reject error includes the source file when loaded from disk."""
    bad = tmp_path / "evil.yaml"
    bad.write_text(
        "name: t\ncommand: echo {x}\nshell: cmd\n"
        "args:\n  - name: x\n    type: str\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shell: cmd") as excinfo:
        load_tools_from_layer(tmp_path)
    assert "evil.yaml" in str(excinfo.value)


def test_cmd_shell_rejected_without_string_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``shell: cmd`` is rejected even with NO string args.

    Option A only warned when string args were present; Option B rejects
    unconditionally — a cmd tool with only int/bool args today is a
    string-arg injection tomorrow, and the cmd.exe quoting ceiling is
    a property of the interpreter, not the current arg set.
    """
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "windows")
    yaml_text = (
        "name: t\n"
        "command: echo {n}\n"
        "shell: cmd\n"
        "args:\n"
        "  - name: n\n"
        "    type: int\n"
    )
    with pytest.raises(ValueError, match="'shell: cmd'"):
        load_yaml_tools(yaml_text)


def test_cmd_shell_rejected_with_no_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``shell: cmd`` is rejected even for a no-arg command.

    A no-arg cmd tool can't inject today, but rejecting only the
    arg-bearing case would let a cmd tool slip through and gain an arg
    later without re-triggering the check. The interpreter is unsafe;
    reject it uniformly.
    """
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "windows")
    with pytest.raises(ValueError, match="'shell: cmd'"):
        load_yaml_tools("name: t\ncommand: ver\nshell: cmd\n")


def test_preview_rejects_cmd_shell() -> None:
    """``preview`` shares ``_compile`` and rejects ``shell: cmd`` too.

    The two paths cannot drift on what a valid YAMLTool is — both go
    through ``_compile``, so the cmd ban holds for preview as well.
    """
    yaml_text = (
        "name: t\ncommand: echo {x}\nshell: cmd\n"
        "args:\n  - name: x\n    type: str\n"
    )
    with pytest.raises(ValueError, match="'shell: cmd'"):
        preview(yaml_text)


def test_pwsh_shell_accepted_on_windows(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``shell: pwsh`` is unaffected — the recommended Windows interpreter.

    PowerShell single-quote quoting (via ``shlex.quote``) is sound, so
    pwsh loads cleanly with no cmd warning and no rejection. This pins
    the unaffected pwsh path required by the acceptance criteria.
    """
    import logging

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "windows")
    yaml_text = (
        "name: git-branch\n"
        "command: git branch {name}\n"
        "shell: pwsh\n"
        "args:\n"
        "  - name: name\n"
        "    type: str\n"
    )
    with caplog.at_level(logging.WARNING, logger="cothis.tools.yaml"):
        tool = load_yaml_tools(yaml_text)[0]
    assert tool.__name__ == "git-branch"
    cmd_warnings = [
        r for r in caplog.records
        if "cmd" in r.message and "metacharacter" in r.message
    ]
    assert cmd_warnings == []


def test_argv_mode_unaffected_no_cmd_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Argv mode (``command: [list]``) is unaffected — inherently safe.

    ``shell=False`` / ``execve`` does its own tokenisation, so no shell
    quoting is involved and the cmd ban never applies. Pins the
    unaffected argv path required by the acceptance criteria.
    """
    import logging

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    yaml_text = (
        'name: echo-argv\n'
        'command: ["echo", "{msg}"]\n'
        'args:\n'
        '  - name: msg\n'
        '    type: str\n'
    )
    with caplog.at_level(logging.WARNING, logger="cothis.tools.yaml"):
        tool = load_yaml_tools(yaml_text)[0]
    assert tool.__name__ == "echo-argv"
    cmd_warnings = [
        r for r in caplog.records
        if "cmd" in r.message and "metacharacter" in r.message
    ]
    assert cmd_warnings == []


def test_posix_sh_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """``shell: sh`` (POSIX default) is unaffected — ``shlex.quote`` is sound."""
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "linux")
    yaml_text = 'name: t\ncommand: "echo {x}"\nargs:\n  - name: x\n    type: str\n'
    tool = _shell_tool(yaml_text)  # auto-selects sh on POSIX, no raise
    assert tool._block.shell == "sh"


def test_default_shell_autoselected_cmd_on_windows_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Author omits ``shell:`` on Windows → auto-selected ``cmd`` is rejected (#61).

    The auto-select path (story 16) still picks ``cmd`` on Windows when
    ``shell:`` is omitted. With Option B that auto-selected cmd is
    rejected too — a Windows author who omits ``shell:`` MUST declare
    ``pwsh`` or switch to argv mode. The error message is the only
    migration signal, so it must fire here.
    """
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "windows")
    yaml_text = (
        "name: git-branch\n"
        "command: git branch {name}\n"
        "args:\n"
        "  - name: name\n"
        "    type: str\n"
    )
    with pytest.raises(ValueError, match="'shell: cmd'"):
        load_yaml_tools(yaml_text)


def test_windows_argv_command_without_shell_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows + argv mode (no ``shell:``) is accepted — argv needs no shell.

    The cmd ban only applies to shell mode. A Windows author who omits
    ``shell:`` and uses a list ``command:`` is on the safe argv path
    regardless of platform, so the tool loads. This is the argv-mode
    migration target the error message points Windows authors to.
    """
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("cothis.tools.yaml._current_platform", lambda: "windows")
    tool = _shell_tool('name: t\ncommand: ["echo", "hi"]\n')
    assert tool._block.shell is None  # argv mode, never reached cmd auto-select


# ====================================================================
# _shell_argv: shell-mode argv construction (Windows path-with-spaces fix)
# ====================================================================


def test_shell_argv_cmd_returns_none() -> None:
    """cmd.exe returns ``None`` so dispatch keeps the ``shell=True`` path.

    cmd.exe is the one interpreter dispatched via ``shell=True`` rather than
    argv mode: ``_shell_quote`` cmd-quotes values with ``list2cmdline``, and
    ``shell=True`` passes that rendered string to cmd.exe verbatim so the
    quoting round-trips. argv-mode dispatch would re-apply ``list2cmdline``
    and double-escape the inner double-quotes. cmd.exe also lives in a
    spaces-free path, so the path-split bug argv mode exists to fix does
    not apply.
    """
    assert _shell_argv(r"C:\Windows\System32\cmd.exe", "cmd", "echo hi") is None


def test_shell_argv_cmd_absolute_path_returns_none() -> None:
    r"""An absolute ``cmd.exe`` path passed as ``shell_name`` still matches,
    via the ``\cmd.exe`` suffix (defensive — ``shell:`` is usually a bare
    name, but a full path is accepted)."""
    assert _shell_argv(
        r"C:\Windows\System32\cmd.exe", r"C:\Windows\System32\cmd.exe", "echo hi"
    ) is None


def test_shell_argv_pwsh_uses_dash_command() -> None:
    r"""pwsh mode builds ``[exe, "-Command", command]``.

    PowerShell's script-string flag is ``-Command`` (not cmd's ``/c``). No
    ``-NoProfile`` is injected: that would silently drop the user's
    ``$PROFILE`` (aliases, env, loaded modules). ``-NoProfile`` is the
    author's call, not cothis's.
    """
    argv = _shell_argv(r"C:\Program Files\PowerShell\pwsh.EXE", "pwsh", "Get-Date")
    assert argv == [r"C:\Program Files\PowerShell\pwsh.EXE", "-Command", "Get-Date"]


def test_shell_argv_powershell_alias_matches_pwsh() -> None:
    r"""Windows PowerShell 5.1 (``powershell``) uses the same ``-Command`` flag."""
    argv = _shell_argv(
        r"C:\WINDOWS\System32\WindowsPowerShell1.0\powershell.EXE",
        "powershell",
        "Get-Date",
    )
    assert argv is not None
    assert argv[1:] == ["-Command", "Get-Date"]


def test_shell_argv_pwsh_does_not_substring_match() -> None:
    """A name merely containing ``pwsh`` (e.g. ``mypwshwrapper``) does NOT
    route to the PowerShell ``-Command`` branch — it falls through to the
    POSIX ``-c`` default. Substring matching would silently hand the wrong
    flag to an unrelated interpreter."""
    argv = _shell_argv("/usr/bin/mypwshwrapper", "mypwshwrapper", "echo hi")
    assert argv == ["/usr/bin/mypwshwrapper", "-c", "echo hi"]


def test_shell_argv_posix_uses_dash_c() -> None:
    """POSIX shells (sh / bash / zsh / dash) get ``-c`` per POSIX."""
    for name in ("sh", "bash", "zsh", "dash"):
        argv = _shell_argv(f"/usr/bin/{name}", name, "echo hi")
        assert argv == [f"/usr/bin/{name}", "-c", "echo hi"], name


def test_shell_argv_unknown_falls_back_to_posix_dash_c() -> None:
    """An unrecognised shell name falls back to the POSIX ``-c`` form.

    Fallback is safe-by-construction: ``-c`` is the POSIX script-string
    flag, and the path is whatever ``shutil.which`` resolved. Unknown
    shells are rare (every common interpreter is enumerated above).
    """
    argv = _shell_argv("/usr/bin/fictional-shell", "fictional-shell", "echo hi")
    assert argv == ["/usr/bin/fictional-shell", "-c", "echo hi"]


def test_shell_argv_none_shell_name_falls_back_to_posix() -> None:
    """``shell_name=None`` falls back to the POSIX ``-c`` form.

    Defensive: ``_ShellTool`` is only constructed for shell mode (where
    ``block.shell`` is always set), but the helper tolerates None so
    unit tests and future callers don't need to fabricate a name.
    """
    argv = _shell_argv("/usr/bin/sh", None, "echo hi")
    assert argv == ["/usr/bin/sh", "-c", "echo hi"]


def test_shell_argv_path_with_spaces_survives() -> None:
    r"""A shell_path containing spaces reaches the child intact.

    The common Windows case is ``C:\Program Files\PowerShell\pwsh.EXE``.
    argv mode (``shell=False``) does not embed the exe path in
    ``lpCommandLine``, so Windows does not re-tokenise it. This test pins
    the argv shape that makes that true: the path is one element.
    """
    path = r"C:\Program Files\PowerShell\pwsh.EXE"
    argv = _shell_argv(path, "pwsh", "Get-Date -Format 'yyyy-MM-dd'")
    assert argv is not None
    assert argv[0] == path
    assert len(argv) == 3  # [exe, -Command, script]


def test_cmd_dispatch_keeps_shell_true_and_preserves_inner_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cmd.exe dispatch keeps ``shell=True`` and preserves inner double-quotes.

    In cmd shell mode, ``_shell_quote`` cmd-quotes each value with
    ``list2cmdline`` (``foo & bar`` → ``\"foo & bar\"``). ``__call__`` must
    dispatch this on the ``shell=True`` path, not argv mode: argv-mode
    ``subprocess.run(list, shell=False)`` would re-apply ``list2cmdline``
    to the whole argv and escape the inner double-quotes as backslash-quote,
    which cmd.exe does not honour (the ``&`` would become a live
    metacharacter).

    The YAML pipeline rejects ``shell: cmd`` at compile time (#61 Option
    B), so this dispatch path is now unreachable from ``load_yaml_tools``
    — the cmd branch in ``_ShellTool.__call__`` is retained as defensive
    code (see the ceiling note on ``_shell_quote``). To exercise it
    without going through ``_compile``, we build a ``CommandBlock`` with
    ``shell=\"cmd\"`` directly and drive ``_ShellTool`` from there.

    Captures the exact ``subprocess.run`` call (``__call__`` is async, so
    the blocking call runs via ``asyncio.to_thread``) and pins
    ``shell=True``, the resolved executable, and the rendered string's
    inner quotes.
    """
    import asyncio

    import cothis.tools.yaml as yaml_mod

    # Build a cmd-shell CommandBlock directly — ``_compile`` now rejects
    # ``shell: cmd`` (Option B), but the dispatch branch is retained as
    # defensive code, so we bypass compile to exercise it.
    block = yaml_mod.CommandBlock(
        name="echo_cmd",
        description="Shell tool: echo_cmd",
        command="echo {pattern}",
        shell="cmd",
        arg_specs=[{"name": "pattern", "type": "str"}],
    )
    tool = yaml_mod._ShellTool(block, shell_path=r"C:\Windows\System32\cmd.exe")

    captured: dict[str, Any] = {}

    class _FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _spy_run(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(yaml_mod.subprocess, "run", _spy_run)
    asyncio.run(tool(pattern="foo & bar"))

    # Dispatch keeps shell=True (the cmd.exe path), not shell=False with a
    # re-tokenised argv.
    assert captured["kwargs"].get("shell") is True
    assert captured["kwargs"].get("executable") == r"C:\Windows\System32\cmd.exe"
    # The rendered command preserves the cmd-quoted inner double-quotes
    # (``"foo & bar"``), not backslash-escaped ones.
    assert captured["args"] == ('echo "foo & bar"',)
    assert '\\"' not in captured["args"][0]
