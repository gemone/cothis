"""YAML shell-tool pipeline — compile, render, gate, dispatch.

Extracted from ``core.py``. Owns the YAMLTool concern end-to-end:
``load_yaml_tools`` is the entry point (``type:`` routing + compile + gate
+ wrap), ``_compile`` is the single validation pass shared with ``preview``,
``CommandBlock`` is the validated artifact, ``_ShellTool`` is the
Tool-protocol adapter over it.

Validation helpers shared with MCP (``_require`` / ``_check_unknown_keys``)
stay in ``core.py``; the YAML-specific ones (``_TOOL_KEYS`` /
``_ARG_TYPES`` / ``_validate_arg_type`` / ``_build_signature`` /
``_build_tool_schema``) live here.

Lazy import of ``cothis.tools.mcp`` builders inside ``load_yaml_tools``
breaks the import cycle (mcp → core is module-level; yaml → mcp is
function-level).
"""

from __future__ import annotations

import asyncio
import inspect
import shlex
import shutil
import string
import subprocess
import sys
import warnings
from typing import TYPE_CHECKING, Any

import yaml

from cothis.tools.core import (
    _check_unknown_keys,
    _HookableTool,
    _require,
    logger,
)

if TYPE_CHECKING:
    from cothis.tools.core import Tool


def load_yaml_tools(yaml_text: str, *, source: str | None = None) -> list[Tool]:
    """Compile a YAML tool declaration into a callable.

    Routes on ``type:``. A declaration with ``type: mcp.stdio`` (or
    ``type: mcp.http``) is an MCP server (an ``MCPServer`` handle producing
    many tools at Agent startup, not one shell tool) — see
    ``_build_mcp_stdio_server`` / ``_build_mcp_http_server`` and ADR-0005. An
    unknown ``type:`` value raises ``ValueError`` naming the file + value +
    valid options (story 30). A declaration with ``type:`` absent is a
    shell-template tool: compile, gate the executable via ``shutil.which``,
    wrap in ``_ShellTool``. If the executable is not on PATH the tool is not
    registered — the model never sees a tool it cannot dispatch on this host.
    The skip is logged at ``WARNING`` (every startup decision is observable
    by default — see CONTEXT.md "Tool lifecycle").

    See ``_compile`` for the shell-YAML shape and ``CommandBlock`` for the
    contract. ``preview`` shares the same compile path, so the two cannot
    drift on what a valid YAMLTool is.
    """
    from cothis.tools.mcp import _build_mcp_http_server, _build_mcp_stdio_server

    # Peek before ``_compile`` — it's shell-only and would reject MCP keys.
    spec = yaml.safe_load(yaml_text)
    if isinstance(spec, dict) and spec.get("type") == "mcp.stdio":
        return [_build_mcp_stdio_server(spec, source)]
    if isinstance(spec, dict) and spec.get("type") == "mcp.http":
        return [_build_mcp_http_server(spec, source)]
    if isinstance(spec, dict) and "type" in spec:
        where = f" in {source}" if source else ""
        msg = (
            f"unknown tool type {spec['type']!r}{where}; "
            f"valid: 'mcp.stdio', 'mcp.http', or omit 'type:' for a shell tool"
        )
        raise ValueError(msg)
    block = _compile(yaml_text, source=source)
    exe = shutil.which(block.gate_target)
    if exe is None:
        logger.warning(
            "tool %r gated off: %s not on PATH", block.name, block.gate_target
        )
        return []
    return [_ShellTool(block, shell_path=exe if block.shell else None)]


class CommandBlock:
    """A fully-validated, platform-selected YAMLTool, ready to render or dispatch.

    Produced by ``_compile``; consumed by ``load_yaml_tools`` (wraps it + a
    resolved executable into ``_ShellTool``) and ``preview`` (renders it
    directly). Both call ``_compile``, so the YAMLTool contract lives in
    exactly one place — the drift between load and preview is closed by
    construction.

    The resolved executable path is deliberately NOT carried: Gating is
    ``load_yaml_tools``'s concern, so ``preview`` can render any platform's
    branch regardless of host PATH.

    Execution mode (ADR-0001) is carried by ``command``'s type: ``list[str]``
    = argv mode, ``str`` = shell mode. ``shell`` is the declared interpreter
    name in shell mode, ``None`` in argv mode — locked to the command type
    by ``_compile``.

    Distinct from ``_CommandBlock``: that is the transient per-level parse
    triple (command/shell/args, no identity) produced by
    ``_parse_command_block`` and consumed by ``_select_platform``. This
    class is the post-validation whole (with name/description/gate_target/
    render) produced by ``_compile``.
    """

    __slots__ = ("name", "description", "command", "shell", "arg_specs", "_arg_names")

    def __init__(
        self,
        *,
        name: str,
        description: str,
        command: list[str] | str,
        shell: str | None,
        arg_specs: list[dict[str, Any]],
    ) -> None:
        self.name = name
        self.description = description
        self.command = command
        self.shell = shell
        self.arg_specs = arg_specs
        self._arg_names = [a["name"] for a in arg_specs]

    @property
    def gate_target(self) -> str:
        """The executable name Gating must find on PATH.

        argv mode → ``command[0]`` (validated placeholder-free at compile
        time). shell mode → ``shell`` (the declared interpreter). Read by
        ``load_yaml_tools`` to resolve+gate; ``preview`` does not read it.
        Hides the argv[0]-vs-interpreter fork behind one named field.
        """
        if isinstance(self.command, list):
            return self.command[0]
        # Assert narrows ``shell`` to ``str`` for the type checker; ``_compile`` guarantees non-None in shell mode.
        assert self.shell is not None
        return self.shell

    def render(self, **kwargs: Any) -> list[str] | str:
        """Substitute Placeholders into the selected command.

        argv mode → ``[rendered element per argv item]`` (list preserved).
        shell mode → rendered template string. Uses ``str.format_map``
        semantics (``{name}`` substitutes, ``{{`` escapes a literal ``{``,
        ``{n:03d}`` / ``{p!r}`` honoured). A referenced arg missing from
        ``kwargs`` raises ``KeyError`` — a caller-side contract violation,
        not a compile-time bug.

        Pure: no subprocess, no filesystem, no gating. The list-vs-string
        render fork lives here and only here. Shell-mode values are quoted
        for THIS tool's declared interpreter (POSIX vs ``cmd``) so a value
        with spaces or metacharacters cannot inject (story 22).
        """
        shell_mode = isinstance(self.command, str)
        mapping = _value_mapping(
            self.arg_specs, kwargs, shell=self.shell if shell_mode else None
        )
        if isinstance(self.command, list):
            # argv mode: drop elements that render to empty (e.g. a ``to:``
            # bool flag rendered false → ``""``). An empty string passed to
            # a subprocess is a real positional argument and breaks commands
            # like ``uv add requests ''`` (finding: ``to:`` flag injection).
            return [
                rendered for rendered in (part.format_map(mapping) for part in self.command)
                if rendered != ""
            ]
        return self.command.format_map(mapping)


def _compile(
    yaml_text: str,
    *,
    source: str | None = None,
    platform: str | None = None,
) -> CommandBlock:
    """Validate a YAML tool declaration and select a branch.

    Runs ALL validation that determines whether the spec is a well-formed
    YAMLTool — the contract both ``load_yaml_tools`` and ``preview`` share:
    unknown keys, required fields, command shape, arg types, placeholder
    discipline, execution-mode pairing rules. Then selects the platform
    branch (``platform`` overrides ``sys.platform`` detection; used by
    ``preview`` to render any branch from any host) and merges args.

    Does NOT gate: ``shutil.which`` is ``load_yaml_tools``'s concern, so
    ``preview`` can compile any branch regardless of host executables.
    The gated name is exposed via ``CommandBlock.gate_target``.

    Raises ``ValueError`` on any contract violation (message names the tool,
    the field, and ``source``). Emits ``UserWarning`` for declared-but-
    unreferenced args (``preview`` suppresses; ``load_yaml_tools`` surfaces).
    """
    spec = yaml.safe_load(yaml_text)
    _check_unknown_keys(spec, _TOOL_KEYS, source, what="YAML tool")
    name = _normalise_tool_name(str(_require(spec, "name", source)), source)
    description = _stringify(spec.get("description")) or f"Shell tool: {name}"

    top = _parse_command_block(spec, name, source, what="YAML tool")
    platforms_raw = spec.get("platforms") or {}
    _check_unknown_keys(
        platforms_raw, _PLATFORM_KEYS, source, what=f"tool {name!r}: platforms"
    )
    platforms = {
        plat: _parse_command_block(
            block,
            name,
            source,
            what=f"tool {name!r}: platforms.{plat}",
            require_command=False,
        )
        for plat, block in platforms_raw.items()
    }

    current = platform if platform is not None else _current_platform()
    selected = _select_platform(top, platforms, platform=current)

    merged_args = _merge_arg_specs(top.args, selected.args)
    declared = {a["name"] for a in merged_args}
    placeholders = _all_placeholders(selected.command)
    undeclared = placeholders - declared
    if undeclared:
        where = f" in {source}" if source else ""
        msg = (
            f"tool {name!r}: command references undeclared placeholder(s) "
            f"{sorted(undeclared)!r}{where}; declare them under 'args:'"
        )
        raise ValueError(msg)
    referenced = placeholders & declared
    final_args = [a for a in merged_args if a["name"] in referenced]
    unused = [a["name"] for a in merged_args if a["name"] not in referenced]
    if unused:
        where = f" in {source}" if source else ""
        warnings.warn(
            f"tool {name!r}: declared arg(s) {unused!r} not referenced by the "
            f"selected command{where}; dropped from schema",
            stacklevel=2,
        )

    if isinstance(selected.command, list):
        # argv[0] must be literal — gating runs at load time, before args are available.
        if _extract_field_names(selected.command[0]):
            where = f" in {source}" if source else ""
            msg = (
                f"tool {name!r}: argv[0] (the executable) must not contain "
                f"placeholders{where}; move the executable name out of {{...}}"
            )
            raise ValueError(msg)
        if selected.shell is not None:
            where = f" in {source}" if source else ""
            msg = (
                f"tool {name!r}: ``shell:`` is meaningless with a list "
                f"``command:`` (argv mode){where}; use a string ``command:`` "
                "if you need shell interpretation"
            )
            raise ValueError(msg)
    else:
        if not selected.shell:
            # Auto-select per selected platform when the author doesn't declare
            # one (story 16). Uses the resolved ``current`` (honours
            # ``platform`` / ``_platform`` overrides) — previewing the windows
            # branch from a POSIX host picks ``cmd``, not the host's ``sh``.
            selected.shell = "cmd" if current == "windows" else "sh"

    # cothis: cmd.exe visibility (#61). Surface the ``_shell_quote``
    # ceiling at load time so tool authors see the gap.
    if selected.shell == "cmd":
        string_args = [
            a["name"] for a in final_args
            if a.get("type", "str") == "str"
        ]
        if string_args:
            where = f" in {source}" if source else ""
            logger.warning(
                "tool %r uses shell: cmd with string arg(s) %s%s; "
                "cmd.exe metacharacters (&, |, %%) in values are NOT "
                "escaped — use shell: pwsh or command: [argv] for "
                "untrusted input.",
                name, string_args, where,
            )

    return CommandBlock(
        name=name,
        description=description,
        command=selected.command,
        shell=selected.shell,
        arg_specs=final_args,
    )


class _CommandBlock:
    """A parsed ``command`` / ``shell`` / ``args`` triple — the per-level parse
    artifact, before platform selection or identity is attached.

    Transient: produced by ``_parse_command_block`` (for the top-level spec
    AND each ``platforms:`` entry), consumed by ``_select_platform``. The
    selected triple is then promoted to a full ``CommandBlock`` by ``_compile``
    (which adds name/description/gate_target/render and filters arg_specs).

    ``command`` is either a ``list[str]`` (argv mode) or a ``str`` (shell
    mode) — the type determines the execution path. ``shell`` is the
    interpreter name for shell mode (``None`` for argv mode). ``args`` is
    the list of arg specs declared at this level. A platform entry may
    carry an empty-string ``command`` sentinel (inheritance resolved by
    ``_select_platform``).
    """

    __slots__ = ("command", "shell", "args")

    def __init__(
        self,
        command: list[str] | str,
        shell: str | None,
        args: list[dict[str, Any]],
    ) -> None:
        self.command = command
        self.shell = shell
        self.args = args


def _parse_command_block(
    spec: dict[str, Any],
    tool_name: str,
    source: str | None,
    *,
    what: str,
    require_command: bool = True,
) -> _CommandBlock:
    """Extract and validate ``command`` / ``shell`` / ``args`` from a mapping.

    Shared between the top-level tool spec and each ``platforms:`` entry so
    both follow the same shape rules. ``command`` is required at the
    top level; a platform entry may omit it (inherits the default).
    """
    where = f" in {source}" if source else ""
    if not isinstance(spec, dict):
        msg = f"{what}: must be a YAML mapping{where}; got {type(spec).__name__}"
        raise ValueError(msg)
    has_command = "command" in spec
    command_field = spec.get("command")
    if not has_command:
        if require_command:
            msg = f"{what}: must define 'command'{where}"
            raise ValueError(msg)
        command: list[str] | str = ""
    elif isinstance(command_field, str):
        command = command_field
    elif isinstance(command_field, list):
        if not command_field:
            msg = f"{what}: 'command' list is empty{where}"
            raise ValueError(msg)
        command = [str(c) for c in command_field]
    else:
        msg = (
            f"{what}: 'command' must be a string (shell mode) or a list "
            f"of strings (argv mode){where}; got {type(command_field).__name__}"
        )
        raise ValueError(msg)

    shell = str(spec["shell"]) if "shell" in spec else None
    arg_specs = spec.get("args") or []
    parsed_args: list[dict[str, Any]] = []
    for a in arg_specs:
        _check_unknown_keys(a, _ARG_KEYS, source, what=f"{what}: arg")
        arg_name = str(_require(a, "name", source, what=f"{what}: arg"))
        _validate_arg_type(a, tool_name, source)
        normalised = dict(a)
        normalised["name"] = arg_name
        parsed_args.append(normalised)

    return _CommandBlock(
        command=command if has_command else "", shell=shell, args=parsed_args
    )


def _select_platform(
    top: _CommandBlock,
    platforms: dict[str, _CommandBlock],
    *,
    platform: str | None = None,
) -> _CommandBlock:
    """Pick the command block for the requested platform.

    Resolution order: exact platform key (``linux``/``macos``/``windows``)
    → ``unix`` (covers linux+macOS) → top-level defaults. Returns a
    ``_CommandBlock`` whose ``command`` and ``shell`` inherit from the
    top-level block when the platform entry didn't set them. ``args`` are
    NOT merged here — that's ``_compile``'s concern (it needs the filtered,
    referenced set, not the raw per-level specs).

    ``platform`` defaults to the detected current platform; pass an override
    (``"windows"``) to preview a branch the host would not normally select.
    A platform entry with an empty ``command`` (sentinel from
    ``_parse_command_block``) inherits the top-level command.
    """
    current = platform if platform is not None else _current_platform()
    branch = platforms.get(current)
    if branch is None and current in ("linux", "macos"):
        branch = platforms.get("unix")
    if branch is None:
        return top
    merged_command = branch.command if branch.command != "" else top.command
    merged_shell = branch.shell if branch.shell is not None else top.shell
    return _CommandBlock(command=merged_command, shell=merged_shell, args=branch.args)


def _current_platform() -> str:
    """Map ``sys.platform`` to one of ``linux`` / ``macos`` / ``windows``."""
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _merge_arg_specs(
    base: list[dict[str, Any]], override: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge arg specs by name: ``override`` entries replace same-named ``base``.

    Order follows ``base`` (so the LLM sees a stable parameter order),
    with new args from ``override`` appended.
    """
    by_name = {a["name"]: a for a in base}
    order = [a["name"] for a in base]
    for a in override:
        if a["name"] not in by_name:
            order.append(a["name"])
        by_name[a["name"]] = a
    return [by_name[n] for n in order]


_FORMATTER = string.Formatter()


def _extract_field_names(template: str) -> set[str]:
    """Return the set of named fields referenced by a Python format-string.

    Uses ``string.Formatter().parse`` to walk the template's literal /
    replacement-field segments. Only the *base name* of each field is kept
    (``{a.b}`` → ``a``, ``{a[0]}`` → ``a``) — we don't support attribute /
    index access into arg values, but ``Formatter.parse`` yields the base
    name either way, and ``format_map`` will surface any misuse as a real
    ``AttributeError`` / ``KeyError`` at render time.

    Positional fields (``{0}``) yield ``''`` or a numeric string; callers
    treat them as undeclared (they're not valid for named args) so they
    surface as load-time errors.
    """
    names: set[str] = set()
    for _literal, field_name, _spec, _conv in _FORMATTER.parse(template):
        if field_name is None:
            continue
        base = field_name.split(".", 1)[0].split("[", 1)[0]
        names.add(base)
    return names


def _all_placeholders(command: list[str] | str) -> set[str]:
    """Return every named field referenced by ``command`` (Python format-string).

    Includes names that are NOT declared args — used to surface undeclared
    placeholders as a load-time error (typo / config drift). The
    intersection with declared args (done by the caller) gives the final
    schema's arg set.

    ``command`` may be a string (shell mode) or a list (argv mode); for a
    list, every element is parsed (an arg may appear in any element).
    """
    if isinstance(command, list):
        names: set[str] = set()
        for part in command:
            names |= _extract_field_names(part)
        return names
    return _extract_field_names(command)


class _ShellTool(_HookableTool):
    """A callable shell tool — the Tool-protocol adapter over a CommandBlock.

    Holds a ``CommandBlock`` (validated command + args) plus the resolved
    executable path (shell mode only). Satisfies the ``Tool`` protocol by
    exposing ``__name__`` / ``__doc__`` / ``__signature__`` and a pre-built
    ``__cothis_schema__`` (so per-arg descriptions reach the LLM without
    going through any-llm's lossy ``callable_to_tool``).

    Dispatch (type-driven, ADR-0001):
    - ``command`` is a list → argv mode: ``subprocess.run(list, shell=False)``.
    - ``command`` is a str → shell mode: ``subprocess.run(str, shell=True,
      executable=self._shell_path)``.

    Rendering delegates to ``CommandBlock.render`` — the list-vs-string
    render fork lives there, not here. This class carries only the dispatch
    fork (``subprocess.run``'s argument shape) and the resolved exe path.

    Inherits ``_HookableTool`` so ``_execute`` can run hooks uniformly without
    per-source branching. YAML tools don't register hooks today (their
    ``_hooks`` lists stay empty); the hook chains are no-ops. If YAML hook
    support is added later, it'll be a loader concern (e.g. load a same-name
    ``.py`` file), not an ``_execute`` change.
    """

    __name__: str
    __doc__: str
    __signature__: inspect.Signature
    __cothis_schema__: dict[str, Any] | None

    def __init__(self, block: CommandBlock, *, shell_path: str | None) -> None:
        super().__init__()
        self.__name__ = block.name
        self.__doc__ = block.description
        self.__signature__ = _build_signature(block.arg_specs)
        self.__cothis_schema__ = _build_tool_schema(
            block.name, block.description, block.arg_specs
        )
        self._block = block
        # None in argv mode — subprocess re-resolves argv[0] via PATH itself.
        self._shell_path = shell_path

    async def __call__(self, **kwargs: Any) -> str:
        rendered = self._block.render(**kwargs)
        # cothis: park the blocking ``subprocess.run`` off the loop
        # thread (#90).
        if isinstance(rendered, list):
            proc = await asyncio.to_thread(
                subprocess.run, rendered, shell=False,
                capture_output=True, text=True,
            )
        else:
            proc = await asyncio.to_thread(
                subprocess.run, rendered, shell=True,
                capture_output=True, text=True,
                executable=self._shell_path,
            )
        return _format_proc_result(proc)


def _format_proc_result(proc: Any) -> str:
    """Format a finished ``subprocess.CompletedProcess`` for the LLM.

    Success (exit 0): returns ``stdout``; appends ``stderr`` only when it's
    non-empty (deprecation warnings / progress notes the model benefits from).
    Failure (non-zero): returns ``Error: exit code N`` plus both streams,
    stdout first then stderr — the crash context (stdout emitted before the
    failure) is often the most actionable signal, and dropping it (the prior
    behaviour) made the LLM blind to why a command crashed mid-output.
    Story 18: capture stdout+stderr.
    """
    if proc.returncode == 0:
        if proc.stderr:
            return f"{proc.stdout}\n[stderr]\n{proc.stderr}"
        return proc.stdout
    parts = [f"Error: exit code {proc.returncode}"]
    if proc.stdout:
        parts.append(f"[stdout]\n{proc.stdout}")
    if proc.stderr:
        parts.append(f"[stderr]\n{proc.stderr}")
    return "\n".join(parts)


def preview(
    yaml_text: str,
    *,
    _platform: str | None = None,
    source: str | None = None,
    **kwargs: Any,
) -> tuple[list[str] | str, str | None]:
    """Return what a tool would run, without dispatching.

    Returns ``(command, shell)`` where ``command`` is the rendered argv list
    (argv mode) or shell string (shell mode), and ``shell`` is the declared
    interpreter name (e.g. ``"pwsh"``) in shell mode or ``None`` in argv mode.
    The shell name is NOT resolved to a path — preview is a diagnostic tool
    and deliberately does NOT perform Gating, so you can preview any
    platform's branch regardless of whether the executable is on PATH.

    ``_platform`` overrides the detected platform (``"windows"`` forces the
    windows branch) so any branch can be previewed from any host.
    Underscore-prefixed so it cannot collide with a tool's own arg names.

    Validation is shared with ``load_yaml_tools`` via ``_compile`` — the
    two paths cannot drift. Raises ``ValueError`` on malformed YAML.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        block = _compile(yaml_text, source=source, platform=_platform)
    return block.render(**kwargs), block.shell


# Known top-level keys on a tool declaration. Unknown keys are rejected at
# parse time (``extra="forbid"`` discipline) so a typo like ``shel:`` or a
# renamed field surfaces immediately, not as a silently-ignored directive.
_TOOL_KEYS = {"name", "description", "command", "shell", "args", "platforms"}
_ARG_KEYS = {"name", "type", "description", "required", "to"}
_PLATFORM_KEYS = {"linux", "macos", "unix", "windows"}

# The single source of truth for the YAML ``type:`` shorthand → Python type.
# ``_build_signature`` reads the Python type; ``_build_tool_schema`` reads the
# JSON-Schema name (via ``_JSON_TYPE``). Both are derived from this map so a
# new type lands in both places at once.
_ARG_TYPES: dict[str, type] = {"int": int, "str": str, "bool": bool, "list": list}
_JSON_TYPE = {
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "list": "array",
}


def _validate_arg_type(arg: dict[str, Any], tool_name: str, source: str | None) -> None:
    """Reject an unknown ``type:`` value, naming the tool, arg, and allowed set.

    Previously an unknown ``type:`` (e.g. ``float``) silently fell back to
    ``"string"`` in the emitted schema, polluting the contract the LLM sees
    with no signal to the author. Now it fails at load time listing the
    legal types. ``type:`` absent defaults to ``str`` (the common case) and
    is not an error.
    """
    if "type" not in arg:
        return
    declared = arg["type"]
    if declared not in _ARG_TYPES:
        where = f" in {source}" if source else ""
        msg = (
            f"tool {tool_name!r}: arg {arg['name']!r} has unknown type "
            f"{declared!r}{where}; allowed: {sorted(_ARG_TYPES)!r}"
        )
        raise ValueError(msg)


def _stringify(value: Any) -> str:
    """Coerce a YAML scalar to its string form (for ``name`` / ``description``).

    ``None`` (field absent) returns an empty string so callers can use
    ``_stringify(x) or fallback`` — without this, ``str(None) == "None"``
    would leak into ``__doc__`` / ``__name__``.
    """
    if value is None:
        return ""
    return str(value)


# Characters allowed in a tool name. OpenAI / Anthropic function-name rules
# converge on ``[A-Za-z0-9_-]`` plus ``.`` for cothis's namespace convention.
# Anything else (spaces, ``/``, ``:``, …) would break provider routing or
# cothis's dotted-namespace / MCP-prefix scheme. We don't reject on violation —
# we normalise: spaces / ``/`` / ``:`` → ``-`` (readable), other special
# characters stripped. Author-facing names like ``uv add`` keep working as
# ``uv-add``. The original is logged at WARNING so the author sees the rename.
_NAME_REPLACEMENTS = {" ": "-", "/": "-", ":": "-"}


def _normalise_tool_name(raw: str, source: str | None) -> str:
    """Normalise a YAML ``name:`` into a tool name safe for provider routing.

    Spaces, ``/`` and ``:`` → ``-`` (``"uv add"`` → ``"uv-add"``, ``"a/b"``
    → ``"a-b"``). Other characters outside ``[A-Za-z0-9_.-]`` are stripped
    (``"a!b"`` → ``"ab"``). The rename is logged at WARNING when it changes
    the input, naming the source file so the author can fix it deliberately.
    Empty after normalisation → ``ValueError`` (a tool with no callable name
    is unusable).
    """
    normalised = ""
    changed = False
    for ch in raw:
        if (ch.isascii() and ch.isalnum()) or ch in "_.-":
            normalised += ch
        elif ch in _NAME_REPLACEMENTS:
            normalised += _NAME_REPLACEMENTS[ch]
            changed = True
        else:
            changed = True
    if not normalised:
        where = f" in {source}" if source else ""
        msg = f"tool name {raw!r} normalises to empty{where}; set a non-empty 'name:'"
        raise ValueError(msg)
    if changed:
        where = f" in {source}" if source else ""
        logger.warning("tool name %r normalised to %r%s", raw, normalised, where)
    return normalised


def _shell_quote(value: str, shell: str | None) -> str:
    """Shell-quote ``value`` for the named interpreter.

    POSIX shells (``sh``, ``bash``, ``zsh``, ``pwsh``, …) get ``shlex.quote``
    (single-quote wrapping). PowerShell treats single quotes as quoting too,
    so the POSIX branch is correct for ``pwsh``.

    Windows ``cmd.exe`` gets ``subprocess.list2cmdline`` — single quotes are
    NOT cmd quoting, so the POSIX branch would silently fail there.

    cothis: ceiling — ``list2cmdline`` does NOT fully close cmd.exe
    injection. Per CPython source (``needquote = (" " in arg) or
    ("\t" in arg) or not arg``) it only double-quotes when the value
    contains whitespace, tab, or is empty. (A literal double-quote in the
    value triggers internal ``\"`` escaping, not wrapping — it round-trips
    correctly but is a different code path.) A value like ``foo&echo
    PWNED`` (no spaces) passes through UNQUOTED — ``&`` is a live cmd
    metacharacter, so story 22 is only partially met on cmd.exe. ``%VAR%``
    expansion is also undefended: cmd.exe expands it inside double quotes
    too. Upgrade path: (a) hand-roll a full cmd.exe escaper that quotes
    every value and escapes ``%`` (notoriously fragile), (b) restrict
    shell-mode-on-Windows to PowerShell (``pwsh``) and require argv mode
    (``command: [list]``) for untrusted input under ``cmd``, or (c) accept
    the ceiling and document it per-tool. Today the partial defence is
    retained (whitespace-bearing values ARE safe) as better than the prior
    POSIX-only path which was wrong for cmd on every value.

    ``shell`` is the declared interpreter name (``self.shell``); ``None`` means
    argv mode where quoting never runs (the caller passes ``None``).
    """
    if shell == "cmd":
        # ``list2cmdline`` expects a token list and returns the cmd.exe-safe
        # rendering of all of them joined by spaces. Pass one token so the
        # output is exactly that token's cmd-safe form, no leading space.
        # See the docstring ceiling: this is partial, not complete, defence.
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


def _value_mapping(
    arg_specs: list[dict[str, Any]],
    values: dict[str, Any],
    *,
    shell: str | None = None,
) -> dict[str, Any]:
    """Build the ``format_map`` input from declared arg specs and runtime values.

    - Bool args carrying a ``to:`` flag render as the flag string (``--dev``)
      when true, empty string when false (story 12). Non-bool values ignore ``to:``.
    - In shell mode (``shell`` is the interpreter name), string values are
      shell-quoted for the SPECIFIC interpreter (POSIX → ``shlex.quote``;
      ``cmd`` → ``subprocess.list2cmdline``) and list elements are quoted
      individually then space-joined (story 22). Argv mode (``shell=None``) is
      inherently safe — ``subprocess.run(list)`` does its own tokenisation — so
      quoting only applies to shell mode.
    - Other values pass through as-is so Python format specs can apply
      (``{n:03d}`` needs an int, not a str).
    """
    mapping: dict[str, Any] = {}
    for spec in arg_specs:
        name = spec["name"]
        if name not in values:
            continue
        value = values[name]
        flag = spec.get("to")
        if flag is not None and isinstance(value, bool):
            mapping[name] = flag if value else ""
            continue
        if isinstance(value, list):
            if shell:
                mapping[name] = " ".join(_shell_quote(str(v), shell) for v in value)
            else:
                mapping[name] = " ".join(str(v) for v in value)
        elif shell and isinstance(value, str):
            mapping[name] = _shell_quote(value, shell)
        else:
            mapping[name] = value
    return mapping


def _build_signature(arg_specs: list[dict[str, Any]]) -> inspect.Signature:
    """Build an ``inspect.Signature`` from YAML arg declarations.

    Each declared arg becomes a ``Parameter`` of the declared ``type``
    (defaulting to ``str``), so any-llm's ``inspect.signature``-based
    schema builder picks them up. All args are keyword-or-positional with
    no default; optional args (``required: false``) are a later slice.
    """
    params = [
        inspect.Parameter(
            a["name"],
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=_ARG_TYPES.get(a.get("type", "str"), str),
        )
        for a in arg_specs
    ]
    return inspect.Signature(params)


def _build_tool_schema(
    name: str, description: str, arg_specs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build an Anthropic-format tool schema dict carrying per-arg descriptions.

    any-llm's ``callable_to_tool`` drops per-parameter ``description``
    fields (it only reads type annotations), so YAML tools pre-build the
    full schema here in Anthropic shape (``{name, description, input_schema}``)
    and Agent passes it straight through to ``any_llm.amessages`` via any-llm's
    dict-passthrough (``prepare_tools`` leaves dicts alone). This is how the
    rich ``description:`` text a YAML author writes actually reaches the model.
    """
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for arg in arg_specs:
        prop: dict[str, Any] = {
            "type": _JSON_TYPE.get(arg.get("type", "str"), "string")
        }
        if arg.get("description"):
            prop["description"] = arg["description"]
        properties[arg["name"]] = prop
        if arg.get("required", True):
            required.append(arg["name"])
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
