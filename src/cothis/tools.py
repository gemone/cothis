"""Built-in tools exposed to the cothis agent."""

from __future__ import annotations

import inspect
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class Tool(Protocol):
    """A callable the Agent registers by ``__name__`` and dispatches by call.

    Defining ``__name__`` on the protocol (rather than using
    ``Callable[..., Any]``) lets the type checker see attribute access on
    tools without ``# type: ignore`` — every real tool (a ``def``, the
    ``_ShellTool`` class, …) structurally satisfies this.

    ``@runtime_checkable`` lets pydantic validate ``list[Tool]`` via
    ``isinstance`` (Agent's ``model_config`` has ``arbitrary_types_allowed``).
    """

    __name__: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def _named(name: str) -> Callable[[Tool], Tool]:
    """Override a callable's tool name.

    ``any-llm`` derives the tool name from ``func.__name__``, which by
    default cannot contain a dot (Python identifiers). This decorator
    rewrites ``__name__`` so we can register namespaced tools such as
    ``fs.read`` and ``fs.write``.
    """

    def decorator(func: Tool) -> Tool:
        func.__name__ = name
        return func

    return decorator


@_named("fs.read")
def read(path: str) -> str:
    """Read the contents of a UTF-8 text file from the filesystem.

    Use this to inspect an existing file before reading or modifying it.

    Args:
        path: Path to the file to read. Relative paths are resolved
            against the current working directory.
            eg. "src/main.py", "./README.md", "/etc/hostname".

    Returns:
        The file contents decoded as UTF-8.
    """
    return Path(path).read_text(encoding="utf-8")


@_named("fs.write")
def write(path: str, content: str) -> str:
    """Write text to a file on the filesystem, creating it if needed.

    Parent directories are created automatically. Existing files are
    overwritten.

    Args:
        path: Path to the file to write.
            eg. "notes.txt", "src/generated.py", "./output/result.json".
        content: The text to write to the file.
            eg. "hello world", a full source file's text.

    Returns:
        A short confirmation with the number of characters written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


TOOLS: list[Tool] = [read, write]


# ====================================================================
# YAML tool loading
#
# A YAML tool declaration describes a shell command (or per-platform
# variants) that becomes a callable the Agent dispatches. The declaration
# never reaches the LLM verbatim — what the model sees is the compiled
# OpenAI-format schema (with per-arg descriptions preserved, bypassing
# any-llm's lossy ``callable_to_tool``).
#
# Compile pipeline: ``_compile`` is the single validation pass — both
# ``load_yaml_tools`` (gate + wrap) and ``preview`` (render) call it, so
# the two paths cannot drift on what a valid YAMLTool is. It returns a
# ``CommandBlock`` carrying the selected command, shell, and filtered arg
# specs. ``CommandBlock.render`` owns the list-vs-string Placeholder fork
# and the ``str.format_map`` substitution; ``_ShellTool.__call__`` owns the
# subprocess dispatch fork (``shell=False`` vs ``shell=True``).
#
# Execution model (type-driven, see ADR-0001):
# - ``command:`` as a YAML list  → argv mode (``shell=False``), argv[0] is
#   the executable and is gated (``shutil.which`` must find it or the tool
#   is not registered).
# - ``command:`` as a YAML string → shell mode (``shell=True``); a
#   ``shell:`` field naming the interpreter is REQUIRED, and that shell is
#   gated the same way. A string without ``shell:`` is a load-time error
#   (forces the author to declare which shell interprets the string).
#
# Placeholder syntax: full Python format-string semantics.
# ``command:`` is rendered via ``str.format_map``; ``{name}`` substitutes
# a named arg, ``{{`` escapes to a literal ``{``, ``{name:spec}`` / ``{name!r}``
# are supported (free, from Python). Shell variables must be escaped as
# ``${{HOME}}`` → ``${HOME}``. Undeclared placeholders raise ``ValueError``
# at compile time (loud, not silent residue reaching the shell).
#
# Per-platform variants live under ``platforms:``, a map keyed by
# ``linux`` / ``macos`` / ``unix`` (= linux+macOS) / ``windows``. The
# top-level ``command:`` / ``shell:`` / ``args:`` are the default; a
# matching platform entry overrides them (args merge by name: branch
# overrides same-named, inherits the rest).
# ====================================================================


def load_yaml_tools(yaml_text: str, *, source: str | None = None) -> list[Tool]:
    """Compile a YAML tool declaration into a callable.

    Thin wrapper over ``_compile``: compile, gate the executable via
    ``shutil.which``, wrap in ``_ShellTool``. If the executable is not on
    PATH the tool is **silently not registered** — the model never sees a
    tool it cannot dispatch on this host. cothis ceiling: the skip is silent
    (no debug log yet); upgrade path: emit a debug log when a logging
    surface exists.

    See ``_compile`` for the YAML shape and ``CommandBlock`` for the
    contract. ``preview`` shares the same compile path, so the two cannot
    drift on what a valid YAMLTool is.
    """
    block = _compile(yaml_text, source=source)
    exe = _resolve_executable(block.gate_target)
    if exe is None:
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
        return self.shell  # type: ignore[return-value]  # locked non-None by _compile

    def render(self, **kwargs: Any) -> list[str] | str:
        """Substitute Placeholders into the selected command.

        argv mode → ``[rendered element per argv item]`` (list preserved).
        shell mode → rendered template string. Uses ``str.format_map``
        semantics (``{name}`` substitutes, ``{{`` escapes a literal ``{``,
        ``{n:03d}`` / ``{p!r}`` honoured). A referenced arg missing from
        ``kwargs`` raises ``KeyError`` — a caller-side contract violation,
        not a compile-time bug.

        Pure: no subprocess, no filesystem, no gating. The list-vs-string
        render fork lives here and only here.
        """
        mapping = _value_mapping(self._arg_names, kwargs)
        if isinstance(self.command, list):
            return [part.format_map(mapping) for part in self.command]
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
    ``preview`` can compile any branch regardless of host executables. The
    The gated name is exposed via ``CommandBlock.gate_target``.

    Raises ``ValueError`` on any contract violation (message names the tool,
    the field, and ``source``). Emits ``UserWarning`` for declared-but-
    unreferenced args (``preview`` suppresses; ``load_yaml_tools`` surfaces).
    """
    import warnings

    spec = yaml.safe_load(yaml_text)
    _check_unknown_keys(spec, _TOOL_KEYS, source, what="YAML tool")
    name = str(_require(spec, "name", source))
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

    selected = _select_platform(top, platforms, platform=platform)

    # Keep only args the selected command actually references; every
    # placeholder must resolve to a declared arg (typos surface here, loud).
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

    # Execution-mode validation (ADR-0001). The command's YAML type IS the
    # mode selector; these checks pin the type↔shell pairing. Gating runs
    # later in load_yaml_tools (preview skips it) — these are pure shape rules.
    if isinstance(selected.command, list):
        # argv mode — argv[0] is the executable. It must be a literal:
        # gating runs at load time, before args are available, so a
        # placeholder in argv[0] can never be resolved via shutil.which.
        if _extract_field_names(selected.command[0]):
            where = f" in {source}" if source else ""
            msg = (
                f"tool {name!r}: argv[0] (the executable) must not contain "
                f"placeholders{where}; move the executable name out of {{...}}"
            )
            raise ValueError(msg)
        # ``shell:`` is meaningless in argv mode (no shell interprets the
        # command); reject it so the author hears about the misconception.
        if selected.shell is not None:
            where = f" in {source}" if source else ""
            msg = (
                f"tool {name!r}: ``shell:`` is meaningless with a list "
                f"``command:`` (argv mode){where}; use a string ``command:`` "
                "if you need shell interpretation"
            )
            raise ValueError(msg)
    else:
        # shell mode — shell: is required and is the gated executable.
        if not selected.shell:
            where = f" in {source}" if source else ""
            msg = (
                f"tool {name!r}: string ``command:`` requires a ``shell:`` field"
                f"{where} (to declare which interpreter runs it)"
            )
            raise ValueError(msg)

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
        # A platform entry may omit command: — inherit the top-level value.
        # Sentinel: empty string; the caller resolves inheritance.
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

    # When command is absent (a platform entry inheriting the default),
    # use an empty string as a sentinel; the caller resolves inheritance.
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


def _resolve_executable(name_or_path: str) -> str | None:
    """Resolve an executable name to a path via ``shutil.which``.

    Returns the resolved path, or ``None`` if not found. ``None`` means
    "not on PATH" → caller skips registration (silent gate). Used for both
    argv[0] (argv mode) and the ``shell:`` interpreter (shell mode).
    """
    return shutil.which(name_or_path)


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
            continue  # trailing literal segment, no field
        # ``field_name`` may be ``name``, ``name.attr``, ``name[idx]``, or
        # positional ``0``. Keep the leading identifier only.
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


class _ShellTool:
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
    """

    __name__: str
    __doc__: str
    __signature__: inspect.Signature
    __cothis_schema__: dict[str, Any] | None

    def __init__(self, block: CommandBlock, *, shell_path: str | None) -> None:
        self.__name__ = block.name
        self.__doc__ = block.description
        self.__signature__ = _build_signature(block.arg_specs)
        self.__cothis_schema__ = _build_tool_schema(
            block.name, block.description, block.arg_specs
        )
        self._block = block
        # Resolved interpreter path (shell mode) or None (argv mode, where
        # subprocess re-resolves argv[0] itself).
        self._shell_path = shell_path

    def __call__(self, **kwargs: Any) -> str:
        rendered = self._block.render(**kwargs)
        if isinstance(rendered, list):
            proc = subprocess.run(rendered, shell=False, capture_output=True, text=True)
        else:
            proc = subprocess.run(
                rendered,
                shell=True,
                capture_output=True,
                text=True,
                executable=self._shell_path,
            )
        # Surface non-zero exits as an error string the model can act on,
        # not an exception that crashes the ReAct loop.
        if proc.returncode != 0:
            return f"Error: exit code {proc.returncode}: {proc.stderr}"
        return proc.stdout


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
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    block = _compile(yaml_text, source=source, platform=_platform)
    return block.render(**kwargs), block.shell


def _require(
    spec: Any, key: str, source: str | None, *, what: str = "YAML tool"
) -> Any:
    """Return ``spec[key]`` or raise a ``ValueError`` naming the field + source.

    A missing required field must fail with a clear, actionable error (file
    + field), not a bare ``KeyError`` / ``TypeError``. Handles the empty-file
    case (``safe_load`` returns ``None``) too. ``source`` is the file path
    when known (from ``load_tools_from_dir``) so the error points at the
    offending file; ``None`` for direct callers (``preview``).
    """
    where = f" in {source}" if source else ""
    if not isinstance(spec, dict):
        msg = f"{what} must be a YAML mapping{where}; got {type(spec).__name__}"
        raise ValueError(msg)
    if key not in spec:
        msg = f"{what} must define {key!r}{where}"
        raise ValueError(msg)
    return spec[key]


# Known top-level keys on a tool declaration. Unknown keys are rejected at
# parse time (``extra="forbid"`` discipline) so a typo like ``shel:`` or a
# renamed field surfaces immediately, not as a silently-ignored directive.
_TOOL_KEYS = {"name", "description", "command", "shell", "args", "platforms"}
# Known keys on a single ``args:`` entry.
_ARG_KEYS = {"name", "type", "description", "required"}
# Known keys under ``platforms:`` (platform names).
_PLATFORM_KEYS = {"linux", "macos", "unix", "windows"}


def _check_unknown_keys(
    spec: Any, allowed: set[str], source: str | None, *, what: str
) -> None:
    """Raise if ``spec`` has keys outside ``allowed`` (``extra="forbid"``).

    Mirrors pydantic's ``extra='forbid'`` model config: a YAML tool is a
    configuration contract, and a key the loader doesn't recognise is
    either a typo or a field from a newer/older format — either way the
    author wants to hear about it, not have it silently dropped. The error
    names the offending keys, the scope (``what``), and the file (``source``).
    """
    if not isinstance(spec, dict):
        return  # shape errors are surfaced by ``_require`` / ``_parse_command_block``
    extra = set(spec) - allowed
    if extra:
        where = f" in {source}" if source else ""
        msg = (
            f"{what}: unknown field(s) {sorted(extra)!r}{where}; "
            f"allowed: {sorted(allowed)!r}"
        )
        raise ValueError(msg)


# The single source of truth for the YAML ``type:`` shorthand → Python type.
# ``_build_signature`` reads the Python type; ``_build_tool_schema`` reads the
# JSON-Schema name (via ``_JSON_TYPE``). Both are derived from this map so a
# new type lands in both places at once.
_ARG_TYPES: dict[str, type] = {"int": int, "str": str, "bool": bool, "list": list}
# JSON-Schema type names, derived from ``_ARG_TYPES``.
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


def _value_mapping(arg_names: list[str], values: dict[str, Any]) -> dict[str, Any]:
    """Build the ``format_map`` input: declared names → values.

    Values are passed through as-is (not pre-stringified) so Python format
    specs can apply (``{n:03d}`` needs an int, not a str). Lists are an
    exception — pre-joined with spaces so ``{ids}`` renders as ``"1 2 3"``
    rather than ``"[1, 2, 3]"``.
    """
    mapping: dict[str, Any] = {}
    for name in arg_names:
        if name not in values:
            continue
        value = values[name]
        if isinstance(value, list):
            mapping[name] = " ".join(str(v) for v in value)
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
    """Build an OpenAI-format tool schema dict carrying per-arg descriptions.

    any-llm's ``callable_to_tool`` drops per-parameter ``description``
    fields (it only reads type annotations), so YAML tools pre-build the
    full schema here and Agent passes it straight through to the provider
    via any-llm's dict-passthrough (``prepare_tools`` leaves dicts alone).
    This is how the rich ``description:`` text a YAML author writes
    actually reaches the model.
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
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _schema_for(tool: Tool) -> Tool | dict[str, Any]:
    """Return ``tool`` in the form any-llm's ``acompletion`` expects.

    YAML tools carry a pre-built OpenAI schema on ``__cothis_schema__`` (so
    per-arg ``description:`` text reaches the model — any-llm's
    ``callable_to_tool`` would strip it). Tools without the attribute fall
    through as callables and any-llm converts them.

    Keeping this fork here (next to ``_build_tool_schema``, the producer of
    the attribute) means ``Agent`` stays blind to the ``__cothis_schema__``
    name — the schema serialisation rule lives in ``tools.py``, where the
    Tools are defined, not in ``agent.py``.
    """
    return getattr(tool, "__cothis_schema__", tool)


def load_tools_from_dir(dir_path: Path) -> list[Tool]:
    """Load every YAML tool declaration in a directory tree.

    Globs ``**/*.yaml`` and ``**/*.yml`` recursively (sorted by path, so
    load order is stable across platforms) and compiles each via
    ``load_yaml_tools``. Non-YAML files are ignored. An empty / missing
    directory yields ``[]``.

    cothis: tool ``name`` still comes from each file's ``name:`` field,
    not from the filename or path. A file at ``date/current.yaml`` is
    named whatever its ``name:`` says (typically ``date.current``); the
    directory layout is pure organisation for the human author.
    """
    if not dir_path.is_dir():
        return []
    files = sorted(
        {*dir_path.rglob("*.yaml"), *dir_path.rglob("*.yml")},
        key=lambda p: str(p.relative_to(dir_path)),
    )
    tools: list[Tool] = []
    for yml in files:
        tools.extend(load_yaml_tools(yml.read_text(encoding="utf-8"), source=str(yml)))
    return tools
