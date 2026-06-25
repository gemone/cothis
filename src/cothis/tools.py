"""Built-in tools exposed to the cothis agent."""

from __future__ import annotations

import inspect
import shutil
import string
import subprocess
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


def load_yaml_tools(yaml_text: str, *, source: str | None = None) -> list[Tool]:
    """Compile shell-template tool declarations from YAML into callables.

    ``command:`` is the only command-bearing field. It has two shapes:

    1. Flat (single command, no per-platform branching):

           name: hello
           description: Say hello.
           command: echo hello
           args: [...]

    2. Conditional list (per-platform branching). Each entry carries an
       optional ``if:`` GitHub-Actions expression over ``runner.os`` /
       ``runner.arch``, an optional ``shell:`` (default ``sh`` on POSIX,
       ``cmd`` on Windows), and a ``run:`` string. The loader picks the
       first branch whose ``if:`` is true under the current platform:

           name: date.current
           description: Get current date and time.
           command:
             - if: runner.os == 'Linux' || runner.os == 'macOS'
               shell: bash
               run: date
             - if: runner.os == 'Windows'
               shell: pwsh
               run: Get-Date
           args: [...]

    Both shapes accept the same ``args:`` list; arg values are substituted
    into whichever ``run`` (or flat ``command``) is selected at load time.

    Rendering rules:
    - ``str`` / ``int``: ``str(value)`` is substituted at ``{arg_name}``.
    - ``list``: elements are ``str``-ed and joined by spaces.
    - ``bool``: substituted as ``str(value)`` (``to:`` flag injection later).

    ``if:`` expression language (GA-faithful subset, evaluated at load
    time against a ``runner`` context built from ``sys.platform``):
    - ``runner.os`` ∈ {'Linux', 'macOS', 'Windows'}
    - ``runner.arch`` ∈ {'X64', 'ARM64', 'X86'}
    - ``==`` / ``!=`` (case-insensitive string compare, GA semantics)
    - ``&&`` / ``||`` (logical and/or)
    - single-quoted string literals (``'Linux'``)
    - parentheses for grouping
    - ``has_shell(name)`` / ``has_exe(name)``: true iff ``shutil.which(name)``
      finds the binary on PATH (e.g. ``has_shell('pwsh')`` gates a
      PowerShell branch on pwsh actually being installed)

    The matching branch is selected at **load time** (not dispatch
    time): the loader evaluates the ``if:`` predicates against the
    current platform once and closes over the winning ``(shell, run)``
    pair. An unsupported platform fails at startup, not mid-conversation.

    A ``default: true`` branch (at most one, no accompanying ``if:``) is the
    explicit fallback selected when no conditional branch matches.

    Cothis divergence from GA: if no branch's ``if:`` matches and there is
    no ``default`` branch, the tool is **silently not registered** — it
    does not appear in the agent's tool list, so the model never sees a
    tool it cannot dispatch on this platform. GA's hard error makes sense
    for a CI job graph (the gap must surface); a runtime toolset should
    just omit unavailable tools. cothis ceiling: the skip is silent today
    (no debug log yet); upgrade path: emit a debug log naming the skipped
    tool + the platform once a logging surface exists (story 43).

    cothis: substitution pre-rewrites ``{`` → ``${`` so ``string.Template``
    can consume it, then ``safe_substitute`` fills declared args and leaves
    unknown placeholders as ``${name}`` (NOT the original ``{name}``).
    Ceiling: a command containing a literal ``{`` (e.g. bash ``${VAR}`` or
    a brace expansion) gets mangled — ``{VAR}`` → ``${VAR}`` and may then
    collide with an arg name. Upgrade path: distinguish placeholders from
    literal braces (e.g. an explicit ``{{ }}`` escape, or argv mode that
    skips the template engine entirely). No shell-escaping yet either —
    a YAML tool is trusted the same way a hand-written Python tool is,
    so an arg value can inject shell metacharacters. Upgrade path for
    that: per-arg ``shlex.quote`` once untrusted YAML needs to be supported.
    """
    spec = yaml.safe_load(yaml_text)
    _check_unknown_keys(spec, _TOOL_KEYS, source, what="YAML tool")
    name = str(_require(spec, "name", source))
    arg_specs = spec.get("args") or []
    arg_names: list[str] = []
    for a in arg_specs:
        _check_unknown_keys(a, _ARG_KEYS, source, what=f"tool {name!r}: arg")
        arg_name = str(_require(a, "name", source, what=f"tool {name!r}: arg"))
        arg_names.append(arg_name)
        a["name"] = arg_name  # normalise so downstream sees the string form
        _validate_arg_type(a, name, source)
    command_field = _require(spec, "command", source)
    steps = _steps_from_command(command_field, name, source)
    selected = _select_step(steps)
    if selected is None:
        # No branch matched and no default — silently skip registration.
        # The model never sees a tool it can't dispatch on this platform.
        return []
    selected_shell, selected_command = selected
    raw_desc = spec.get("description")
    description = (
        str(raw_desc)
        if raw_desc is not None and raw_desc != ""
        else f"Shell tool: {selected_command}"
    )

    return [
        _ShellTool(
            name=name,
            description=description,
            command=selected_command,
            shell=selected_shell,
            arg_names=arg_names,
            arg_specs=arg_specs,
        )
    ]


class _ShellTool:
    """A callable shell tool compiled from YAML.

    Wraps the selected ``(shell, command)`` pair and the arg specs into an
    object that looks like a function to any-llm (``__name__``, ``__doc__``,
    ``__signature__``, ``__call__``) while also carrying a pre-built schema
    (``__cothis_schema__``) with per-arg descriptions.

    Using a class instead of monkey-patching attributes onto a closure
    function avoids type-checker errors: ``FunctionType`` doesn't declare
    ``__signature__`` or ``__cothis_schema__``, so direct assignment fails
    strict type checking. A class declares them as real attributes.
    """

    __name__: str
    __doc__: str
    __signature__: inspect.Signature
    # OpenAI-format tool schema dict with per-arg descriptions, or None if
    # not applicable (built-in tools don't set this).
    __cothis_schema__: dict[str, Any] | None

    def __init__(
        self,
        *,
        name: str,
        description: str,
        command: str,
        shell: str | None,
        arg_names: list[str],
        arg_specs: list[dict[str, Any]],
    ) -> None:
        self.__name__ = name
        self.__doc__ = description
        self.__signature__ = _build_signature(arg_specs)
        self.__cothis_schema__ = _build_tool_schema(name, description, arg_specs)
        self._command = command
        self._shell = shell
        self._arg_names = arg_names

    def __call__(self, **kwargs: Any) -> str:
        rendered = _render_command(self._command, self._arg_names, kwargs)
        proc = subprocess.run(
            rendered,
            shell=True,
            capture_output=True,
            text=True,
            executable=self._shell,
        )
        # Stories 18/19: surface non-zero exits as an error string the model
        # can act on, not an exception that crashes the ReAct loop. stdout is
        # returned only on success; on failure stderr (with the exit code) is
        # the actionable signal.
        if proc.returncode != 0:
            return f"Error: exit code {proc.returncode}: {proc.stderr}"
        return proc.stdout


def preview(
    yaml_text: str,
    *,
    _os: str | None = None,
    _arch: str | None = None,
    source: str | None = None,
    **kwargs: Any,
) -> tuple[str | None, str]:
    """Return the ``(shell, rendered_command)`` a tool would run, without dispatching.

    Parses the YAML, selects the platform-appropriate ``steps:`` branch,
    and substitutes ``kwargs`` into the command template — exactly what
    ``load_yaml_tools(...)[0](**kwargs)`` would hand to ``subprocess.run``,
    minus the subprocess. This is the verification surface for tests
    that need to assert shell content without spawning processes (e.g.
    asserting the Windows/pwsh branch on a Linux CI host).

    ``_os`` / ``_arch`` override the detected platform so any branch can
    be previewed from any host (``_os="Windows"`` forces the Windows
    branch regardless of ``sys.platform``). Underscore-prefixed so they
    cannot collide with a tool's own arg names.

    Raises the same ``ValueError`` as the loader when no platform branch
    matches.
    """
    spec = yaml.safe_load(yaml_text)
    _check_unknown_keys(spec, _TOOL_KEYS, source, what="YAML tool")
    name = str(_require(spec, "name", source))
    arg_names: list[str] = []
    for a in spec.get("args") or []:
        _check_unknown_keys(a, _ARG_KEYS, source, what=f"tool {name!r}: arg")
        arg_name = str(_require(a, "name", source, what=f"tool {name!r}: arg"))
        arg_names.append(arg_name)
        a["name"] = arg_name
        _validate_arg_type(a, name, source)
    command_field = _require(spec, "command", source)
    steps = _steps_from_command(command_field, name, source)
    selected = _select_step(steps, _os=_os, _arch=_arch)
    if selected is None:
        context = _runner_context(_os=_os, _arch=_arch)
        msg = (
            "no matching branch for this platform and no default "
            f"(runner.os={context['os']!r}); declared predicates: "
            f"{[s.get('if') for s in steps]}"
        )
        raise ValueError(msg)
    shell, command = selected
    return shell, _render_command(command, arg_names, kwargs)


def _steps_from_command(
    command_field: Any, name: str, source: str | None
) -> list[dict[str, Any]]:
    """Normalise the ``command:`` field into a list of branch dicts.

    Shared by ``load_yaml_tools`` and ``preview`` so both follow one rule.
    A flat string is sugar for a single unconditional branch (no ``if:``);
    a list is the conditional form — each entry must carry a ``run:``
    string and optional ``if:`` / ``shell:``. An empty list, a missing
    ``run:`` on an entry, or a non-str/non-list ``command:`` are all
    load-time errors naming the tool (and file, when known).
    """
    where = f" in {source}" if source else ""
    if isinstance(command_field, str):
        return [{"run": command_field}]
    if not isinstance(command_field, list):
        msg = (
            f"YAML tool {name!r}: 'command' must be a string or a list of "
            f"branches{where}; got {type(command_field).__name__}"
        )
        raise ValueError(msg)
    if not command_field:
        msg = f"YAML tool {name!r}: 'command' list is empty{where}"
        raise ValueError(msg)
    steps: list[dict[str, Any]] = []
    seen_default = False
    for i, entry in enumerate(command_field):
        if not isinstance(entry, dict) or "run" not in entry:
            msg = (
                f"YAML tool {name!r}: 'command' branch #{i} must be a mapping "
                f"with a 'run' key{where}"
            )
            raise ValueError(msg)
        _check_unknown_keys(
            entry, _BRANCH_KEYS, source, what=f"tool {name!r}: command branch #{i}"
        )
        has_default = bool(entry.get("default"))
        has_if = "if" in entry
        if has_default and has_if:
            msg = (
                f"YAML tool {name!r}: 'command' branch #{i} cannot have both "
                f"'if' and 'default'{where}"
            )
            raise ValueError(msg)
        if has_default:
            if seen_default:
                msg = (
                    f"YAML tool {name!r}: multiple 'default' branches "
                    f"(second at branch #{i}){where}"
                )
                raise ValueError(msg)
            seen_default = True
        steps.append(cast("dict[str, Any]", entry))
    return steps


def _require(
    spec: Any, key: str, source: str | None, *, what: str = "YAML tool"
) -> Any:
    """Return ``spec[key]`` or raise a ``ValueError`` naming the field + source.

    Story 24: a missing required field must fail with a clear, actionable
    error (file + field), not a bare ``KeyError`` / ``TypeError``. Handles
    the empty-file case (``safe_load`` returns ``None``) too. ``source`` is
    the file path when known (from ``load_tools_from_dir``) so the error
    points at the offending file; ``None`` for direct callers (``preview``).
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
_TOOL_KEYS = {"name", "description", "command", "args"}
# Known keys on a single ``args:`` entry.
_ARG_KEYS = {"name", "type", "description", "required"}
# Known keys on a ``command:`` list branch.
_BRANCH_KEYS = {"if", "run", "shell", "default"}


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
        return  # shape errors are surfaced by ``_require`` / ``_steps_from_command``
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


def _select_step(
    steps: list[dict[str, Any]],
    *,
    _os: str | None = None,
    _arch: str | None = None,
) -> tuple[str | None, str] | None:
    """Pick the first branch whose ``if:`` is true; fall back to ``default``.

    Returns ``(shell, run)`` for the winning branch, or ``None`` if no branch
    matches and there is no ``default`` branch. A ``None`` return means "do
    not register this tool on this platform" — ``load_yaml_tools`` turns it
    into an empty tool list; ``preview`` raises so the caller knows the branch
    set is empty for the requested platform.

    A branch without ``if:`` (the single-command sugar form) always matches.
    ``_os``/``_arch`` override the detected platform (used by ``preview`` to
    inspect a branch the host would not normally select).
    """
    context = _runner_context(_os=_os, _arch=_arch)
    default_step: dict[str, Any] | None = None
    for step in steps:
        if step.get("default"):
            default_step = step
            continue
        predicate = step.get("if")
        if predicate is None or _eval_if(predicate, context):
            shell = _resolve_shell(step.get("shell"))
            return shell, step["run"]
    if default_step is not None:
        return _resolve_shell(default_step.get("shell")), default_step["run"]
    return None


def _runner_context(
    *,
    _os: str | None = None,
    _arch: str | None = None,
) -> dict[str, str]:
    """Map ``sys.platform``/``platform.machine`` onto GA's ``runner.os``/``arch``.

    GA values (verified): runner.os ∈ {'Linux','macOS','Windows'},
    runner.arch ∈ {'X64','ARM64','X86'}. ``_os``/``_arch`` override the
    detected values when provided (used by ``preview`` to preview branches
    the host would not normally select).
    """
    import platform
    import sys

    os_map = {"linux": "Linux", "darwin": "macOS", "win32": "Windows"}
    arch_map = {
        "x86_64": "X64",
        "amd64": "X64",
        "aarch64": "ARM64",
        "arm64": "ARM64",
        "i386": "X86",
        "i686": "X86",
    }
    return {
        "os": _os if _os is not None else os_map.get(sys.platform, sys.platform),
        "arch": _arch
        if _arch is not None
        else arch_map.get(platform.machine(), platform.machine()),
    }


def _resolve_shell(shell: str | None) -> str | None:
    """Resolve a YAML ``shell:`` value to a subprocess ``executable=`` path.

    ``None`` → ``None`` (subprocess default: sh on POSIX, cmd on Windows).
    Named shells (``bash``, ``pwsh``, ``sh``, ``cmd``) are looked up via
    ``shutil.which`` so the user's PATH decides; an unresolved name is
    returned as-is so the dispatch error names the missing binary.
    """
    if shell is None:
        return None
    return shutil.which(shell) or shell


def _eval_if(expr: str, context: dict[str, str]) -> bool:
    """Evaluate a GitHub-Actions ``if:`` expression against ``context``.

    Supports the GA subset documented on ``load_yaml_tools``: ``runner.*``
    lookups, ``==``/``!=``/``&&``/``||``, single-quoted strings, parentheses
    for grouping, and the ``has_shell(name)`` / ``has_exe(name)`` predicate
    functions (true iff ``shutil.which(name)`` finds the binary). Returns a
    Python bool. GA case-insensitive string comparison is honoured.

    Cothis ceiling: no ``contains()``, no unary ``!``, no ``${{ }}`` wrapper,
    no ``github.*`` context, no arithmetic, no multi-arg functions. These
    were trimmed per AGENTS.md ("no abstraction that wasn't requested").
    Unknown identifiers / syntax / functions raise ``ValueError`` at
    evaluation time.
    """
    return _IfEvaluator(context).eval(expr.strip())


class _IfEvaluator:
    """Minimal recursive-descent evaluator for the GA ``if:`` subset.

    Grammar (precedence low → high):

        or_expr    := and_expr ( '||' and_expr )*
        and_expr   := comparison ( '&&' comparison )*
        comparison := primary ( ('==' | '!=') primary )?
        primary    := '(' or_expr ')' | func_call | literal | ident_path
        literal    := quoted_string
        ident_path := NAME ( '.' NAME )*             # resolved against context
        func_call  := NAME '(' STRING ')'            # has_shell / has_exe
    """

    def __init__(self, context: dict[str, str]) -> None:
        self.context = context
        self.tokens: list[tuple[str, str]] = []
        self.pos = 0

    def eval(self, expr: str) -> bool:
        self.tokens = _tokenize_if(expr)
        self.pos = 0
        result = self._or_expr()
        if self.pos != len(self.tokens):
            msg = f"unexpected trailing tokens in if-expression: {expr!r}"
            raise ValueError(msg)
        return bool(result)

    def _peek_kind(self) -> str | None:
        """Kind of the next token, or ``None`` at end-of-input."""
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None

    def _advance(self) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _or_expr(self) -> bool:
        left = self._and_expr()
        while self._peek_kind() == "OR":
            self._advance()
            left = self._and_expr() or left
        return left

    def _and_expr(self) -> bool:
        left = self._comparison()
        while self._peek_kind() == "AND":
            self._advance()
            left = self._comparison() and left
        return left

    def _comparison(self) -> bool:
        left = self._primary()
        kind = self._peek_kind()
        if kind in ("EQ", "NEQ"):
            op = self._advance()[0]
            right = self._primary()
            equal = _ga_equal(left, right)
            return equal if op == "EQ" else not equal
        return left

    def _primary(self) -> Any:
        tok = self._peek()
        if tok is None:
            msg = "unexpected end of if-expression"
            raise ValueError(msg)
        kind, value = tok
        if kind == "LPAREN":
            self._advance()
            inner = self._or_expr()
            if self._peek_kind() != "RPAREN":
                msg = "missing ')' in if-expression"
                raise ValueError(msg)
            self._advance()
            return inner
        if kind == "STRING":
            self._advance()
            return value
        if kind == "IDENT":
            self._advance()
            if self._peek_kind() == "LPAREN":
                return self._func_call(value)
            return self._resolve_ident(value)
        msg = f"unexpected token {kind!r} ({value!r}) in if-expression"
        raise ValueError(msg)

    def _peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _resolve_ident(self, path: str) -> Any:
        # ``runner.os`` → context['os']; ``runner.arch`` → context['arch'].
        parts = path.split(".")
        if len(parts) != 2 or parts[0] != "runner":
            msg = (
                f"unsupported identifier {path!r}; only 'runner.os' and "
                "'runner.arch' are available"
            )
            raise ValueError(msg)
        return self.context[parts[1]]

    def _func_call(self, name: str) -> bool:
        """Parse a ``name('arg')`` call; ``LPAREN`` is the pending token."""
        self._advance()  # consume LPAREN
        arg_tok = self._peek()
        if arg_tok is None or arg_tok[0] != "STRING":
            msg = (
                f"function {name!r} expects exactly one string argument "
                "in if-expression"
            )
            raise ValueError(msg)
        self._advance()
        if self._peek_kind() != "RPAREN":
            msg = f"missing ')' after argument to function {name!r}"
            raise ValueError(msg)
        self._advance()
        return _call_if_func(name, arg_tok[1])


def _ga_equal(left: Any, right: Any) -> bool:
    """GA-style case-insensitive string equality."""
    return str(left).lower() == str(right).lower()


# Functions callable from ``if:`` expressions. Each takes one string arg
# and returns a bool. ``has_shell`` and ``has_exe`` are the same check
# (``shutil.which``) under two names — the intent differs: ``has_shell``
# frames the branch as "run under this shell", ``has_exe`` frames it as
# "this executable is available on PATH".
_IF_FUNCS: dict[str, Any] = {
    "has_shell": lambda name: shutil.which(name) is not None,
    "has_exe": lambda name: shutil.which(name) is not None,
}


def _call_if_func(name: str, arg: str) -> bool:
    """Dispatch an ``if:``-expression function call."""
    fn = _IF_FUNCS.get(name)
    if fn is None:
        msg = (
            f"unsupported function {name!r} in if-expression; only "
            f"{sorted(_IF_FUNCS)} are available"
        )
        raise ValueError(msg)
    return bool(fn(arg))


def _tokenize_if(expr: str) -> list[tuple[str, str]]:
    """Tokenize the GA ``if:`` subset.

    Returns ``[(kind, value), ...]``. Kinds: ``STRING``, ``IDENT``,
    ``EQ``, ``NEQ``, ``AND``, ``OR``, ``LPAREN``, ``RPAREN``.
    """
    tokens: list[tuple[str, str]] = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c == "'":
            # single-quoted string; GA escapes '' → '
            end = i + 1
            buf: list[str] = []
            while end < len(expr):
                if expr[end] == "'":
                    if end + 1 < len(expr) and expr[end + 1] == "'":
                        buf.append("'")
                        end += 2
                        continue
                    break
                buf.append(expr[end])
                end += 1
            if end >= len(expr):
                msg = f"unterminated string in if-expression: {expr!r}"
                raise ValueError(msg)
            tokens.append(("STRING", "".join(buf)))
            i = end + 1
            continue
        if expr[i : i + 2] == "==":
            tokens.append(("EQ", "=="))
            i += 2
            continue
        if expr[i : i + 2] == "!=":
            tokens.append(("NEQ", "!="))
            i += 2
            continue
        if expr[i : i + 2] == "&&":
            tokens.append(("AND", "&&"))
            i += 2
            continue
        if expr[i : i + 2] == "||":
            tokens.append(("OR", "||"))
            i += 2
            continue
        if c == "(":
            tokens.append(("LPAREN", c))
            i += 1
            continue
        if c == ")":
            tokens.append(("RPAREN", c))
            i += 1
            continue
        if c.isalpha() or c == "_":
            end = i
            while end < len(expr) and (expr[end].isalnum() or expr[end] in "_."):
                end += 1
            tokens.append(("IDENT", expr[i:end]))
            i = end
            continue
        msg = f"unexpected character {c!r} in if-expression: {expr!r}"
        raise ValueError(msg)
    return tokens


def _render_command(command: str, arg_names: list[str], values: dict[str, Any]) -> str:
    """Substitute declared ``{arg}`` placeholders with rendered values.

    List values are space-joined; scalars are ``str()``-ed.

    cothis: the implementation pre-rewrites ``{`` → ``${`` (because
    ``string.Template`` uses ``${name}`` syntax) and then calls
    ``safe_substitute``. A declared arg with a provided value is filled in;
    a declared arg with no provided value, OR an undeclared ``{name}``, is
    left as ``${name}`` in the output — NOT preserved as the original
    ``{name}``. Ceiling: a command containing a literal ``{`` (bash
    ``${VAR}``, brace expansion) is mangled by the rewrite. Upgrade path:
    distinguish literal braces from placeholders (``{{ }}`` escape, or an
    argv mode that bypasses the template engine).
    """
    rendered: dict[str, str] = {}
    for arg_name in arg_names:
        if arg_name not in values:
            continue
        value = values[arg_name]
        rendered[arg_name] = (
            " ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        )
    template = string.Template(command.replace("{", "${"))
    # ``string.Template`` uses ``$name`` / ``${name}``; we converted ``{name}``
    # to ``${name}`` above. ``safe_substitute`` leaves unknown placeholders
    # in place rather than raising ``KeyError``.
    return template.safe_substitute(rendered)


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
