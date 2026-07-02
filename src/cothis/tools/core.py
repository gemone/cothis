"""Built-in tools exposed to the cothis agent.

The original ``tools.py`` was split into a ``tools/`` package; this module
holds the bulk (``Tool`` protocol, ``@tool`` decorator, built-in fs tools,
YAML shell-tool compiler, discovery). MCP servers live in ``tools.mcp`` and
output formatting in ``tools.format``; ``tools/__init__.py`` re-exports the
union so ``from cothis.tools import X`` is unchanged.
"""

from __future__ import annotations

import inspect
import logging
import shutil
import string
import subprocess
import sys
import typing
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    overload,
    runtime_checkable,
)

import griffe
import yaml

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger("cothis.tools")

# Re-export surface: every name the old flat ``tools.py`` exposed at module
# level (public + underscore-private). ``tools/__init__.py`` does
# ``from cothis.tools.core import *`` to lift these unchanged, so the
# ``from cothis.tools import X`` contract holds.
__all__ = [
    "AfterExecuteError",
    "CommandBlock",
    "Tool",
    "ToolDef",
    "load_tools_from_layer",
    "load_yaml_tools",
    "logger",
    "preview",
    "run_hooks_safe",
    "schema_for",
    "tool",
]


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


def run_hooks_safe(tool: Any, method_name: str, *args: Any) -> Any:
    """Call a ``_run_*`` hook method on ``tool`` if it exists; no-op if not.

    Real tools (``ToolDef``, ``_ShellTool``) inherit from ``_HookableTool``
    and always have the ``_run_*`` methods. Bare callables (lambdas in tests,
    legacy ``def`` tools) don't — they skip hooks entirely. This duck-types
    the hook surface so ``_execute`` doesn't need ``isinstance`` narrowing.
    """
    method = getattr(tool, method_name, None)
    if method is None:
        return args[0] if args else None
    return method(*args)


# Python type → JSON-Schema type name. Shared with the YAML arg-type map
# (``_ARG_TYPES`` / ``_JSON_TYPE`` below) so a new type lands in both places.
_PY_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Parse a Google-style docstring into (summary, per-arg descriptions).

    ``summary`` is the first paragraph (everything before ``Args:``), with
    embedded newlines collapsed to spaces. The per-arg dict maps parameter
    name → description string (also newline-collapsed). Both are empty
    string / empty dict when the docstring is absent or has no ``Args:``
    section.

    Uses ``griffe`` because docstring parsing is brittle (multi-line
    descriptions, indentation edge cases, mixed formats) and a hand-rolled
    regex parser would be a long-term source of silent schema drift.
    """
    if not doc:
        return "", {}
    # griffe emits "No type or annotation for parameter" log warnings when
    # ``parent`` is None (it can't cross-check names against a signature).
    # We don't need that cross-check — types come from ``inspect.signature``
    # — so silence griffe's logger during parse.
    # cothis: ceiling — we discard griffe's type cross-check entirely and
    # trust our own ``get_type_hints`` path. If a future griffe version
    # emits at a different log level or via a different logger name, the
    # warnings resurface; upgrade path is to construct a minimal ``parent``
    # Function object so griffe has the signature it wants.
    griffe_logger = logging.getLogger("griffe")
    old_level = griffe_logger.level
    griffe_logger.setLevel(logging.ERROR)
    try:
        parsed = griffe.Docstring(doc, parent=None).parse("google")
    finally:
        griffe_logger.setLevel(old_level)
    summary = ""
    arg_descs: dict[str, str] = {}
    for section in parsed:
        kind = section.kind.value
        if kind == "text" and not summary:
            # First text section's first paragraph only — that's the one-line
            # summary the LLM sees as the tool's top-level description. The
            # rest of the section (extended notes, parameter-behaviour prose)
            # stays out of the schema; per-arg detail lives in each parameter's
            # own ``description`` via the ``Args:`` section below.
            first_para = section.value.split("\n\n", 1)[0]
            summary = " ".join(first_para.split())
        elif kind == "parameters":
            for param in section.value:
                arg_descs[param.name] = " ".join(param.description.split())
    return summary, arg_descs


@overload
def tool(func: Callable[..., Any], /) -> ToolDef: ...


@overload
def tool(
    func: str, /, *, name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Any]], ToolDef]: ...


@overload
def tool(
    *, name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Any]], ToolDef]: ...


def tool(
    func: Callable[..., Any] | str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> ToolDef | Callable[[Callable[..., Any]], ToolDef]:
    """Decorate a function as a cothis tool with a rich LLM schema.

    Three usage forms::

        @tool                           # name from ``__name__``
        def simple(x: str) -> str: ...

        @tool("fs.read")                # positional name override
        def read(path: str) -> str: ...

        @tool(name="fs.read", description="...")  # keyword overrides
        def read(path: str) -> str: ...

    Returns a ``ToolDef`` instance that wraps the function. ``ToolDef``
    satisfies the ``Tool`` Protocol (``__name__`` + ``__call__``) and carries
    a pre-built OpenAI schema on ``__cothis_schema__`` (bypassing any-llm's
    lossy ``callable_to_tool``, which drops per-parameter ``description``
    fields). It also exposes the five lifecycle hook decorators
    (``.pre_load()`` / ``.after_load()`` / ``.pre_execute()`` /
    ``.after_execute()`` / ``.on_error()``) — see CONTEXT.md "Tool lifecycle".
    """

    def decorate(fn: Any) -> ToolDef:
        return ToolDef(fn, name=name, description=description)

    # ``@tool("fs.read")`` — positional arg is the name string, not the function.
    if isinstance(func, str):
        name = func
        func = None

    # ``@tool`` (no parens) → ``func`` is the function directly.
    if func is not None:
        return decorate(func)
    # ``@tool(...)`` (with parens) → return a decorator for later application.
    return decorate


# The five lifecycle stages a hook can register on. Used as dict keys in
# ``ToolDef._hooks`` and as the ``phase`` argument to ``on_error`` callbacks.
# Order matches the lifecycle (see CONTEXT.md "Tool lifecycle").
_HOOK_STAGES = ("pre_load", "after_load", "pre_execute", "after_execute", "on_error")


class AfterExecuteError(Exception):
    """Sentinel raised by ``_run_after_execute`` when a callback crashes.

    Carries the original tool ``result`` so ``_execute`` can recover it
    (a broken after_execute must not hide what the tool actually returned).
    The ``__cause__`` chain preserves the callback's original exception for
    debug logging.
    """

    def __init__(self, original_result: Any) -> None:
        super().__init__("after_execute callback raised; using original result")
        self.original_result = original_result


class _HookableTool:
    """Base for tools that carry lifecycle hooks.

    Owns hook storage (``_hooks``: ordered list per stage) and the four
    invocation methods (``_run_load_hooks`` / ``_run_pre_execute`` /
    ``_run_after_execute`` / ``_run_on_error``). Both ``ToolDef`` (Python
    tools) and ``_ShellTool`` (YAML tools) inherit it so ``_execute`` can
    run hooks uniformly without per-source branching.

    YAML ``_ShellTool`` inherits but never registers callbacks (its
    ``_hooks`` lists stay empty), so the hook chains are no-ops for YAML
    tools. The shared base is what lets ``_execute`` treat every tool the
    same — the ``Tool`` protocol's "no per-source branching" promise
    (CONTEXT.md) holds because the gate is the type itself, not an
    ``isinstance`` check in the dispatch path.

    This base does NOT carry schema, ``__name__``, ``__call__``, or any
    dispatch logic — those stay on the concrete subclasses. It is purely
    the hook mechanism.
    """

    __name__: str  # set by subclasses; used in logger.debug messages

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable[..., Any]]] = {
            stage: [] for stage in _HOOK_STAGES
        }
        # Source path where this tool was discovered (file path for YAML/Python
        # tools, ``None`` for builtins). Set by ``load_tools_from_layer`` after
        # construction; read by ``_all_tools`` for duplicate-name diagnostics.
        self._source: str | None = None

    # --- Hook registration (decorators that append to the chain) --------
    #
    # Each method returns a decorator that appends a callback to the stage's
    # list in registration order. The decorator returns the callback unchanged
    # so stacking works::
    #
    #     @tool.pre_execute()
    #     def first(args): ...
    #
    #     @tool.pre_execute()   # appended, runs after first
    #     def second(args): ...

    def _make_hook_decorator(
        self, stage: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that appends to ``self._hooks[stage]``."""
        chain = self._hooks[stage]

        def decorator(cb: Callable[..., Any]) -> Callable[..., Any]:
            chain.append(cb)
            return cb

        return decorator

    def pre_load(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a ``pre_load`` callback (environment gate, pre-registration)."""
        return self._make_hook_decorator("pre_load")

    def after_load(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an ``after_load`` callback (initialisation, post-registration)."""
        return self._make_hook_decorator("after_load")

    def pre_execute(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a ``pre_execute`` callback (input interception, pipeline)."""
        return self._make_hook_decorator("pre_execute")

    def after_execute(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an ``after_execute`` callback (output interception, pipeline)."""
        return self._make_hook_decorator("after_execute")

    def on_error(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an ``on_error`` callback (failure observer, side-effect only)."""
        return self._make_hook_decorator("on_error")

    # --- Hook invocation ------------------------------------------------
    #
    # Load hooks (``_run_load_hooks``) are called by ``_all_tools`` in cli.py,
    # AFTER cross-layer shadow resolution, on the winner only (see ADR-0003).
    # Execute hooks (``_run_pre_execute`` / ``_run_after_execute``) are called
    # by ``Agent._execute``. ``_run_on_error`` is called on any stage failure.

    def _run_on_error(
        self, exc: Exception, phase: str, args: Any = None, result: Any = None
    ) -> None:
        """Run the ``on_error`` chain. Pure side-effect; its own errors swallowed.

        ``phase`` names which stage raised (``pre_load`` / ``after_load`` /
        ``pre_execute`` / ``tool`` / ``after_execute``). ``args`` / ``result``
        are the context available at the point of failure (``None`` at load
        time — no args/result exist yet). Each callback receives
        ``(exc, phase, args, result)``; exceptions they raise are swallowed
        and logged at warning (the observer must not manufacture new failures —
        an observer failure is itself a startup decision worth surfacing).
        """
        for cb in self._hooks["on_error"]:
            try:
                cb(exc, phase, args, result)
            except Exception as cb_exc:  # noqa: BLE001 — observer failure is non-fatal
                logger.warning(
                    "tool %r on_error callback raised: %s; swallowed",
                    self.__name__,
                    cb_exc,
                )
                # Short-circuit the on_error chain: one observer failing means
                # the rest don't run (same short-circuit rule as other stages).
                # The original ``exc`` still propagates to the caller.
                break

    def _run_load_hooks(self) -> bool:
        """Run ``pre_load`` + ``after_load`` chains. Return True if tool registers.

        Called by ``_all_tools`` AFTER cross-layer shadow resolution, on the
        winning tool only (see ADR-0003). Returns ``True`` when the tool should
        be registered; ``False`` when ``pre_load`` short-circuited (any callback
        returned ``False``) or any load hook raised. Every skip is logged at
        ``WARNING`` — no startup decision is silent (see CONTEXT.md "Tool
        lifecycle").

        Chain semantics (see CONTEXT.md "Tool lifecycle"):
        - ``pre_load``: short-circuit AND. Any ``False`` → skip, remaining
          callbacks don't run. Any exception → ``on_error`` fires
          (phase=``"pre_load"``), then skip.
        - ``after_load``: all run in order (no short-circuit). Any exception
          → ``on_error`` fires (phase=``"after_load"``), then skip.
        """
        # --- pre_load: short-circuit AND ---
        for cb in self._hooks["pre_load"]:
            try:
                ok = cb()
            except Exception as exc:  # noqa: BLE001 — author code
                # on_error fires BEFORE the debug log — the observer should
                # see the error at the point of failure, not after the skip
                # decision is already logged.
                self._run_on_error(exc, phase="pre_load")
                logger.warning(
                    "tool %r pre_load callback raised: %s; skipping",
                    self.__name__,
                    exc,
                )
                return False
            if ok is False:
                logger.warning(
                    "tool %r pre_load callback returned False; skipping",
                    self.__name__,
                )
                return False
        # --- after_load: all run, no short-circuit ---
        for cb in self._hooks["after_load"]:
            try:
                cb()
            except Exception as exc:  # noqa: BLE001 — author code
                self._run_on_error(exc, phase="after_load")
                logger.warning(
                    "tool %r after_load callback raised: %s; skipping",
                    self.__name__,
                    exc,
                )
                return False
        return True

    def _run_pre_execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run the ``pre_execute`` pipeline. Returns the (possibly modified) args.

        Each callback receives the previous callback's returned dict; the
        final dict is what ``tool(**args)`` should receive. Exception → chain
        short-circuits, ``on_error`` fires (phase=``"pre_execute"``), and the
        exception re-raises so ``_execute`` can return the error string.
        """
        current = args
        for cb in self._hooks["pre_execute"]:
            try:
                current = cb(current)
            except Exception as exc:  # noqa: BLE001 — author code
                self._run_on_error(exc, phase="pre_execute", args=current)
                raise
        return current

    def _run_after_execute(self, result: Any, args: dict[str, Any]) -> Any:
        """Run the ``after_execute`` pipeline. Returns the (possibly modified) result.

        Each callback receives ``(previous_result, args)``; ``args`` is the
        post-pre_execute args dict, carried unchanged for context. The final
        result flows into ``_format_tool_output`` / ``str()``.

        Exception → chain short-circuits, ``on_error`` fires
        (phase=``"after_execute"``), then the exception **re-raises** so
        ``_execute`` can log the skip and use the original result. The
        re-raise carries a sentinel ``AfterExecuteError`` wrapping the
        original result, so ``_execute`` knows to use ``result`` (not the
        pipeline-so-far value) — a broken after_execute must not hide what
        the tool actually returned.
        """
        current = result
        for cb in self._hooks["after_execute"]:
            try:
                current = cb(current, args)
            except Exception as exc:  # noqa: BLE001 — author code
                self._run_on_error(
                    exc, phase="after_execute", args=args, result=current
                )
                # Re-raise with the original result attached so ``_execute``
                # can recover it. The sentinel pattern avoids a second return
                # channel — the exception IS the signal, the attribute IS the
                # fallback value.
                raise AfterExecuteError(result) from exc
        return current


class ToolDef(_HookableTool):
    """A Python tool: a callable + its schema + lifecycle hooks.

    Produced by the ``@tool`` decorator. Wraps a function with:
    - ``__name__`` / ``__doc__`` / ``__signature__`` / ``__cothis_schema__`` —
      the surface the ``Tool`` protocol + any-llm expect.
    - ``__call__(**args)`` — delegates to the wrapped function.
    - Five hook-decorator methods (inherited from ``_HookableTool``) — register
      callbacks into an ordered list per stage. Callbacks are stored here but
      invoked by the discovery loader and ``_execute``; this class only owns
      registration + storage (via the base).

    Hook callbacks are stored in ``self._hooks[stage]`` as a list in
    registration order. The decorator methods return the callback unchanged
    so stacking works::

        @tool("x")
        def x(arg: str) -> str: ...

        @x.pre_execute()
        def validate(args): ...

        @x.pre_execute()   # second callback — appended, runs after validate
        def sanitize(args): ...
    """

    __name__: str
    __doc__: str
    __signature__: inspect.Signature
    __cothis_schema__: dict[str, Any]

    def __init__(
        self,
        fn: Any,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__()
        self._fn = fn
        tool_name = name or fn.__name__
        # Schema construction: parse docstring + signature once, here.
        # ``__name__`` etc. are set so this object satisfies ``Tool`` Protocol.
        self.__name__ = tool_name
        self.__doc__ = description or inspect.getdoc(fn) or f"Python tool: {tool_name}"
        self.__signature__ = inspect.signature(fn)
        self.__cothis_schema__ = _build_schema(fn, tool_name, description)

    def __call__(self, **kwargs: Any) -> Any:
        return self._fn(**kwargs)


def _build_schema(
    fn: Any, tool_name: str, description_override: str | None
) -> dict[str, Any]:
    """Build the OpenAI-format tool schema from a function's docstring + signature.

    Reads the Google-style docstring (``griffe``) for the summary line and
    per-arg descriptions, and ``inspect.signature`` + ``typing.get_type_hints``
    for arg types + required/optional. ``description_override`` (from
    ``@tool(description=…)``) replaces the docstring summary if given.

    This is the same logic that used to live inline in the ``tool`` decorator;
    extracted so ``ToolDef.__init__`` stays focused on object construction.
    """
    summary, arg_descs = _parse_docstring(inspect.getdoc(fn))
    sig = inspect.signature(fn)
    # ``from __future__ import annotations`` makes annotations strings;
    # ``get_type_hints`` resolves them back to real type objects so the
    # ``_PY_JSON_TYPE`` lookup works. Failure (forward-ref, eval error)
    # leaves the annotation unresolved → falls back to ``string``.
    # cothis: ceiling — unresolved annotations silently become "string"
    # in the schema rather than failing loudly. Acceptable today because
    # every shipped tool uses builtins; upgrade path: surface unresolved
    # annotations as a load-time warning so a typo in a type hint
    # doesn't silently mistype a parameter for the model.
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 — any hint-resolution failure is non-fatal
        hints = {}
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue  # *args / **kwargs have no schema representation
        annotation = hints.get(pname, param.annotation)
        # ``int | None`` (Optional) resolves to a Union; unwrap the
        # non-None arm so the type map lookup sees ``int``, not ``Union``.
        # Falls back to ``string`` if the annotation can't be resolved
        # (forward ref, custom type, eval error) — the common case is a
        # builtin, so this is the safe default.
        origin = typing.get_origin(annotation)
        if origin in (typing.Union, type(None)):
            non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
            annotation = non_none[0] if non_none else str
        prop: dict[str, Any] = {"type": _PY_JSON_TYPE.get(annotation, "string")}
        desc = arg_descs.get(pname)
        if desc:
            prop["description"] = desc
        properties[pname] = prop
        # An arg is required unless it has a default. ``None`` default
        # still counts as "has a default" (the caller can omit it).
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    tool_desc = (
        description_override
        or summary
        or inspect.getdoc(fn)
        or f"Python tool: {tool_name}"
    )
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


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

    Routes on ``type:``. A declaration with ``type: mcp.stdio`` (or
    ``type: mcp.http``) is an MCP server (an ``_MCPServer`` handle producing
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
    # Lazy import breaks a module-init cycle: ``mcp`` imports ``_HookableTool``
    # from ``_core`` at module level, so a top-level ``_core`` → ``mcp`` import
    # would be circular. ``load_yaml_tools`` is the sole call site, so a
    # function-local import keeps the cycle out of import time.
    from cothis.tools.mcp import _build_mcp_http_server, _build_mcp_stdio_server

    # Peek at ``type:`` before ``_compile`` (which is shell-only and would
    # reject the MCP-specific keys). MCP tools carry a different schema (the
    # transport config — ``command``/``args``/``env`` for stdio, ``url``/
    # ``headers`` for http) so they route to their own builder, not
    # ``_compile``.
    spec = yaml.safe_load(yaml_text)
    if isinstance(spec, dict) and spec.get("type") == "mcp.stdio":
        return [_build_mcp_stdio_server(spec, source)]
    if isinstance(spec, dict) and spec.get("type") == "mcp.http":
        return [_build_mcp_http_server(spec, source)]
    # Unknown ``type:`` value → actionable error (story 30).
    if isinstance(spec, dict) and "type" in spec:
        where = f" in {source}" if source else ""
        msg = (
            f"unknown tool type {spec['type']!r}{where}; "
            f"valid: 'mcp.stdio', 'mcp.http', or omit 'type:' for a shell tool"
        )
        raise ValueError(msg)
    block = _compile(yaml_text, source=source)
    exe = _resolve_executable(block.gate_target)
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
        # Shell mode: ``_compile`` guarantees ``shell`` is non-None here
        # (it rejects string commands without a ``shell:`` field). The assert
        # turns that compile-time invariant into a runtime check + gives the
        # type checker a narrowing point.
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
    when known (from ``load_tools_from_layer``) so the error points at the
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


def schema_for(tool: Tool) -> Tool | dict[str, Any]:
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


def _check_same_layer_duplicate(tool: Tool, source: str, seen: dict[str, str]) -> None:
    """Raise ``ValueError`` if ``tool.__name__`` is already in ``seen``.

    **Same-layer** duplicate detection — called by ``load_tools_from_layer``
    with a single shared ``seen`` dict spanning both YAML and Python files
    across the entire layer directory tree (``rglob``, including
    subdirectories). A YAML file and a Python file in the same layer
    claiming one name raise here (format is never a layer; see ADR-0003).
    This is an author error, not an intentional override.

    **Cross-layer** conflicts (project-local vs user-global, custom vs
    builtin) do NOT raise — they shadow. That's ``_all_tools``'s concern
    (ascending-precedence dict overwrite + ``WARNING``), not this check.

    ``seen`` maps tool names to the first source path that declared them;
    on conflict the error names both paths so the author can fix it.
    """
    name = tool.__name__
    if name in seen:
        msg = (
            f"duplicate tool name {name!r} declared in:\n  - {seen[name]}\n  - {source}"
        )
        raise ValueError(msg)
    seen[name] = source


def load_tools_from_layer(dir_path: Path) -> list[Tool]:
    """Load every tool declaration in a directory tree (one discovery layer).

    Globs YAML (``**/*.yaml`` / ``**/*.yml``) and Python (``**/*.py``) files
    recursively, sorted by path for stable cross-platform load order. YAML
    files compile via ``load_yaml_tools``; Python files are imported via
    ``importlib`` and auto-scanned for ``ToolDef`` instances (anything
    ``@tool``-decorated at module level). No ``TOOLS`` export contract; the
    author just decorates.

    YAML and Python share a single ``seen`` dict — a YAML file and a Python
    file in the same directory claiming one ``name:`` is a **same-layer
    conflict** and raises ``ValueError`` (format is never a layer; see
    ADR-0003). This is the load-time check that catches author errors.

    Does NOT run lifecycle hooks (``pre_load`` / ``after_load``). Hook gating
    is ``_all_tools``'s concern — it runs after cross-layer shadow resolution,
    on the winning tool only (see ADR-0003).

    Empty / missing directory yields ``[]``. Python import failures
    (``ImportError``, ``SyntaxError``, module top-level errors) are logged at
    ``ERROR`` and the remaining files still load.
    """
    import importlib.util

    if not dir_path.is_dir():
        return []
    yaml_files = sorted(
        {*dir_path.rglob("*.yaml"), *dir_path.rglob("*.yml")},
        key=lambda p: str(p.relative_to(dir_path)),
    )
    py_files = sorted(
        dir_path.rglob("*.py"), key=lambda p: str(p.relative_to(dir_path))
    )
    tools: list[Tool] = []
    seen: dict[str, str] = {}  # name → source path (first occurrence, any format)
    # YAML first (matches historical builtins→YAML→Python order), then Python.
    # Both share ``seen`` so cross-format same-name conflicts raise.
    for yml in yaml_files:
        for tool_obj in load_yaml_tools(
            yml.read_text(encoding="utf-8"), source=str(yml)
        ):
            _check_same_layer_duplicate(tool_obj, str(yml), seen)
            setattr(tool_obj, "_source", str(yml))
            tools.append(tool_obj)
    for py in py_files:
        # Unique module name so repeated loads don't collide in sys.modules.
        mod_name = f"cothis_user_tools_{py.stem}_{hash(str(py)) & 0xFFFFFFFF:x}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, py)
            if spec is None or spec.loader is None:
                continue  # not a real module (e.g. namespace package)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — author code, any failure is non-fatal
            logger.error("failed to import Python tool module %s: %s", py, exc)
            continue
        # Auto-scan: collect every module-level ToolDef instance.
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if isinstance(obj, ToolDef):
                _check_same_layer_duplicate(obj, str(py), seen)
                obj._source = str(py)
                tools.append(obj)
    return tools
