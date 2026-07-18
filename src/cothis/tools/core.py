"""Tool protocol, ``@tool`` decorator, and the hookable tool base.

The original ``tools.py`` was split into a ``tools/`` package. This module
holds the shared foundation: the ``Tool`` protocol, ``_HookableTool``
(lifecycle hooks), the ``@tool`` / ``ToolDef`` Python-tool API, the shared
YAML/MCP validation helpers (``_require`` / ``_check_unknown_keys``),
schema serialisation (``schema_for``), layer loading (``load_tools_from_layer``),
and discovery composition (``discover_tools`` — builtins + user + project merge).

YAML shell-tool pipeline lives in ``tools.yaml``; MCP servers/tools in
``tools.mcp``; built-in fs tools in ``tools.builtins``; output formatting
in ``tools.format``. ``tools/__init__.py`` re-exports the public surface.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import types
import typing
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    overload,
    runtime_checkable,
)

import griffe
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger("cothis.tools")

# Re-export surface for ``from cothis.tools.core import *``.
__all__ = [
    "AfterExecuteError",
    "HandleManager",
    "ResourceHandle",
    "Tool",
    "ToolDef",
    "discover_tools",
    "ensure_handle_ready",
    "handle_call_done",
    "load_tools_from_layer",
    "mark_inflight",
    "logger",
    "resource",
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
    # griffe logs "No type or annotation for parameter" warnings when
    # ``parent`` is None; we silence them because types come from
    # ``inspect.signature``, not griffe's cross-check.
    # cothis: ceiling — if a future griffe version emits at a different
    # level or logger name, the warnings resurface; upgrade path is to
    # construct a minimal ``parent`` Function so griffe has the signature.
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
            # First paragraph only — extended notes stay out of the schema.
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
    func: str, /, *, name: str | None = None, description: str | None = None,
    handle: type[ResourceHandle] | None = None,
) -> Callable[[Callable[..., Any]], ToolDef]: ...


@overload
def tool(
    *, name: str | None = None, description: str | None = None,
    handle: type[ResourceHandle] | None = None,
) -> Callable[[Callable[..., Any]], ToolDef]: ...


def tool(
    func: Callable[..., Any] | str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    handle: type[ResourceHandle] | None = None,
) -> ToolDef | Callable[[Callable[..., Any]], ToolDef]:
    """Decorate a function as a cothis tool with a rich LLM schema.

    Three usage forms::

        @tool                           # name from ``__name__``
        def simple(x: str) -> str: ...

        @tool("fs.read")                # positional name override
        def read(path: str) -> str: ...

        @tool(name="fs.read", description="...")  # keyword overrides
        def read(path: str) -> str: ...

    Bind a ``ResourceHandle`` so the Agent manages an external
    resource for this tool::

        @tool("db.query", handle=Db)
        async def query(sql: str) -> str:
            return await query.handle.conn.fetchval(sql)

    Returns a ``ToolDef`` instance that wraps the function. ``ToolDef``
    satisfies the ``Tool`` Protocol (``__name__`` + ``__call__``) and carries
    a pre-built OpenAI schema on ``__cothis_schema__`` (bypassing any-llm's
    lossy ``callable_to_tool``, which drops per-parameter ``description``
    fields). It also exposes the five lifecycle hook decorators
    (``.pre_load()`` / ``.after_load()`` / ``.pre_execute()`` /
    ``.after_execute()`` / ``.on_error()``) — see CONTEXT.md "Tool lifecycle".
    """

    def decorate(fn: Any) -> ToolDef:
        return ToolDef(fn, name=name, description=description, handle=handle)

    if isinstance(func, str):
        name = func
        func = None

    if func is not None:
        return decorate(func)
    return decorate


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

    __name__: str

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable[..., Any]]] = {
            stage: [] for stage in _HOOK_STAGES
        }
        self._source: str | None = None

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
                break

    def _run_load_hooks(self) -> bool:
        """Run ``pre_load`` + ``after_load`` chains. Return True if tool registers.

        Called by ``discover_tools`` AFTER cross-layer shadow resolution, on the
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
        for cb in self._hooks["pre_load"]:
            try:
                ok = cb()
            except Exception as exc:  # noqa: BLE001 — author code
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
                raise AfterExecuteError(result) from exc
        return current


class ResourceHandle:
    """An external resource a tool depends on between calls.

    Declared independently (``@resource``), bound to one or more tools
    (``@tool(handle=…)``). The Agent's ``HandleManager`` owns the instance:
    it calls ``acquire`` before the first call (and after any idle
    reclamation), ``release`` when the handle is idle past ``keepalive``
    seconds or evicted under LRU pressure. A handle shared by several
    tools is one instance with one ``last_used`` — any tool calling
    refreshes it.

    Subclasses define ``acquire`` / ``release`` and set ``keepalive``::

        @resource(keepalive=300)
        class Db(ResourceHandle):
            async def acquire(self): self.conn = await connect()
            async def release(self): await self.conn.close()

        @tool("db.query", handle=Db)
        async def query(sql: str) -> str:
            return await query.handle.conn.fetchval(sql)

    Subclasses define ``acquire`` / ``release`` and may set the class
    attributes below::

        @resource(keepalive=300, pin=True)
        class Db(ResourceHandle):
            async def acquire(self): self.conn = await connect()
            async def release(self): await self.conn.close()

        @tool("db.query", handle=Db)
        async def query(sql: str) -> str:
            return await query.handle.conn.fetchval(sql)

    The handle instance is assigned to each bound tool's ``.handle``
    attribute by the ``HandleManager``; the tool function body reads it
    there (never as a parameter — it must not pollute the LLM schema).
    """

    keepalive: float = 600.0
    #: Acquire on the Agent's first run instead of waiting for the first
    #: call. After that first acquire the normal lifecycle applies —
    #: eager handles are still reclaimed when idle or evicted.
    eager: bool = False
    #: Keep the resource alive until Agent ``aclose``: exempt from
    #: keepalive reclamation and LRU eviction, and not counted against
    #: the ``max_handles`` budget. Implies ``eager``.
    pin: bool = False

    async def acquire(self) -> None:
        """Establish the resource. Called before first use and after reclamation."""

    async def release(self) -> None:
        """Release the resource. Must be idempotent — the manager may call it
        on an already-released handle under races."""


def resource(
    cls: type[ResourceHandle] | None = None,
    *,
    keepalive: float | None = None,
    eager: bool = False,
    pin: bool = False,
) -> Any:
    """Mark a class as a ``ResourceHandle`` and optionally tune its lifecycle.

    Three forms::

        @resource
        class H(ResourceHandle): ...

        @resource(keepalive=120)
        class H(ResourceHandle): ...

        @resource(eager=True)   # acquire on first run; pin=True pins until aclose
        class H(ResourceHandle): ...
    """

    def decorate(target: type[ResourceHandle]) -> type[ResourceHandle]:
        if keepalive is not None:
            target.keepalive = keepalive
        if eager:
            target.eager = True
        if pin:
            target.pin = True
            target.eager = True  # pin implies eager: it must be alive to stay alive
        return target

    if cls is not None:
        return decorate(cls)
    return decorate


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
    - Optional ``handle`` — a ``ResourceHandle`` subclass bound via
      ``@tool(handle=…)``. The Agent's ``HandleManager`` owns the
      instance; the tool function body reaches it via ``self.handle``. The
      handle does NOT appear in the LLM schema.

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
        handle: type[ResourceHandle] | None = None,
    ) -> None:
        super().__init__()
        self._fn = fn
        tool_name = name or fn.__name__
        self.__name__ = tool_name
        self.__doc__ = description or inspect.getdoc(fn) or f"Python tool: {tool_name}"
        self.__signature__ = inspect.signature(fn)
        self.__cothis_schema__ = _build_schema(fn, tool_name, description)
        self._handle_cls: type[ResourceHandle] | None = handle
        self.handle: ResourceHandle | None = None

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
    # cothis: ceiling — unresolved annotations silently become "string"
    # in the schema rather than failing loudly. Acceptable today because
    # every shipped tool uses builtins; upgrade path: surface unresolved
    # annotations as a load-time warning so a typo in a type hint
    # doesn't silently mistype a parameter for the model.
    try:
        # ``include_extras=True`` preserves ``Annotated[T, Field(...)]`` metadata
        # so ``TypeAdapter`` below can extract constraints (story 4).
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:  # noqa: BLE001 — any hint-resolution failure is non-fatal
        hints = {}
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(pname, param.annotation)
        # Unwrap ``int | None`` → ``int`` so the type map sees the real type.
        # Handles both ``typing.Union[X, None]`` (origin is ``typing.Union``) and
        # PEP 604 ``X | None`` (origin is ``types.UnionType``).
        origin = typing.get_origin(annotation)
        if origin in (typing.Union, types.UnionType, type(None)):
            non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
            annotation = non_none[0] if non_none else str
        prop: dict[str, Any]
        try:
            # ``TypeAdapter`` honours ``Annotated[T, Field(...)]`` constraints
            # (``ge``/``le``/``min_length``/…) so they reach the LLM schema
            # (story 4). Falls back to the primitive-type map for anything
            # ``TypeAdapter`` can't handle.
            prop = TypeAdapter(annotation).json_schema()
            # pydantic returns ``{}`` for ``Any`` — always supply a ``type``
            # so the LLM schema is well-formed.
            if "type" not in prop:
                prop["type"] = _PY_JSON_TYPE.get(annotation, "string")
        except Exception:  # noqa: BLE001 — unresolved/forward-ref types fall back
            prop = {"type": _PY_JSON_TYPE.get(annotation, "string")}
        desc = arg_descs.get(pname)
        if desc:
            prop["description"] = desc
        properties[pname] = prop
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
        return
    extra = set(spec) - allowed
    if extra:
        where = f" in {source}" if source else ""
        msg = (
            f"{what}: unknown field(s) {sorted(extra)!r}{where}; "
            f"allowed: {sorted(allowed)!r}"
        )
        raise ValueError(msg)


def schema_for(tool: Tool) -> Tool | dict[str, Any]:
    """Return ``tool`` in the form any-llm's ``acompletion`` expects.

    YAML tools carry a pre-built OpenAI schema on ``__cothis_schema__`` (so
    per-arg ``description:`` text reaches the model — any-llm's
    ``callable_to_tool`` would strip it). Tools without the attribute fall
    through as callables and any-llm converts them.

    Keeping this fork here (next to ``_build_tool_schema``, the producer of
    the attribute) means ``Agent`` stays blind to the ``__cothis_schema__``
    name — the schema serialisation rule lives in ``tools.core``, where the
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
    builtin) do NOT raise — they shadow. That's ``discover_tools``'s concern
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
    is ``discover_tools``'s concern — it runs after cross-layer shadow resolution,
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
    seen: dict[str, str] = {}
    for yml in yaml_files:
        from cothis.tools.yaml import load_yaml_tools

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
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — author code, any failure is non-fatal
            logger.error("failed to import Python tool module %s: %s", py, exc)
            continue
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if isinstance(obj, ToolDef):
                _check_same_layer_duplicate(obj, str(py), seen)
                obj._source = str(py)
                tools.append(obj)
    return tools


def discover_tools(project_dir: Path, user_dir: Path) -> list[Tool]:
    """Built-in tools plus any declared in the two discovery layers.

    Loads YAML and Python tool declarations from ``user_dir`` (user-global)
    and ``project_dir`` (project-local). Both are optional; absence is not
    an error. Each directory is one **layer** (see CONTEXT.md "Layer").

    Layers resolve in ascending precedence: builtins → user-global →
    project-local. A higher-precedence tool with the same ``__name__``
    **shadows** the lower one (dict overwrite). Each shadow emits a
    ``logger.warning`` naming both layers + source paths. Same-layer
    conflicts (two YAML files in one directory, or a YAML + Python file in
    the same directory claiming one name) still raise ``ValueError`` —
    that's an author error caught by ``load_tools_from_layer``'s shared
    ``seen`` dict.

    Lifecycle hooks (``pre_load`` / ``after_load``) run AFTER shadow
    resolution, on the winning tool only. A shadowed tool's hooks never
    fire. If the winner's ``pre_load`` returns ``False`` or raises, the
    slot goes **empty — no fallback** to the shadowed tool (shadowing is a
    replacement, not a try). See ADR-0003.
    """
    from cothis.tools.builtins import TOOLS

    layers: list[tuple[str, list[Tool]]] = [
        ("builtins", TOOLS),
        ("user-global", load_tools_from_layer(user_dir)),
        ("project-local", load_tools_from_layer(project_dir)),
    ]
    registry: dict[str, Tool] = {}
    layer_of: dict[str, str] = {}
    shadow_count = 0
    for layer_name, layer_tools in layers:
        for tool in layer_tools:
            name = tool.__name__
            if name in registry:
                prev_src = getattr(registry[name], "_source", None) or "builtins"
                new_src = getattr(tool, "_source", None) or "builtins"
                logger.warning(
                    "tool %r from %s (%s) shadows %s (%s)",
                    name,
                    layer_name,
                    new_src,
                    layer_of[name],
                    prev_src,
                )
                shadow_count += 1
            registry[name] = tool
            layer_of[name] = layer_name

    registered: list[Tool] = []
    for tool in registry.values():
        run_hooks = getattr(tool, "_run_load_hooks", None)
        if run_hooks is None or run_hooks():
            registered.append(tool)
            # Per-tool DEBUG line: which tool loaded and from where. Story 43 —
            # the user-facing way to diagnose "why didn't my tool load?".
            # Tools dropped by ``pre_load`` / hooks below never reach this line;
            # they're logged at WARNING by ``_run_load_hooks``.
            src = getattr(tool, "_source", None) or "builtins"
            logger.debug("loaded tool %r from %s", tool.__name__, src)

    logger.warning(
        "discovery: %d tools active (%d shadowed)",
        len(registered),
        shadow_count,
    )
    return registered


async def ensure_handle_ready(tool: Any) -> None:
    """Ensure a tool's bound handle is acquired; no-op if it has none.

    Duck-typed like ``run_hooks_safe`` — tools without a ``_handle_cls``
    attribute (bare callables, YAML tools, ``fs.read``) skip this entirely.
    For tools with a handle, this is the self-healing path: if the
    handle was reclaimed while idle, it is re-acquired here so the tool body
    sees a live resource. Called from ``_execute`` after ``pre_execute``,
    before the tool body.
    """
    manager = getattr(tool, "_handle_manager", None)
    if manager is None:
        return
    await manager.ensure_acquired(tool)


def mark_inflight(tool: Any) -> None:
    """Mark a tool's handle as in-flight; no-op if it has none.

    Duck-typed companion to ``ensure_handle_ready`` — called from
    ``_execute`` after the handle is acquired, paired with
    ``handle_call_done`` in the body's ``finally``.
    """
    manager = getattr(tool, "_handle_manager", None)
    if manager is None:
        return
    manager.mark_inflight(tool)


def handle_call_done(tool: Any) -> None:
    """End the tool's in-flight handle window; no-op if it has none.

    Duck-typed companion to ``ensure_handle_ready`` — called from
    ``_execute``'s ``finally`` after the tool body, so a handle is never
    reclaimed mid-call (finding: the reaper only saw ``last_used`` from
    call start and could release a handle whose call outlived
    ``keepalive``).
    """
    manager = getattr(tool, "_handle_manager", None)
    if manager is None:
        return
    manager.call_done(tool)


class HandleManager:
    """Owns the lifecycle of every ``ResourceHandle`` instance in an Agent.

    A handle class bound to one or more tools (``@tool(handle=…)``) maps to
    **one** instance here — shared across all tools that reference it. The
    instance is acquired lazily on first use and reclaimed when idle past
    ``keepalive`` or evicted under LRU pressure (``max_handles``). Reclamation
    is driven by the background ``_reaper_task`` (firing every
    ``reaper_interval``); ``ensure_acquired`` is the synchronous self-heal on
    the dispatch path.

    ``last_used`` is a wall-clock timestamp (``time.time()``) per handle
    instance; any tool calling ``ensure_acquired`` refreshes it.
    """

    def __init__(self, *, max_handles: int = 8, reaper_interval: float = 60.0) -> None:
        self._max_handles = max_handles
        self._reaper_interval = reaper_interval
        self._instances: dict[type[ResourceHandle], ResourceHandle] = {}
        self._live: dict[type[ResourceHandle], float] = {}
        self._last_used: dict[type[ResourceHandle], float] = {}
        self._inflight: dict[type[ResourceHandle], int] = {}
        self._reaper_task: asyncio.Task[None] | None = None

    def _start_reaper(self) -> None:
        """Start the background idle-reaper if not already running.

        Ensures keepalive is honored while the Agent is idle between turns
        (e.g. a ``chat`` session waiting on user input). Idempotent.
        """
        if self._reaper_task is not None:
            return

        async def _reap_loop() -> None:
            while True:
                await asyncio.sleep(self._reaper_interval)
                await self.reclaim_idle()

        self._reaper_task = asyncio.ensure_future(_reap_loop())

    def bind(self, tool: Any) -> None:
        """Attach this manager to a tool and record its handle class.

        Called at Agent startup for every registered tool that has a
        ``_handle_cls``. Idempotent — multiple tools sharing one handle
        class register it once.
        """
        cls = getattr(tool, "_handle_cls", None)
        if cls is None:
            return
        tool._handle_manager = self
        if cls not in self._instances:
            self._instances[cls] = cls()

    async def start_eager(self) -> None:
        """Acquire every ``eager`` handle now (Agent's first run).

        Pinned handles imply eager (``@resource(pin=True)``), so this is
        also how pinned handles stay alive. Idempotent — already-live
        handles are skipped. Errors per handle are logged and swallowed
        (an eager handle failing to start must not abort the Agent).
        """
        for cls, instance in self._instances.items():
            if not cls.eager or cls in self._live:
                continue
            try:
                await instance.acquire()
            except Exception as exc:  # noqa: BLE001 — eager start is best-effort
                logger.warning("eager handle %s failed to start: %s", cls.__name__, exc)
                continue
            self._live[cls] = time.time()
            self._last_used[cls] = time.time()
            self._start_reaper()

    def _evictable(self, cls: type[ResourceHandle]) -> bool:
        """A handle is evictable when not in-flight and not pinned."""
        return self._inflight.get(cls, 0) == 0 and not getattr(cls, "pin", False)

    async def ensure_acquired(self, tool: Any) -> None:
        """Ensure the tool's handle is live; acquire (or re-acquire) if not.

        Refreshes ``last_used`` and begins an in-flight window: the handle
        is skipped by ``reclaim_idle`` / ``_evict_coldest`` until the Agent
        reports the call finished via ``call_done`` (``_execute``'s
        ``finally``). Under LRU pressure, evicts the coldest *evictable*
        live handle before acquiring a new one; pinned handles are neither
        evicted nor counted against ``max_handles``, so a pool full of
        pinned handles still admits a new one. If every non-pinned live
        handle is in-flight, the pool temporarily exceeds ``max_handles``
        — a tool call must never fail because the pool is busy.
        """
        cls = getattr(tool, "_handle_cls", None)
        if cls is None:
            return
        instance = self._instances.get(cls)
        if instance is None:
            instance = cls()
            self._instances[cls] = instance
        if cls not in self._live:
            if self._unpinned_count() >= self._max_handles:
                await self._evict_coldest()
            await instance.acquire()
            self._live[cls] = time.time()
            self._start_reaper()
        self._last_used[cls] = time.time()
        tool.handle = instance

    def adopt(
        self, cls: type[ResourceHandle], instance: ResourceHandle
    ) -> None:
        """Seed an already-live handle instance into the pool.

        The MCP startup path connects each server once to list its tools —
        that connection *is* the handle's first acquire. Rather than drop it
        and reconnect on the first call, the session is adopted: the instance
        is recorded as live with a fresh ``last_used``, and the next
        ``ensure_acquired`` finds it already in ``_live`` and skips acquire.
        """
        self._instances[cls] = instance
        self._live[cls] = time.time()
        self._last_used[cls] = time.time()
        self._start_reaper()

    def mark_inflight(self, tool: Any) -> None:
        """Begin a tool's in-flight window so the reaper can't reclaim it.

        Called from ``_execute`` right after ``ensure_acquired`` succeeds.
        Pairs with ``call_done`` in the body's ``finally``. Separate from
        ``ensure_acquired`` because acquiring (eager start, self-heal,
        inspection) does not always imply a call follows.
        """
        cls = getattr(tool, "_handle_cls", None)
        if cls is None:
            return
        self._inflight[cls] = self._inflight.get(cls, 0) + 1

    def _unpinned_count(self) -> int:
        return sum(1 for cls in self._live if not getattr(cls, "pin", False))

    def call_done(self, tool: Any) -> None:
        """End the tool's in-flight window and refresh ``last_used``.

        Called from ``_execute``'s ``finally`` — pairs with the increment
        in ``ensure_acquired``. The refresh restarts the keepalive window
        at call *end*, so a long call isn't reclaimed on the next reaper
        tick (its ``last_used`` would otherwise still be the call start).
        """
        cls = getattr(tool, "_handle_cls", None)
        if cls is None:
            return
        count = self._inflight.get(cls, 0)
        if count > 0:
            self._inflight[cls] = count - 1
        self._last_used[cls] = time.time()

    async def _evict_coldest(self) -> None:
        """Evict the least-recently-used evictable handle to make room.

        Only evictable handles (not in-flight, not pinned) are candidates;
        if none qualify, returns without evicting (the pool exceeds
        ``max_handles`` until a call finishes or a pinned handle is the
        only thing live — but pinned handles don't count toward the budget).
        """
        candidates = [c for c in self._live if self._evictable(c)]
        if not candidates:
            return
        coldest = min(candidates, key=lambda c: self._last_used.get(c, 0.0))
        await self._release_one(coldest)

    async def _release_one(self, cls: type[ResourceHandle]) -> None:
        instance = self._instances.get(cls)
        if instance is None or cls not in self._live:
            return
        try:
            await instance.release()
        except Exception as exc:  # noqa: BLE001 — release must not raise
            logger.debug("handle %s release error: %s", cls.__name__, exc)
        self._live.pop(cls, None)

    async def reclaim_idle(self) -> int:
        """Release handles idle past their ``keepalive``. Returns count reclaimed.

        Called between agent turns. Uses ``last_used`` + the handle class's
        ``keepalive``.         Live handles still within their window are untouched.
        """
        now = time.time()
        reclaimed = 0
        for cls in list(self._live):
            if getattr(cls, "pin", False):
                continue
            if self._inflight.get(cls, 0) > 0:
                continue
            idle = now - self._last_used.get(cls, now)
            if idle >= cls.keepalive:
                await self._release_one(cls)
                reclaimed += 1
                logger.debug("handle %s reclaimed (idle %.0fs)", cls.__name__, idle)
        return reclaimed

    async def release_all(self) -> None:
        """Release every live handle. Called at Agent ``aclose``."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except (Exception, asyncio.CancelledError):  # noqa: BLE001 — either way done
                pass
            self._reaper_task = None
        for cls in list(self._live):
            await self._release_one(cls)
        self._instances.clear()  # permanent teardown — stale entries can't re-acquire
