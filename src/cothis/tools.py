"""Built-in tools exposed to the cothis agent."""

from __future__ import annotations

import csv
import inspect
import io
import json
import logging
import os
import shutil
import string
import subprocess
import sys
import typing
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import griffe
import pathspec
import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("cothis.tools")


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


def tool(
    func: Callable[..., Any] | str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
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


class ToolDef:
    """A tool definition: a callable + its schema + lifecycle hooks.

    Produced by the ``@tool`` decorator. Wraps a function with:
    - ``__name__`` / ``__doc__`` / ``__signature__`` / ``__cothis_schema__`` —
      the surface the ``Tool`` protocol + any-llm expect.
    - ``__call__(**args)`` — delegates to the wrapped function.
    - Five hook-decorator methods (``.pre_load()`` etc.) — register callbacks
      into an ordered list per stage. Callbacks are stored here but invoked
      by the discovery loader (#5/#6) and ``_execute`` (#7); this class only
      owns registration + storage.

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
        self._fn = fn
        tool_name = name or fn.__name__
        # Schema construction: parse docstring + signature once, here.
        # ``__name__`` etc. are set so this object satisfies ``Tool`` Protocol.
        self.__name__ = tool_name
        self.__doc__ = description or inspect.getdoc(fn) or f"Python tool: {tool_name}"
        self.__signature__ = inspect.signature(fn)
        self.__cothis_schema__ = _build_schema(fn, tool_name, description)
        # Hook storage: ordered list per stage. Populated lazily by the
        # decorator methods below; invoked by #6 (load hooks) and #7
        # (execute hooks). Empty list = no callbacks = current behavior.
        self._hooks: dict[str, list[Callable[..., Any]]] = {
            stage: [] for stage in _HOOK_STAGES
        }

    def __call__(self, **kwargs: Any) -> Any:
        return self._fn(**kwargs)

    # --- Hook registration (decorators that append to the chain) --------
    #
    # Each method takes a callback, appends it to ``self._hooks[stage]`` in
    # registration order, and returns the callback unchanged (so the decorator
    # doesn't change the callback's visibility / reference). The callbacks
    # are NOT invoked here — the discovery loader (#6) and ``_execute`` (#7)
    # are responsible for running the chains with the right semantics
    # (pipeline / short-circuit-AND / side-effect).

    def pre_load(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a ``pre_load`` callback (environment gate, pre-registration).

        Returns a decorator that appends the callback to the chain. The
        callback is invoked by the discovery loader (#6) — returns ``False``
        to skip the tool, raises to skip + trigger ``on_error``.
        """
        stage = self._hooks["pre_load"]

        def decorator(cb: Callable[..., Any]) -> Callable[..., Any]:
            stage.append(cb)
            return cb

        return decorator

    def after_load(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an ``after_load`` callback (initialisation, post-registration).

        Returns a decorator that appends the callback to the chain. Invoked by
        the discovery loader (#6) after ``pre_load`` passes — side-effect only.
        """
        stage = self._hooks["after_load"]

        def decorator(cb: Callable[..., Any]) -> Callable[..., Any]:
            stage.append(cb)
            return cb

        return decorator

    def pre_execute(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a ``pre_execute`` callback (input interception, pipeline).

        Returns a decorator that appends the callback to the chain. Invoked by
        ``_execute`` (#7) — receives ``args: dict``, returns the (possibly
        modified) dict; the final dict reaches ``tool(**args)``.
        """
        stage = self._hooks["pre_execute"]

        def decorator(cb: Callable[..., Any]) -> Callable[..., Any]:
            stage.append(cb)
            return cb

        return decorator

    def after_execute(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an ``after_execute`` callback (output interception, pipeline).

        Returns a decorator that appends the callback to the chain. Invoked by
        ``_execute`` (#7) — receives ``(result, args)``, returns the (possibly
        modified) result; flows into ``_format_tool_output`` / ``str()``.
        """
        stage = self._hooks["after_execute"]

        def decorator(cb: Callable[..., Any]) -> Callable[..., Any]:
            stage.append(cb)
            return cb

        return decorator

    def on_error(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an ``on_error`` callback (failure observer, side-effect only).

        Returns a decorator that appends the callback to the chain. Invoked
        when any prior stage raises — receives ``(exc, phase, args, result)``.
        Pure side-effect: cannot recover. Its own exceptions are swallowed.
        """
        stage = self._hooks["on_error"]

        def decorator(cb: Callable[..., Any]) -> Callable[..., Any]:
            stage.append(cb)
            return cb

        return decorator

    # --- Hook invocation (load-time stages) -----------------------------
    #
    # ``_run_load_hooks`` is called by ``load_python_tools_from_dir`` (#6) for
    # each discovered ToolDef, before adding it to the result list. Execute-
    # time hooks (``pre_execute`` / ``after_execute``) are invoked by
    # ``Agent._execute`` (#7), not here.

    def _run_on_error(
        self, exc: Exception, phase: str, args: Any = None, result: Any = None
    ) -> None:
        """Run the ``on_error`` chain. Pure side-effect; its own errors swallowed.

        ``phase`` names which stage raised (``"pre_load"`` / ``"after_load"`` /
        ``"pre_execute"`` / ``"tool"`` / ``"after_execute"``). ``args`` / ``result``
        are the context available at the point of failure (``None`` at load
        time — no args/result exist yet). Each callback receives
        ``(exc, phase, args, result)``; exceptions they raise are swallowed
        and logged at debug (the observer must not manufacture new failures).
        """
        for cb in self._hooks["on_error"]:
            try:
                cb(exc, phase, args, result)
            except Exception as cb_exc:  # noqa: BLE001 — observer failure is non-fatal
                logger.debug(
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

        Called by ``load_python_tools_from_dir`` for each discovered ToolDef.
        Returns ``True`` when the tool should be added to the result list;
        ``False`` when ``pre_load`` short-circuited (any callback returned
        ``False``) or any load hook raised.

        Chain semantics (see CONTEXT.md "Tool lifecycle"):
        - ``pre_load``: short-circuit AND. Any ``False`` → skip, remaining
          callbacks don't run. Any exception → skip, ``on_error`` fires
          (phase=``"pre_load"``).
        - ``after_load``: all run in order (no short-circuit). Any exception
          → skip, ``on_error`` fires (phase=``"after_load"``).
        """
        # --- pre_load: short-circuit AND ---
        for cb in self._hooks["pre_load"]:
            try:
                ok = cb()
            except Exception as exc:  # noqa: BLE001 — author code
                logger.debug(
                    "tool %r pre_load callback raised: %s; skipping",
                    self.__name__,
                    exc,
                )
                self._run_on_error(exc, phase="pre_load")
                return False
            if ok is False:
                logger.debug(
                    "tool %r pre_load callback returned False; skipping",
                    self.__name__,
                )
                return False
        # --- after_load: all run, no short-circuit ---
        for cb in self._hooks["after_load"]:
            try:
                cb()
            except Exception as exc:  # noqa: BLE001 — author code
                logger.debug(
                    "tool %r after_load callback raised: %s; skipping",
                    self.__name__,
                    exc,
                )
                self._run_on_error(exc, phase="after_load")
                return False
        return True


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


@tool("fs.read")
def read(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read the contents of a UTF-8 text file, optionally a line range.

    Use this to inspect an existing file before reading or modifying it.
    Without ``start_line`` / ``end_line``, returns the whole file. With
    them, returns the slice (1-based, inclusive on both ends).

    Output carries 1-based line numbers (right-aligned, tab-separated) so
    the model can reference exact lines in follow-up calls (e.g. a
    precise ``start_line`` / ``end_line`` for a large file).

    Args:
        path: Path to the file to read. Relative paths are resolved
            against the current working directory.
            eg. "src/main.py", "./README.md", "/etc/hostname".
        start_line: 1-based line number to start reading from (inclusive).
            Omit or pass null to read from the beginning of the file.
            eg. 10, 1.
        end_line: 1-based line number to stop reading at (inclusive).
            Omit or pass null to read to the end of the file.
            eg. 20, 100.

    Returns:
        The requested line range with 1-based line-number prefixes.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    total = len(lines)
    # 1-based, inclusive on both ends. ``None`` means "from start" / "to end".
    # ``end_line`` beyond EOF is clamped (the model can't know the file length);
    # ``start_line`` beyond EOF is an actionable error — returning "" would
    # give the model nothing to act on (AGENTS.md: "error messages that the
    # LLM can act on").
    start = max(1, start_line or 1)
    end = min(total, end_line or total)
    if start > total:
        return f"Error: start_line {start} is beyond EOF (file has {total} lines)"
    width = len(str(end))
    selected = [f"{i:>{width}}\t{lines[i - 1]}" for i in range(start, end + 1)]
    return "\n".join(selected)


# Directories ``fs.dir`` never descends into, even with ``all=True`` — they're
# either huge (``.venv``, ``node_modules``), not source (``.git``), or build
# artifacts (``__pycache__``). Hardcoded, not configurable: cothis is basic,
# and every entry here is a directory whose contents would never help the
# model understand a project. Listed as a module constant so future noise
# sources are added in one place.
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "node_modules",
        ".next",
        "dist",
        "build",
        "target",
        ".DS_Store",
    }
)
# Cap on entries ``fs.dir(recursive=True)`` returns. A recursive listing is
# meant for "show me the project shape", not "dump 50k site-packages files";
# a sane project fits well under this. Over-cap → truncate with a count.
_MAX_DIR_ENTRIES = 500


def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    """Load ``.gitignore`` patterns from ``root`` (the directory being listed).

    Returns a ``PathSpec`` for matching, or ``None`` if ``root`` has no
    ``.gitignore`` (so callers skip the matching pass entirely). Patterns
    are resolved relative to ``root`` — this is the simplest correct scope:
    it doesn't walk up the directory tree (cothis: YAGNI; the common case is
    a single ``.gitignore`` at the project root the user cd'd into).
    """
    ignore_file = root / ".gitignore"
    if not ignore_file.is_file():
        return None
    return pathspec.PathSpec.from_lines(
        "gitignore", ignore_file.read_text().splitlines()
    )


@tool("fs.dir")
def _list_dir(
    path: str, recursive: bool = False, all: bool = False
) -> list[dict[str, str]] | dict[str, Any] | str:
    """List the contents of a directory.

    Use this to discover the structure of a project before reading specific
    files. Returns a list of entries, each with a ``name`` (path relative to
    ``path``) and ``type`` (``"dir"`` or ``"file"``). Without ``recursive``,
    lists one level; with ``recursive=True``, walks the whole subtree.

    By default, follows the same hygiene rules ``git status`` does:
    paths matched by the directory's ``.gitignore`` are excluded, and
    dotfiles/dot-directories (``.env``, ``.config``, …) are hidden. Pass
    ``all=True`` to override both — useful when the model needs to inspect
    configuration that lives in dotfiles. Hardcoded noise directories
    (``.git``, ``.venv``, ``__pycache__``, ``node_modules``, …) are always
    excluded regardless of ``all``; their contents never help the model.

    Recursive listings are capped at 500 entries; over-cap listings include
    a ``truncated`` count so the model knows to narrow its path.

    Args:
        path: Path to the directory to list. Relative paths are resolved
            against the current working directory.
            eg. ".", "src", "./.agents/tools".
        recursive: If true, list entries recursively (the full subtree).
            Omit or pass false for a single-level listing.
        all: If true, include dotfiles and gitignore-excluded entries
            (hardcoded noise dirs are still skipped). Omit or pass false
            for the default git-hygienic listing.

    Returns:
        A list of ``{"name": <rel-path>, "type": "dir"|"file"}`` entries,
        or an ``"Error: ..."`` string if ``path`` doesn't exist or isn't a
        directory. Over-cap recursive listings come back as
        ``{"entries": [...], "truncated": <count>}``.
    """
    root = Path(path)
    if not root.exists():
        return f"Error: no such directory: {path}"
    if not root.is_dir():
        return f"Error: not a directory: {path}"

    gitignore = None if all else _load_gitignore(root)

    def _is_excluded(p: Path) -> bool:
        """True if ``p`` should be omitted from the listing."""
        rel = p.relative_to(root)
        rel_str = rel.as_posix()
        # Hardcoded noise: always excluded, even with ``all=True``.
        if any(part in _IGNORED_DIRS for part in rel.parts):
            return True
        if all:
            return False
        # Dotfiles / dot-directories: hidden by default ("." prefix on any
        # path component, not just the leaf — so ``.config/foo`` is hidden too).
        if any(part.startswith(".") for part in rel.parts):
            return True
        # ``.gitignore`` patterns.
        if gitignore is not None and gitignore.match_file(rel_str):
            return True
        return False

    if recursive:
        all_paths = sorted(
            (p for p in root.rglob("*") if not _is_excluded(p)),
            key=lambda p: str(p.relative_to(root)),
        )
    else:
        all_paths = sorted(
            (p for p in root.iterdir() if not _is_excluded(p)),
            key=lambda p: p.name,
        )

    truncated_count = len(all_paths) - _MAX_DIR_ENTRIES
    paths = all_paths[:_MAX_DIR_ENTRIES]
    entries = [
        {
            "name": p.relative_to(root).as_posix(),
            "type": "dir" if p.is_dir() else "file",
        }
        for p in paths
    ]
    if truncated_count > 0:
        return {"entries": entries, "truncated": truncated_count}
    return entries


@tool("fs.write")
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


TOOLS: list[Tool] = [read, _list_dir, write]


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
    tool it cannot dispatch on this host. The skip is silent at the UI
    layer but emits a ``logger.debug`` entry under ``cothis.tools`` — enable
    with ``--debug`` (or ``LOGLEVEL=DEBUG``) to see which tools gated off.

    See ``_compile`` for the YAML shape and ``CommandBlock`` for the
    contract. ``preview`` shares the same compile path, so the two cannot
    drift on what a valid YAMLTool is.
    """
    block = _compile(yaml_text, source=source)
    exe = _resolve_executable(block.gate_target)
    if exe is None:
        logger.debug("tool %r gated off: %s not on PATH", block.name, block.gate_target)
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


# --------------------------------------------------------------------
# Tool output formatting
#
# ``_execute`` calls ``_format_tool_output(result)`` on every structured
# (dict/list) tool result. The format is chosen via the
# ``COTHIS_TOOL_OUTPUT_FORMAT`` env var (``json`` | ``csv`` | ``tsv`` | ``yaml``),
# defaulting to ``json``.
#
# Format applicability (CSV/TSV are tabular; YAML/JSON can express anything):
# - ``list[dict]``  → table in csv/tsv; native in yaml/json.
# - ``single dict`` → one-row table (nested dicts flattened with dotted
#   key paths: ``{"a": {"b": 1}}`` → ``a.b``); native in yaml/json.
# - ``list[non-dict]`` or deeply nested → csv/tsv FALL BACK to json (a bare
#   list of scalars isn't a table; flattening would lose too much).
# - ``str`` results bypass formatting entirely (text is text).
# --------------------------------------------------------------------


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts with dotted key paths (``{"a": {"b": 1}}`` → ``{"a.b": 1}``).

    Non-dict values (including lists) are left as-is on the leaf — they'll be
    JSON-encoded per cell by the CSV writer.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _to_tabular(data: Any, delimiter: str) -> str | None:
    """Render ``data`` as CSV/TSV (``delimiter`` = ``,`` or ``\t``).

    Returns ``None`` when ``data`` isn't tabular (bare list of scalars, or a
    shape CSV can't express) — caller falls back to JSON. Nested dicts are
    flattened with dotted paths; nested lists/scalars are JSON-encoded per cell.
    """
    # Normalise to a list of single-row records.
    if isinstance(data, dict):
        rows = [_flatten_dict(data)]
    elif isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
        rows = [_flatten_dict(r) for r in data]
    else:
        # Bare list of scalars, empty list, or list with non-dict items:
        # not a table. Signal fallback.
        return None

    # Union of keys across rows preserves column discovery order.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, delimiter=delimiter)
    writer.writeheader()
    for row in rows:
        # Cells must be strings; JSON-encode non-scalars so values stay
        # model-parseable instead of Python repr.
        writer.writerow(
            {
                k: v if isinstance(v, str) else ("" if v is None else json.dumps(v))
                for k, v in row.items()
            }
        )
    return buf.getvalue().rstrip("\r\n")


def _format_tool_output(result: Any) -> str:
    """Serialise a structured tool result for the tool message.

    Format is chosen via ``COTHIS_TOOL_OUTPUT_FORMAT`` (``json`` | ``csv`` |
    ``tsv`` | ``yaml``), defaulting to ``json``. Only ``dict``/``list`` results
    go through this path; ``str`` results bypass it (text is text).

    CSV/TSV fall back to JSON when the shape isn't tabular (bare list of
    scalars, deeply nested structures). YAML handles every shape natively.
    """
    fmt = os.environ.get("COTHIS_TOOL_OUTPUT_FORMAT", "json").lower()
    if fmt in ("csv", "tsv"):
        delim = "\t" if fmt == "tsv" else ","
        rendered = _to_tabular(result, delim)
        if rendered is not None:
            return rendered
        # Non-tabular shape → fall back to JSON so nothing is lost.
    if fmt == "yaml":
        # ``allow_unicode=True`` keeps CJK / emoji readable; ``sort_keys=False``
        # preserves insertion order so the model sees fields in the author's
        # intended order.
        return yaml.dump(result, allow_unicode=True, sort_keys=False).rstrip("\n")
    return json.dumps(result)


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


def load_python_tools_from_dir(dir_path: Path) -> list[Tool]:
    """Load every Python tool module in a directory tree.

    Globs ``**/*.py`` recursively (sorted by path, stable across platforms).
    Each file is imported via ``importlib`` (no ``sys.path`` mutation), then
    the module's top-level namespace is scanned for ``ToolDef`` instances —
    anything decorated with ``@tool`` at module level is auto-discovered.
    No ``TOOLS`` export contract; the author just decorates.

    Import failures (``ImportError``, ``SyntaxError``, errors raised at
    module top level) do NOT crash the agent. The failure is logged at
    ``ERROR`` level with the file path + exception so the author can fix
    it, and the remaining files still load.

    Empty / missing directory yields ``[]``.
    """
    import importlib.util

    if not dir_path.is_dir():
        return []
    files = sorted(dir_path.rglob("*.py"), key=lambda p: str(p.relative_to(dir_path)))
    tools: list[Tool] = []
    for py in files:
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
                # Run pre_load / after_load hooks. If pre_load short-circuits
                # (any callback returns False) or any hook raises, the tool is
                # silently skipped (``on_error`` fired for audit). See
                # ``ToolDef._run_load_hooks`` for the chain semantics.
                if obj._run_load_hooks():
                    tools.append(obj)
    return tools
