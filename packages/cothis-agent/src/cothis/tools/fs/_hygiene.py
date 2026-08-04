"""``cothis.tools.fs._hygiene`` — WORKDIR ContextVar + path boundary.

The injection spine every fs tool reads from. The Agent sets WORKDIR
on turn entry (try/finally) so every tool call inside the turn sees
the same cwd; tools resolve user-supplied paths through
:func:`_resolve_under` which rejects absolute paths and cwd escapes.

Pure: no disk I/O. The first real consumer (``fs.read``) exercises
the WORKDIR contract; the temporary probe tool shipped in slice #2
has been removed.

cothis: ADR deferred per PRD #46 — current shape is the floor, not
the ceiling. ``contextvars`` over a schema param / ``injects=``
mechanism is the load-bearing choice; documenting it formally waits
until a second consumer validates the shape.
"""

from __future__ import annotations

import contextlib
import contextvars
from pathlib import Path
from typing import TYPE_CHECKING

import pathspec

if TYPE_CHECKING:
    from collections.abc import Iterator

WORKDIR: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "cothis.tools.fs.WORKDIR", default=None,
)

# mtime-keyed cache for parsed .gitignore PathSpec (#361). Keyed by
# (resolved_path, mtime) so a file edit (new mtime) invalidates
# automatically. Bounded by unique files × unique mtimes — a typical
# session touches 1–2 .gitignore files; no eviction needed.
_gitignore_cache: dict[tuple[Path, float], pathspec.PathSpec] = {}


class PathBoundaryError(ValueError):
    """Raised by :func:`_resolve_under` when a path escapes cwd.

    Sibling modules (e.g. ``patch``) that surface errors to the LLM
    catch this and translate to their own error type. Owned here so
    the foundation module doesn't import upward.
    """


def workdir_path() -> Path | None:
    """Return the cwd active for the current turn, or ``None`` outside a turn."""
    return WORKDIR.get()


@contextlib.contextmanager
def workdir_context(cwd: Path | None) -> Iterator[Path]:
    """Set WORKDIR for the duration of the block; reset on exit.

    The Agent wraps each ``run`` / ``run_stream`` body in this so tool
    calls see a consistent cwd. ``None`` falls back to ``Path.cwd()``.

    cothis: cwd is Agent-owned, never tool-schema-supplied — schema
    input would defeat the boundary (model could fill any path).
    """
    actual = cwd if cwd is not None else Path.cwd()
    token = WORKDIR.set(actual)
    try:
        yield actual
    finally:
        WORKDIR.reset(token)


def _resolve_under(path: str, cwd: Path) -> Path:
    """Resolve ``path`` against ``cwd``; reject absolute + cwd-escape.

    Coalesces two layers: syntactic (absolute path → reject up front)
    and post-resolve (``relative_to`` after ``resolve()`` rejects
    symlink + ``..`` escapes). Path components are the only authority.
    """
    if path.startswith("/"):
        raise PathBoundaryError(
            f"absolute paths not allowed; use paths relative to cwd: {path!r}"
        )
    cwd_resolved = cwd.resolve()
    resolved = (cwd / path).resolve()
    try:
        resolved.relative_to(cwd_resolved)
    except ValueError as exc:
        raise PathBoundaryError(
            f"path resolves outside cwd: {path!r} → {resolved}"
        ) from exc
    return resolved


# ---------------------------------------------------------------------
# Listing hygiene — shared by fs.list (#54) and fs.search (#55)
# ---------------------------------------------------------------------

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
        ".ssh",
        ".aws",
        ".gnupg",
    }
)

# 500 — a sane project fits well under this; larger listings truncate
# to bound the agent's turn budget (token + wall-clock).
_MAX_DIR_ENTRIES = 500

# Resource caps for multi-path fs ops (#95). One source of truth so
# every fs tool reads the same numbers.
# 64 — a focused call references a few files; hundreds in one call is
# almost always a mistake (the LLM can't act on 64+ paths in one turn).
_MAX_PATHS = 64
# 1 MiB — keeps a single ``fs.read`` / ``fs.write`` call from
# saturating the agent's context budget. Larger files should be read
# in slices (``start_line`` / ``end_line``) or written in chunks.
_MAX_BYTES = 1024 * 1024
# 2000 — bounds the LLM-context cost of a single unbounded ``fs.read``.
# The byte cap alone misses files that are ≤ _MAX_BYTES yet carry tens of
# thousands of short lines (minified JS, CSVs, generated code, lockfiles):
# one read used to emit every line. 2000 fits a typical source file; larger
# reads continue with ``start_line=`` (the truncation notice points there).
_MAX_LINES = 2000


def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    """Load ``.gitignore`` patterns from ``root`` (mtime-keyed cache, #361).

    Returns ``None`` if no ``.gitignore`` exists. Patterns resolve
    relative to ``root`` — this is the simplest correct scope (no
    upward walk; the common case is a single ``.gitignore`` at the
    project root).

    Caching: the parsed ``PathSpec`` is cached by ``(resolved_path,
    mtime)``. A file edit (new mtime) misses the cache → re-parse.
    Within a session, repeated ``fs.list`` / ``fs.search`` calls
    against the same root skip the file read + pathspec parse entirely
    (~2–4 ms saved per call).
    """
    ignore_file = root / ".gitignore"
    if not ignore_file.is_file():
        return None
    mtime = ignore_file.stat().st_mtime
    cache_key = (ignore_file.resolve(), mtime)
    cached = _gitignore_cache.get(cache_key)
    if cached is not None:
        return cached
    spec = pathspec.PathSpec.from_lines(
        "gitignore", ignore_file.read_text(encoding="utf-8").splitlines()
    )
    _gitignore_cache[cache_key] = spec
    return spec
