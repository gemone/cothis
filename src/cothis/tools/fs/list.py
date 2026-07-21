"""``cothis.tools.fs.list`` — directory listing with filters.

Replaces ``fs.dir``. Supports name-glob filtering, type filtering,
recursive walks, dotfile/gitignore hygiene, and a 500-entry cap.

cothis: fd subprocess backend + ADR deferred per PRD #46. Current
shape uses the stdlib pathlib walker exclusively; the backend choice
is logged once at DEBUG on first use.

``Path.rglob`` does not follow symlinks by default (Python 3.13+).
Symlink-loop risk is zero without an explicit ``follow_symlinks=True``
opt-in; if a future version adds one, a visited-realpath set will be
required.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import (
    _IGNORED_DIRS,
    _MAX_DIR_ENTRIES,
    WORKDIR,
    _load_gitignore,
    _resolve_under,
)

if TYPE_CHECKING:
    import pathspec

logger = logging.getLogger(__name__)

_backend_logged = False


def _log_backend_choice() -> None:
    global _backend_logged
    if not _backend_logged:
        logger.debug("fs tools: fs.list using backend stdlib")
        _backend_logged = True


def _is_excluded(
    rel_parts: tuple[str, ...],
    gitignore: pathspec.PathSpec | None,
    rel_str: str,
    all: bool,
) -> bool:
    if any(part in _IGNORED_DIRS for part in rel_parts):
        return True
    if all:
        return False
    if any(part.startswith(".") for part in rel_parts):
        return True
    if gitignore is not None and gitignore.match_file(rel_str):
        return True
    return False


@tool("fs.list")
def _list(
    path: str = ".",
    pattern: str | None = None,
    type: str | None = None,  # noqa: A002
    recursive: bool = False,
    all: bool = False,  # noqa: A002
) -> list[dict[str, str]] | dict[str, Any] | str:
    """List directory entries with optional filtering.

    Returns ``[{name, type}]`` (name relative to ``path``). Use
    ``pattern`` for glob filtering, ``type`` for ``"file"`` / ``"dir"``,
    ``recursive=True`` for nested paths. Dotfiles and gitignore-excluded
    entries are hidden by default; pass ``all=True`` to show them
    (noise dirs like ``.git`` / ``__pycache__`` are always excluded).

    Paths resolve against the Agent's cwd via the boundary helper;
    absolute paths and ``..`` escapes are rejected.

    Args:
        path: Directory to list. Relative to cwd.
        pattern: Glob pattern on entry names (e.g. ``"*.py"``).
        type: Filter to ``"file"`` or ``"dir"``.
        recursive: Include nested paths.
        all: Show dotfiles + gitignore-excluded.

    Returns:
        List of ``{"name": <rel-path>, "type": "dir"|"file"}`` entries,
        or ``"Error: ..."`` if path doesn't exist.
    """
    _log_backend_choice()
    cwd = WORKDIR.get() or Path.cwd()
    try:
        root = _resolve_under(path, cwd)
    except Exception:
        return f"Error: path outside cwd boundary: {path}"
    if not root.exists():
        return f"Error: no such directory: {path}"
    if not root.is_dir():
        return f"Error: not a directory: {path}"

    gitignore = None if all else _load_gitignore(root)

    entries: list[dict[str, str]] = []
    truncated = 0

    walker = root.rglob("*") if recursive else root.iterdir()
    for p in walker:
        if len(entries) >= _MAX_DIR_ENTRIES:
            truncated = sum(1 for _ in walker) + 1
            break
        rel = p.relative_to(root)
        rel_str = rel.as_posix()
        if _is_excluded(rel.parts, gitignore, rel_str, all):
            continue
        p_type = "dir" if p.is_dir() else "file"
        if pattern and not fnmatch.fnmatch(p.name, pattern):
            continue
        if type and p_type != type:
            continue
        entries.append({"name": rel_str, "type": p_type})

    if truncated > 0:
        entries.sort(key=lambda e: e["name"])
        return {"entries": entries[:_MAX_DIR_ENTRIES], "truncated": truncated}
    return sorted(entries, key=lambda e: e["name"])
