"""``cothis.tools.fs.list`` — directory listing with filters.

Replaces ``fs.dir``. Supports name-glob filtering, type filtering,
recursive walks, dotfile/gitignore hygiene, and a 500-entry cap.

Backend: stdlib ``pathlib`` walker today; gated ``fd`` subprocess
lands as a follow-up. The backend choice is logged once at DEBUG
level on first use.
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
)

if TYPE_CHECKING:
    import pathspec

logger = logging.getLogger(__name__)

_backend_logged = False


def _log_backend_choice() -> None:
    """Log the backend choice once per process at DEBUG level."""
    global _backend_logged
    if not _backend_logged:
        logger.debug("fs tools: fs.list using backend stdlib")
        _backend_logged = True


def _is_excluded(
    p: Path, root: Path, gitignore: pathspec.PathSpec | None, all: bool,
) -> bool:
    """True if ``p`` should be omitted from the listing."""
    rel = p.relative_to(root)
    if any(part in _IGNORED_DIRS for part in rel.parts):
        return True
    if all:
        return False
    if any(part.startswith(".") for part in rel.parts):
        return True
    if gitignore is not None and gitignore.match_file(rel.as_posix()):
        return True
    return False


@tool("fs.list")
def list(  # noqa: A001 — shadows builtin by design (matches tool name)
    path: str = ".",
    pattern: str | None = None,
    type: str | None = None,  # noqa: A002 — matches user-facing param name
    recursive: bool = False,
    all: bool = False,  # noqa: A002
) -> list[dict[str, str]] | dict[str, Any] | str:
    """List directory entries with optional filtering.

    Returns ``[{name, type}]`` (name relative to ``path``). Use
    ``pattern`` for glob filtering, ``type`` for ``"file"`` / ``"dir"``,
    ``recursive=True`` for nested paths. Dotfiles and gitignore-excluded
    entries are hidden by default; pass ``all=True`` to show them
    (noise dirs like ``.git`` / ``__pycache__`` are always excluded).

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
    root = (cwd / path).resolve() if not Path(path).is_absolute() else Path(path)

    if not root.exists():
        return f"Error: no such directory: {path}"
    if not root.is_dir():
        return f"Error: not a directory: {path}"

    gitignore = None if all else _load_gitignore(root)

    if recursive:
        raw = sorted(
            (p for p in root.rglob("*") if not _is_excluded(p, root, gitignore, all)),
            key=lambda p: str(p.relative_to(root)),
        )
    else:
        raw = sorted(
            (p for p in root.iterdir() if not _is_excluded(p, root, gitignore, all)),
            key=lambda p: p.name,
        )

    entries: list[dict[str, str]] = []
    for p in raw:
        rel_name = p.relative_to(root).as_posix()
        p_type = "dir" if p.is_dir() else "file"
        if pattern and not fnmatch.fnmatch(p.name, pattern):
            continue
        if type and p_type != type:
            continue
        entries.append({"name": rel_name, "type": p_type})

    truncated_count = len(entries) - _MAX_DIR_ENTRIES
    if truncated_count > 0:
        return {"entries": entries[:_MAX_DIR_ENTRIES], "truncated": truncated_count}
    return entries
