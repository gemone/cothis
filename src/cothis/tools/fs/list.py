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

# Best-effort, not thread-safe — the Agent runs single-threaded per turn.
_backend_logged = False

_MAX_PATTERN_LEN = 256
_VALID_TYPES = {"file", "dir"}


def _log_backend_choice() -> None:
    global _backend_logged
    if not _backend_logged:
        logger.debug("fs tools: fs.list using backend stdlib")
        _backend_logged = True


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
            Max 256 chars; NUL rejected.
        type: Filter to ``"file"`` or ``"dir"``.
        recursive: Include nested paths.
        all: Show dotfiles + gitignore-excluded.

    Returns:
        List of ``{"name": <rel-path>, "type": "dir"|"file"}`` entries,
        or ``"Error: ..."`` if path doesn't exist.
    """
    _log_backend_choice()

    if pattern is not None:
        if "\x00" in pattern:
            return "Error: pattern contains NUL byte"
        if len(pattern) > _MAX_PATTERN_LEN:
            return f"Error: pattern exceeds {_MAX_PATTERN_LEN} chars"
    if type is not None and type not in _VALID_TYPES:
        return f"Error: type must be 'file' or 'dir', got {type!r}"

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

    def _qualifies(p: Path) -> tuple[bool, str]:
        """Filter predicate shared by the materialise + count passes.

        Returns ``(matches, p_type)``. ``p_type`` is ``""`` when the
        entry fails the type filter (so the caller can short-circuit
        without re-checking). Used by both the materialise pass (below
        cap) and the count-only drain (past cap) — the count pass
        ignores ``p_type`` and only checks ``matches``.
        """
        p_type = "dir" if p.is_dir() else "file"
        if type and p_type != type:
            return False, ""
        if pattern and not fnmatch.fnmatch(p.name, pattern):
            return False, ""
        rel = p.relative_to(root)
        if any(part in _IGNORED_DIRS for part in rel.parts):
            return False, ""
        if not all:
            if any(part.startswith(".") for part in rel.parts):
                return False, ""
            if gitignore is not None and gitignore.match_file(rel.as_posix()):
                return False, ""
        return True, p_type

    entries: list[dict[str, str]] = []
    truncated_count = 0

    walker = root.rglob("*") if recursive else root.iterdir()
    for p in walker:
        if len(entries) >= _MAX_DIR_ENTRIES:
            # cothis: cap hit — drain walker to count remaining entries
            # without materialising dicts (#116). ``p`` was yielded but
            # not appended; count it first.
            if _qualifies(p)[0]:
                truncated_count += 1
            for p_extra in walker:
                if _qualifies(p_extra)[0]:
                    truncated_count += 1
            break
        matches, p_type = _qualifies(p)
        if not matches:
            continue
        entries.append({
            "name": p.relative_to(root).as_posix(), "type": p_type,
        })

    entries.sort(key=lambda e: e["name"])
    if truncated_count > 0:
        return {"entries": entries, "truncated": truncated_count}
    return entries
