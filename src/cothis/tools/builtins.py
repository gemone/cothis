"""Built-in filesystem tools shipped with cothis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pathspec

from cothis.tools.core import Tool, tool
from cothis.tools.fs.read import read
from cothis.tools.fs.write import write

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
        "gitignore", ignore_file.read_text(encoding="utf-8").splitlines()
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
        if any(part in _IGNORED_DIRS for part in rel.parts):
            return True
        if all:
            return False
        # Dotfiles / dot-directories: hidden by default ("." prefix on any
        # path component, not just the leaf — so ``.config/foo`` is hidden too).
        if any(part.startswith(".") for part in rel.parts):
            return True
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


TOOLS: list[Tool] = [read, _list_dir, write]
