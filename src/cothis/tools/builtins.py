"""Built-in filesystem tools shipped with cothis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pathspec

from cothis.tools.core import Tool, tool


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
# artifacts (``__pycache__``). Hardcoded, not configurable: every entry here
# is a directory whose contents would never help the
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
