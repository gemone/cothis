"""``cothis.tools.fs.modify`` — line-range anchored file edit.

``fs.modify(path, start_line, end_line, content)`` replaces lines
``start_line`` through ``end_line`` (1-based, inclusive) with
``content``. The model knows line numbers from ``fs.read``, so it
references exact positions without needing diff/context-line
matching. Content can be multi-line (expands the file), single-line,
or empty (deletes those lines — the file shrinks but stays).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import (
    WORKDIR,
    PathBoundaryError,
    _resolve_under,
)

logger = logging.getLogger(__name__)


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text atomically: temp file → fsync → rename over original (#351).

    The temp file is created in the same directory as ``path`` so
    ``os.replace`` is a same-filesystem rename (atomic on POSIX +
    Windows). A crash mid-write leaves the original file untouched —
    the temp file is orphaned but the original keeps its pre-edit content.
    """
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


_MODIFY_DESCRIPTION = """Edit an existing file by replacing a line range.

Use line numbers from ``fs.read`` (1-based, inclusive). The range
``start_line`` through ``end_line`` is replaced by ``content``.
Multi-line content expands the file; empty content deletes those lines.

Example::

    fs.modify(path='app.py', start_line=10, end_line=15, content='new code\\n')
    → "fs.modify: updated app.py (lines 10-15 → 1 lines, file now 42 lines)"
"""


@tool("fs.modify", description=_MODIFY_DESCRIPTION)
async def _modify(
    path: str,
    start_line: int,
    end_line: int,
    content: str,
) -> str:
    cwd = WORKDIR.get() or Path.cwd()
    try:
        resolved = _resolve_under(path, cwd)
    except PathBoundaryError as exc:
        return f"Error: {exc}"

    if not resolved.exists():
        return f"Error: file not found: {path}"

    if start_line > end_line:
        return f"Error: start_line ({start_line}) must be ≤ end_line ({end_line})."

    if start_line < 1:
        return f"Error: start_line must be ≥ 1 (got {start_line})."

    original = resolved.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    total = len(lines)

    if start_line > total:
        return (
            f"Error: start_line ({start_line}) is past end of file "
            f"(file has {total} lines)."
        )

    actual_end = min(end_line, total)

    content_lines = content.splitlines(keepends=True) if content else []
    if content_lines and not content_lines[-1].endswith(("\n", "\r")):
        ending = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
        content_lines[-1] += ending
    new_lines = lines[: start_line - 1] + content_lines + lines[actual_end:]
    new_text = "".join(new_lines)
    _atomic_write_text(resolved, new_text)

    new_total = len(new_text.splitlines())
    return (
        f"fs.modify: updated {path} "
        f"(lines {start_line}-{actual_end} → {len(content_lines)} lines, "
        f"file now {new_total} lines)"
    )
