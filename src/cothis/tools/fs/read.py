"""``cothis.tools.fs.read`` — multi-path file reader.

The first real resident of ``tools/fs/``. Reads one or more UTF-8 text
files with optional line ranges, returning 1-based numbered output so
the model can reference exact lines in follow-up calls.

Paths are resolved through :func:`cothis.tools.fs._hygiene._resolve_under`
against ``WORKDIR`` (set by Agent at turn entry). Absolute paths and
cwd escapes raise :class:`PathBoundaryError` which surfaces to the model
via the tool error path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import (
    _MAX_BYTES,
    _MAX_PATHS,
    WORKDIR,
    PathBoundaryError,
    _resolve_under,
)


def _read_one(path: str, start_line: int | None, end_line: int | None) -> str:
    """Read a single file's line range with 1-based numbered output.

    Returns the formatted block; the caller handles multi-file assembly.
    Path resolution failures bubble up as :class:`PathBoundaryError`.
    Per-file byte cap ``_MAX_BYTES`` is enforced here: bodies past the
    cap are truncated with a trailing ``… (truncated, N more bytes)``
    line (#95).
    """
    cwd = WORKDIR.get() or Path.cwd()
    resolved = _resolve_under(path, cwd)
    text = resolved.read_text(encoding="utf-8")
    blob = text.encode("utf-8")
    if len(blob) > _MAX_BYTES:
        dropped = len(blob) - _MAX_BYTES
        # Truncate at the byte boundary, then decode whatever survived.
        # ``blob[:_MAX_BYTES]`` may split a multi-byte char; rstrip the
        # tail bytes until decode succeeds.
        truncated = blob[:_MAX_BYTES]
        while truncated:
            try:
                head = truncated.decode("utf-8")
                break
            except UnicodeDecodeError:
                truncated = truncated[:-1]
        else:
            head = ""
        return (
            head
            + f"\n… (truncated, {dropped} more bytes)"
        )
    lines = text.splitlines()
    total = len(lines)
    start = max(1, start_line or 1)
    end = min(total, end_line or total)
    if start > total:
        return f"Error: start_line {start} is beyond EOF (file has {total} lines)"
    width = len(str(end))
    return "\n".join(
        f"{i:>{width}}\t{lines[i - 1]}" for i in range(start, end + 1)
    )


@tool("fs.read")
def read(
    path: str | list[str],
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read one or more UTF-8 text files, optionally a line range each.

    Single path → one numbered block. Multiple paths → one block per
    file under a ``=== <path> ===`` header. ``start_line`` / ``end_line``
    apply per file. Missing files in a multi-path call produce an
    ``Error:`` block for that path (no abort of the whole call).

    Paths resolve against the Agent's cwd (``WORKDIR``); absolute paths
    and ``..`` escapes outside cwd are rejected.

    Args:
        path: One path (string) or many (list of strings). Relative
            paths resolve against the Agent's cwd.
        start_line: 1-based line number to start from (inclusive).
            Applies to every file in a multi-path call.
        end_line: 1-based line number to stop at (inclusive).
            Applies to every file in a multi-path call.

    Returns:
        Numbered line range(s) with 1-based prefixes (tab-separated).
    """
    if isinstance(path, str):
        return _read_one(path, start_line, end_line)

    if len(path) > _MAX_PATHS:
        return (
            f"Error: too many paths ({len(path)}); "
            f"cap is {_MAX_PATHS} per call. Read in smaller batches "
            f"or use a more specific path."
        )

    blocks: list[str] = []
    for p in path:
        try:
            body = _read_one(p, start_line, end_line)
        except PathBoundaryError as exc:
            body = f"Error: {exc}"
        except FileNotFoundError:
            body = f"Error: file not found: {p}"
        except OSError as exc:
            body = f"Error: {type(exc).__name__}: {exc}"
        blocks.append(f"=== {p} ===\n{body}")
    return "\n\n".join(blocks)
