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

import codecs
from pathlib import Path
from typing import Any

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import (
    _MAX_BYTES,
    _MAX_LINES,
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
    line (#95). Files under the byte cap stream once and are capped at
    ``_MAX_LINES`` lines per call (bounds the context cost of files with
    tens of thousands of short lines); a trailing notice names the
    resume ``start_line`` when the cap bites.
    """
    cwd = WORKDIR.get() or Path.cwd()
    resolved = _resolve_under(path, cwd)
    # Stat first to bound peak memory at _MAX_BYTES, not file size (#134).
    size = resolved.stat().st_size
    if size > _MAX_BYTES:
        # Incremental decoder handles multi-byte chars split at the cap
        # natively in a single decode pass (#312). Pre-fix, ``bytes.decode``
        # raised ``UnicodeDecodeError`` on a split surrogate, the loop
        # stripped one byte at a time and re-decoded the *entire* 1 MiB
        # buffer on each retry — up to 4× for 4-byte chars (CJK / emoji),
        # plus a full re-encode of the kept head for byte accounting.
        # ``errors="ignore"`` so the split trailing char is dropped cleanly
        # (no U+FFFD injection). ``decoder.buffer`` carries the byte count
        # for the dropped split char so ``dropped`` is byte-accurate without
        # re-encoding ``head``.
        with resolved.open("rb") as fh:
            truncated = fh.read(_MAX_BYTES)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        head = decoder.decode(truncated, final=True)
        dropped = size - (_MAX_BYTES - len(decoder.buffer))
        return head + f"\n… (truncated, {dropped} more bytes)"
    # Line-range path: stream the file once and collect at most _MAX_LINES
    # lines from ``start``. Streaming (vs ``read_text().splitlines()`` on the
    # whole file) bounds peak memory by the requested window, not the file:
    # asking for lines 5–10 of a 50k-line file now reads ~10 lines, not 50k.
    # The line cap catches files that are ≤ _MAX_BYTES yet carry tens of
    # thousands of short lines (minified JS, CSVs) — one read can't dump an
    # unbounded line count into the conversation.
    #
    # File iteration uses universal newlines (``\n`` / ``\r\n`` / ``\r``) —
    # the same line boundaries an editor counts. ``str.splitlines()`` also
    # splits on exotic separators (``\v`` / ``\f`` / `` `` …); those are
    # not line separators in source files, so iterating matches how the model
    # references lines across calls.
    start = max(1, start_line or 1)
    collected: list[str] = []
    total = 0  # 1-based count of lines read; equals file length if read to EOF
    reached_window = False
    more_after_cap = False
    with resolved.open("r", encoding="utf-8") as fh:
        for raw in fh:
            total += 1
            if total < start:
                continue
            reached_window = True
            if end_line is not None and total > end_line:
                break  # the model's requested window ended on the prior line
            if len(collected) < _MAX_LINES:
                collected.append(raw.rstrip("\n"))
            else:
                # Already emitted the cap's worth; this line proves a tail
                # exists. Stop without materialising the rest of the file —
                # ``more_after_cap`` drives the continuation notice below.
                more_after_cap = True
                break

    if not reached_window:
        # The file ended before ``start`` — ``total`` is the true line count.
        return (
            f"Error: start_line {start} is beyond EOF "
            f"(file has {total} lines)"
        )

    last_shown = start + len(collected) - 1
    width = len(str(last_shown))
    out = "\n".join(
        f"{i:>{width}}\t{collected[i - start]}" for i in range(start, last_shown + 1)
    )
    if more_after_cap:
        # Actionable continuation: tell the model where it left off and the
        # exact resume offset (the model's ``end_line`` lets us count the
        # remaining in-range lines; a full read just signals "more").
        if end_line is not None:
            remaining = end_line - last_shown
            out += (
                f"\n… ({remaining} more line(s) in range; "
                f"use start_line={last_shown + 1} to continue)"
            )
        else:
            out += (
                f"\n… (more lines in file; "
                f"use start_line={last_shown + 1} to continue)"
            )
    return out


_READ_DESCRIPTION = """Read UTF-8 text files with 1-based line numbers (tab-separated).

Pass a single path or a list. Each line in the output is prefixed
with its line number and a tab so you can reference exact lines in
``start_line`` / ``end_line`` on follow-up calls.

Single path — returns one numbered block::

    fs.read(path='config.py')
    → 1\tdebug = True
      2\tport = 8080

Multiple paths — one block per file under a ``=== <path> ===`` header::

    fs.read(path=['a.py', 'b.py'])
    → === a.py ===
        1\tprint('a')
      === b.py ===
        1\tprint('b')

In a multi-path call, one missing file produces an
``Error: file not found: <path>`` block for that file; the others
return normally (the call does not abort).
"""


@tool("fs.read", description=_READ_DESCRIPTION)
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
