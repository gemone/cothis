"""``cothis.tools.fs.edit`` — unique-anchor string search-and-replace edit.

``fs.edit(path, old_string, new_string, replace_all=False)`` finds
``old_string`` in the file and replaces it with ``new_string``. Unlike
``fs.modify`` (line-range anchored), ``fs.edit`` is line-number-robust:
the anchor is the string content itself, so earlier edits in the same
turn that shift line numbers do not invalidate later ones. ``old_string``
must be unique unless ``replace_all=True``.
"""

from __future__ import annotations

from pathlib import Path

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import (
    _MAX_BYTES,
    WORKDIR,
    PathBoundaryError,
    _resolve_under,
)
from cothis.tools.fs.modify import _atomic_write_text

_EDIT_DESCRIPTION = """Edit an existing file by replacing a unique anchor string.

``fs.edit`` finds ``old_string`` in the file and replaces it with
``new_string``. The match is byte-exact (a substring search using
``str.count`` / ``str.replace``), including line endings — if the file
uses CRLF, ``old_string`` must too, or it will not match. Matching is
not whitespace-insensitive.

By default ``old_string`` must occur exactly once; pass
``replace_all=True`` to swap every occurrence. An empty ``new_string``
deletes the anchor (valid). ``new_string`` may legitimately equal
``old_string`` (no-op write).

Paths resolve under the agent workdir (absolute paths and ``..``
escapes are rejected). Files larger than 1 MiB are refused — read them
in chunks or use ``fs.create``. The write is atomic (temp file → fsync
→ rename), so a crash mid-write leaves the original file untouched.

Example::

    fs.edit(path='app.py', old_string='def old():\\n    pass\\n', new_string='def new():\\n    return 1\\n')
    -> "fs.edit: updated app.py (1 replacement, file now 42 lines)"

    fs.edit(path='app.py', old_string='TODO', new_string='DONE', replace_all=True)
    -> "fs.edit: updated app.py (3 replacements, file now 42 lines)"

Returns a one-line confirmation, or ``Error: ...`` when the path escapes
the workdir, the file is missing, the file is too large, ``old_string``
is empty, ``old_string`` is not found, or ``old_string`` is not unique
and ``replace_all`` is not set.
"""


@tool("fs.edit", description=_EDIT_DESCRIPTION)
async def _edit(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    cwd = WORKDIR.get() or Path.cwd()
    try:
        resolved = _resolve_under(path, cwd)
    except PathBoundaryError as exc:
        return f"Error: {exc}"

    if not resolved.exists():
        return f"Error: file not found: {path}"

    # Stat first to bound peak memory — fs.edit can't stream the way
    # fs.read does (it needs the whole file to substring-search), so
    # refuse oversized files instead of OOM-ing the worker (#419).
    size = resolved.stat().st_size
    if size > _MAX_BYTES:
        return (
            f"Error: file is {size} bytes (>{_MAX_BYTES}); "
            f"too large to modify in place — read it in chunks or use fs.create."
        )

    # Read raw bytes + decode explicitly. ``read_text`` opens in universal-
    # newline mode and translates ``\r\n`` → ``\n`` on read, which silently
    # strips CRLF and makes a CRLF-passing ``old_string`` mismatch (#371).
    # ``read_bytes().decode`` preserves the original endings so the
    # substring search is byte-exact.
    original = resolved.read_bytes().decode("utf-8")

    # Reject empty ``old_string`` BEFORE the count: ``str.count("")``
    # returns ``len(original) + 1`` and would otherwise slip through as
    # an ambiguous-match / mass-replace.
    if not old_string:
        return f"Error: old_string must not be empty in {path}"

    n = original.count(old_string)
    if n == 0:
        # No silent no-op — even with ``replace_all=True`` the caller
        # almost certainly intended to match something.
        return f"Error: old_string not found in {path}"

    if not replace_all and n > 1:
        return (
            f"Error: old_string is not unique: {n} matches in {path}; "
            f"pass replace_all=True or include more surrounding context"
        )

    new_text = (
        original.replace(old_string, new_string)
        if replace_all
        else original.replace(old_string, new_string, 1)
    )
    _atomic_write_text(resolved, new_text)

    new_total = len(new_text.splitlines())
    s = "s" if n != 1 else ""
    return (
        f"fs.edit: updated {path} "
        f"({n} replacement{s}, file now {new_total} lines)"
    )
