"""``cothis.tools.fs.delete`` — delete a file.

``fs.delete(path)`` removes a file. Errors if the file doesn't exist
(explicit state feedback for the model). Paths resolve against the
Agent's cwd via ``_hygiene._resolve_under``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import (
    WORKDIR,
    PathBoundaryError,
    _resolve_under,
)

logger = logging.getLogger(__name__)


_DELETE_DESCRIPTION = """Delete a file from the working tree.

The file must exist — errors if not found. Use ``fs.modify`` with
empty content to delete specific lines instead of the whole file.

Example::

    fs.delete(path='obsolete.txt')
    → "fs.delete: deleted obsolete.txt"
"""


@tool("fs.delete", description=_DELETE_DESCRIPTION)
async def _delete(path: str) -> str:
    cwd = WORKDIR.get() or Path.cwd()
    try:
        resolved = _resolve_under(path, cwd)
    except PathBoundaryError as exc:
        return f"Error: {exc}"

    if not resolved.exists():
        return f"Error: file not found: {path}"

    resolved.unlink()
    return f"fs.delete: deleted {path}"
