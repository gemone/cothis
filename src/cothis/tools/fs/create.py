"""``cothis.tools.fs.create`` — create a new file.

Simple-content writer: ``fs.create(path, content)``. Rejects if the
file already exists (security-over-convenience, matching
``apply_patch``'s ``Add File`` semantics). Paths resolve against the
Agent's cwd via ``_hygiene._resolve_under``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import (
    _MAX_BYTES,
    WORKDIR,
    PathBoundaryError,
    _resolve_under,
)

logger = logging.getLogger(__name__)


_CREATE_DESCRIPTION = """Create a new file with the given content.

The file must not already exist — use ``fs.modify`` to edit an
existing file, or ``fs.delete`` + ``fs.create`` to replace one.

Example::

    fs.create(path='hello.txt', content='hello world\\n')
    → "fs.create: created hello.txt (1 lines)"
"""


@tool("fs.create", description=_CREATE_DESCRIPTION)
async def _create(path: str, content: str) -> str:
    cwd = WORKDIR.get() or Path.cwd()
    try:
        resolved = _resolve_under(path, cwd)
    except PathBoundaryError as exc:
        return f"Error: {exc}"

    if resolved.exists():
        return f"Error: file already exists: {path}"

    if len(content.encode("utf-8")) > _MAX_BYTES:
        return f"Error: content exceeds {_MAX_BYTES} byte limit."

    if not resolved.parent.is_dir():
        return f"Error: parent directory does not exist: {resolved.parent.relative_to(cwd)}"

    resolved.write_text(content, encoding="utf-8")
    line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"fs.create: created {path} ({line_count} lines)"
