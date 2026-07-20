"""``cothis.tools.fs._hygiene`` — WORKDIR ContextVar + path boundary.

The injection spine every fs tool reads from. The Agent sets WORKDIR
on turn entry (try/finally) so every tool call inside the turn sees
the same cwd; tools resolve user-supplied paths through
:func:`_resolve_under` which rejects absolute paths and cwd escapes.

Pure: no disk I/O. The temporary ``fs._cwd_probe`` tool proves the
wiring end-to-end; slice #3 deletes it once the first real fs tool
arrives.
"""

from __future__ import annotations

import contextlib
import contextvars
from pathlib import Path
from typing import TYPE_CHECKING

from cothis.tools.core import tool
from cothis.tools.fs.patch import PatchError

if TYPE_CHECKING:
    from collections.abc import Iterator

WORKDIR: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "cothis.tools.fs.WORKDIR", default=None,
)


def workdir_path() -> Path | None:
    """Return the cwd active for the current turn, or ``None`` outside a turn."""
    return WORKDIR.get()


@contextlib.contextmanager
def workdir_context(cwd: Path | None) -> Iterator[Path]:
    """Set WORKDIR for the duration of the block; reset on exit.

    The Agent wraps each ``run`` / ``run_stream`` body in this so tool
    calls see a consistent cwd. ``None`` falls back to ``Path.cwd()``.
    """
    actual = cwd if cwd is not None else Path.cwd()
    token = WORKDIR.set(actual)
    try:
        yield actual
    finally:
        WORKDIR.reset(token)


def _resolve_under(path: str, cwd: Path) -> Path:
    """Resolve ``path`` against ``cwd``; reject absolute + cwd-escape.

    The Agent owns the cwd (passed in at construction, never schema
    supplied); tools resolve user input through here so the model can't
    escape to ``/etc`` or a sibling project. Symlinks are followed
    (``Path.resolve()``), matching hermes-agent's
    ``path_security.validate_within_dir``.
    """
    if path.startswith("/"):
        raise PatchError(
            "absolute paths not allowed; use paths relative to cwd",
            file=path,
        )
    cwd_resolved = cwd.resolve()
    resolved = (cwd / path).resolve()
    try:
        resolved.relative_to(cwd_resolved)
    except ValueError as exc:
        raise PatchError(
            f"path resolves outside cwd: {path} → {resolved}",
            file=path,
        ) from exc
    return resolved


@tool
def _cwd_probe() -> str:
    """Return the cwd active for the current turn.

    Temporary: exists only to prove WORKDIR injection end-to-end. Slice
    #3 deletes this once the first real fs tool reads WORKDIR itself.
    Returns ``"<unset>"`` when called outside an Agent turn.
    """
    wd = workdir_path()
    return str(wd) if wd is not None else "<unset>"

