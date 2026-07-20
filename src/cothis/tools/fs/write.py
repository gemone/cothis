"""``cothis.tools.fs.write`` — codex apply_patch writer.

Single-argument signature: ``fs.write(content: str) -> str`` where
``content`` is a codex ``apply_patch`` document. Parses via
:func:`cothis.tools.fs.patch.parse_patch`, applies in memory via
:func:`apply_patch`, commits to disk on success. Errors (malformed
patch, pre-image miss, two-ops-on-one-path) raise before any disk
write — multi-file patches are atomic.

cwd boundary enforcement + atomic rollback land in slices #5 / #6;
this slice keeps the ``mkdir(parents=True)`` legacy for compatibility.
"""

from __future__ import annotations

from pathlib import Path

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import WORKDIR
from cothis.tools.fs.patch import (
    AddFile,
    ApplyError,
    DeleteFile,
    PatchError,
    UpdateFile,
    apply_patch,
    parse_patch,
)


def _gather_disk_state(
    ops: list[AddFile | UpdateFile | DeleteFile], cwd: Path,
) -> dict[str, str]:
    """Read every file touched by ``ops`` into a ``{path: content}`` dict.

    Only files that exist on disk are included — missing files stay out
    so Add sees the path as absent (its precondition), and Update/Delete
    see it as missing (their precondition failure).
    """
    paths = {getattr(op, "path") for op in ops}
    state: dict[str, str] = {}
    for rel in paths:
        resolved = (cwd / rel).resolve()
        if resolved.is_file():
            state[rel] = resolved.read_text(encoding="utf-8")
    return state


def _check_one_op_per_path(ops: list[AddFile | UpdateFile | DeleteFile]) -> None:
    """Reject patches with two ops on the same path."""
    seen: dict[str, int] = {}
    for op in ops:
        path = getattr(op, "path")
        if path in seen:
            raise PatchError(
                f"more than one op on path {path!r} — split into separate calls",
                file=path,
            )
        seen[path] = 1


def _commit(
    prior: dict[str, str], post: dict[str, str], cwd: Path,
) -> tuple[int, int, int]:
    """Write the post-apply state back to disk. Returns (added, updated, deleted).

    Diffs ``prior`` (pre-apply) against ``post`` (post-apply):
    - path in both, content differs → update.
    - path in post only → add (mkdir parents, for now).
    - path in prior only → delete.
    """
    added = updated = deleted = 0
    for path in post:
        resolved = (cwd / path).resolve()
        if path in prior:
            if post[path] != prior[path]:
                resolved.write_text(post[path], encoding="utf-8")
                updated += 1
        else:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(post[path], encoding="utf-8")
            added += 1
    for path in prior:
        if path not in post:
            resolved = (cwd / path).resolve()
            if resolved.exists():
                resolved.unlink()
            deleted += 1
    return added, updated, deleted


@tool("fs.write")
def write(content: str) -> str:
    """Apply a codex ``apply_patch`` document to the working tree.

    ``content`` is a single patch document (Add/Update/Delete ops). One
    op per file per call — two ops on the same path is rejected. Errors
    (malformed patch, pre-image miss, Add-on-existing) raise
    ``PatchError`` before any disk write, so a multi-file patch is
    atomic: either every op commits or none do.

    Paths in the patch are relative to the Agent's cwd. This slice does
    not yet enforce the cwd boundary (slice #5) or rollback safety
    (slice #6) — ``mkdir(parents=True)`` is kept for compat.

    Args:
        content: A codex ``apply_patch`` document.

    Returns:
        A summary like ``"fs.write: added N, updated M, deleted K"``.
    """
    cwd = WORKDIR.get() or Path.cwd()
    ops = parse_patch(content)
    _check_one_op_per_path(ops)
    prior = _gather_disk_state(ops, cwd)
    try:
        post = apply_patch(prior, ops)
    except ApplyError:
        raise
    added, updated, deleted = _commit(prior, post, cwd)
    return f"fs.write: added {added}, updated {updated}, deleted {deleted}"
