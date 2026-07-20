"""``cothis.tools.fs.write`` — codex apply_patch writer.

Single-argument signature: ``fs.write(content: str) -> str`` where
``content`` is a codex ``apply_patch`` document. Parses via
:func:`cothis.tools.fs.patch.parse_patch`, applies in memory via
:func:`apply_patch`, commits to disk on success. Errors (malformed
patch, pre-image miss, two-ops-on-one-path) raise before any disk
write — multi-file patches are atomic at the in-memory layer.

Paths resolve against the Agent's cwd (``WORKDIR``) via
:func:`cothis.tools.fs._hygiene._resolve_under`. Absolute paths and
``..`` escapes are rejected before any disk write; in-cwd symlinks
are followed (a link target inside cwd is allowed, outside is rejected).
"""

from __future__ import annotations

from pathlib import Path

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import WORKDIR, PathBoundaryError, _resolve_under
from cothis.tools.fs.patch import (
    AddFile,
    ApplyError,
    DeleteFile,
    PatchError,
    UpdateFile,
    apply_patch,
    parse_patch,
)


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


def _preflight(
    ops: list[AddFile | UpdateFile | DeleteFile], cwd: Path,
) -> dict[str, str]:
    """Resolve every op.path against ``cwd`` and read current disk state.

    Runs all boundary checks (``_resolve_under``) before any disk write,
    so a violation leaves the working tree untouched. Add ops targeting
    a path whose parent directory doesn't exist are rejected — the
    model must target an existing directory (security over convenience).

    Returns ``{rel_path: content}`` for every existing file the patch
    touches. Missing files stay out so Add sees the path as absent.
    """
    state: dict[str, str] = {}
    for op in ops:
        rel = op.path
        try:
            resolved = _resolve_under(rel, cwd)
        except PathBoundaryError as exc:
            raise PatchError(str(exc), file=rel) from exc
        if isinstance(op, AddFile) and not resolved.parent.is_dir():
            raise PatchError(
                f"parent directory does not exist: {resolved.parent}",
                file=rel,
            )
        if resolved.is_file():
            state[rel] = resolved.read_text(encoding="utf-8")
    return state


def _commit(
    prior: dict[str, str], post: dict[str, str], cwd: Path,
) -> tuple[int, int, int]:
    """Write the post-apply state back to disk. Returns ``(added, updated, deleted)``.

    Diffs ``prior`` (pre-apply) against ``post`` (post-apply):
    - path in both, content differs → update.
    - path in post only → add (parent dir already verified by _preflight).
    - path in prior only → delete.

    cothis: slice #6 (#53) will wrap this in snapshot+reverse; current
    shape leaves partial state on crash mid-loop. Marked so a grep finds
    the deferral when wiring atomicity.
    """
    added = updated = deleted = 0
    for path in post:
        resolved = (cwd / path).resolve()
        if path in prior:
            if post[path] != prior[path]:
                resolved.write_text(post[path], encoding="utf-8")
                updated += 1
        else:
            # _preflight already verified resolved.parent.is_dir().
            resolved.write_text(post[path], encoding="utf-8")
            added += 1
    for path in prior:
        if path not in post:
            resolved = (cwd / path).resolve()
            try:
                resolved.unlink()
            except FileNotFoundError:
                pass  # idempotent — file already gone
            else:
                deleted += 1
    return added, updated, deleted


@tool("fs.write")
def write(content: str) -> str:
    """Apply a codex ``apply_patch`` document to the working tree.

    ``content`` is a single patch document (Add/Update/Delete ops). One
    op per file per call — two ops on the same path is rejected. Errors
    (malformed patch, pre-image miss, Add-on-existing, boundary
    violation) raise ``PatchError`` before any disk write, so a
    multi-file patch is atomic at the in-memory layer: either every op
    commits or none do.

    Paths resolve against the Agent's cwd (``WORKDIR``). Absolute paths
    and ``..`` escapes are rejected; in-cwd symlinks are followed
    (``Path.resolve()``), matching the hermes path-security pattern.

    Args:
        content: A codex ``apply_patch`` document.

    Returns:
        A summary like ``"fs.write: added N, updated M, deleted K"``.
    """
    cwd = WORKDIR.get() or Path.cwd()
    ops = parse_patch(content)
    _check_one_op_per_path(ops)
    prior = _preflight(ops, cwd)
    post = apply_patch(prior, ops)  # raises ApplyError on pre-image miss etc.
    added, updated, deleted = _commit(prior, post, cwd)
    return f"fs.write: added {added}, updated {updated}, deleted {deleted}"
