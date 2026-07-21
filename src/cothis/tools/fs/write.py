"""``cothis.tools.fs.write`` — codex apply_patch writer.

Single-argument signature: ``fs.write(content: str) -> str`` where
``content`` is a codex ``apply_patch`` document. Parses via
:func:`cothis.tools.fs.patch.parse_patch`, applies in memory via
:func:`apply_patch`, commits to disk on success. Errors (malformed
patch, pre-image miss, two-ops-on-one-path) raise before any disk
write — multi-file patches are atomic at the in-memory layer.

Paths resolve against the Agent's cwd (``WORKDIR``) via the path
boundary helper in the ``_hygiene`` module. Absolute paths and ``..``
escapes are rejected before any disk write; in-cwd symlinks are
followed (a link target inside cwd is allowed, outside is rejected).

Resolved paths are cached at preflight and reused at commit so a
symlink swap between the two steps can't bypass the boundary (TOCTOU
window closed). Disk commits are atomic via snapshot-and-reverse
rollback: an ``OSError`` mid-commit reverses every prior target to
its pre-call state (best-effort; rollback failure is logged but does
not mask the primary error).
"""

from __future__ import annotations

import logging
from pathlib import Path

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import (
    _MAX_BYTES,
    _MAX_PATHS,
    WORKDIR,
    PathBoundaryError,
    _resolve_under,
)
from cothis.tools.fs.patch import (
    AddFile,
    ApplyError,
    DeleteFile,
    PatchError,
    UpdateFile,
    apply_patch,
    parse_patch,
)

logger = logging.getLogger(__name__)


def _check_one_op_per_path(ops: list[AddFile | UpdateFile | DeleteFile]) -> None:
    """Reject patches with two ops on the same path."""
    seen: set[str] = set()
    for op in ops:
        path = getattr(op, "path")
        if path in seen:
            raise PatchError(
                f"more than one op on path {path!r} — split into separate calls",
                file=path,
            )
        seen.add(path)


def _preflight(
    ops: list[AddFile | UpdateFile | DeleteFile], cwd: Path,
) -> tuple[dict[str, str], dict[str, Path]]:
    """Resolve every ``op.path`` against ``cwd`` and read current disk state.

    Runs all boundary checks before any disk write, so a violation
    leaves the working tree untouched. Add ops targeting a path whose
    parent directory doesn't exist are rejected — the model must target
    an existing directory (security over convenience).

    Returns ``(state, resolved_paths)``:
    - ``state``: ``{rel_path: content}`` for every existing file the patch
      touches. Missing files stay out so Add sees the path as absent.
    - ``resolved_paths``: ``{rel_path: resolved_path}`` cached for
      ``_commit`` to reuse — closes the TOCTOU window where a symlink
      swap between preflight and commit would bypass the boundary.

    User-facing error messages keep paths relative; absolute resolved
    paths land in ``PatchError.file`` for log-only diagnostics.
    """
    state: dict[str, str] = {}
    resolved_paths: dict[str, Path] = {}
    for op in ops:
        rel = op.path
        try:
            resolved = _resolve_under(rel, cwd)
        except PathBoundaryError as exc:
            raise PatchError(
                f"absolute path or '../' escape outside cwd not allowed: {rel!r}",
                file=rel,
            ) from exc
        resolved_paths[rel] = resolved
        if isinstance(op, AddFile) and not resolved.parent.is_dir():
            raise PatchError(
                f"parent directory does not exist: {rel!r}",
                file=rel,
            )
        if resolved.is_file():
            state[rel] = resolved.read_text(encoding="utf-8")
    return state, resolved_paths


def _snapshot(resolved_paths: dict[str, Path]) -> dict[str, bytes | None]:
    """Capture each target's prior state as bytes (or ``None`` if missing).

    Used by ``_commit`` to reverse writes on OSError. Bytes (not str)
    because the reverse path uses ``write_bytes`` — decoding isn't
    necessary for rollback, and avoids any UTF-8 edge case in the
    snapshot pass.
    """
    snap: dict[str, bytes | None] = {}
    for rel, resolved in resolved_paths.items():
        try:
            snap[rel] = resolved.read_bytes()
        except FileNotFoundError:
            snap[rel] = None
    return snap


def _commit_atomic(
    prior: dict[str, str],
    post: dict[str, str],
    resolved_paths: dict[str, Path],
) -> list[tuple[str, str]]:
    """Write the post-apply state to disk atomically.

    Returns ``[(verb, rel_path), ...]`` for the summary (verb is
    ``added`` / ``updated`` / ``deleted``). On ``OSError`` mid-commit,
    reverses every committed target in reverse order to its snapshot
    state, then re-raises the original ``OSError``. Rollback failure
    on any target is logged at ERROR but does not mask the primary
    error.
    """
    snapshot = _snapshot(resolved_paths)
    committed: list[tuple[str, str]] = []

    def _rollback(primary: BaseException) -> None:
        """Reverse every committed target in reverse order, best-effort."""
        for verb, rel in reversed(committed):
            resolved = resolved_paths[rel]
            prior_bytes = snapshot[rel]
            try:
                if prior_bytes is None:
                    # Was added; remove it.
                    resolved.unlink()
                else:
                    # Was updated or deleted; restore prior bytes.
                    resolved.write_bytes(prior_bytes)
            except OSError as rollback_err:
                logger.error(
                    "fs.write rollback FAILED for %s: %s (primary error: %s)",
                    rel, rollback_err, primary,
                )

    for path in post:
        resolved = resolved_paths[path]
        if path in prior:
            if post[path] != prior[path]:
                try:
                    resolved.write_text(post[path], encoding="utf-8")
                except OSError as exc:
                    _rollback(exc)
                    raise
                committed.append(("updated", path))
        else:
            try:
                resolved.write_text(post[path], encoding="utf-8")
            except OSError as exc:
                _rollback(exc)
                raise
            committed.append(("added", path))
    for path in prior:
        if path not in post:
            resolved = resolved_paths[path]
            try:
                resolved.unlink()
            except FileNotFoundError:
                pass  # idempotent — file already gone
            except OSError as exc:
                _rollback(exc)
                raise
            committed.append(("deleted", path))
    return committed


@tool("fs.write")
def write(content: str) -> str:
    """Apply a codex ``apply_patch`` document to the working tree.

    ``content`` is a single patch document (Add/Update/Delete ops). One
    op per file per call — two ops on the same path is rejected. Errors
    (malformed patch, pre-image miss, Add-on-existing, boundary
    violation) raise ``PatchError`` before any disk write, so a
    multi-file patch is atomic at the in-memory layer: either every op
    commits or none do.

    Disk commits are atomic via snapshot-and-reverse rollback: an
    ``OSError`` mid-commit reverses every prior target to its pre-call
    state (best-effort; rollback failure is logged but does not mask
    the primary error).

    Paths resolve against the Agent's cwd (``WORKDIR``). Absolute paths
    and ``..`` escapes are rejected; in-cwd symlinks are followed
    (``Path.resolve()``), matching the hermes path-security pattern.

    Args:
        content: A codex ``apply_patch`` document.

    Returns:
        A summary listing each affected file with its verb, e.g.
        ``"fs.write: updated a.py, added lib.py"``. No diff preview —
        the patch is already in the model's context.
    """
    cwd = WORKDIR.get() or Path.cwd()
    # cothis: resource caps (#95). Patch-string byte cap fires first
    # (cheaper than parsing then rejecting); op count fires after
    # parse. Both surface actionable errors before any disk write.
    total_bytes = len(content.encode("utf-8"))
    if total_bytes > _MAX_BYTES:
        raise ValueError(
            f"fs.write patch is {total_bytes} bytes; "
            f"cap is {_MAX_BYTES // (1024 * 1024)} MiB. Split into "
            f"smaller patches."
        )
    ops = parse_patch(content)
    if len(ops) > _MAX_PATHS:
        raise ValueError(
            f"fs.write patch has {len(ops)} ops; "
            f"cap is {_MAX_PATHS} per call. Split into smaller patches."
        )
    _check_one_op_per_path(ops)
    prior, resolved_paths = _preflight(ops, cwd)
    post = apply_patch(prior, ops)  # raises ApplyError on pre-image miss etc.
    committed = _commit_atomic(prior, post, resolved_paths)
    summary = ", ".join(f"{verb} {path}" for verb, path in committed)
    return f"fs.write: {summary}" if summary else "fs.write: no changes"
