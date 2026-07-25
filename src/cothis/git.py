"""``cothis.git`` — read-only git introspection helpers.

Used by the session picker (TUI #234) to populate a "new session in
worktree X" picker. ``git`` itself is invoked via ``subprocess``; this
module owns the porcelain-output parsing + the "not a git repo" fallback.

Write paths (creating worktrees, checking out branches) are out of
scope — the user runs ``git worktree add`` themselves; cothis only
needs to discover what already exists.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Worktree(NamedTuple):
    """One ``git worktree`` entry — path + the branch currently checked out.

    ``branch`` is the short name (``main``, ``feature/x``) when the
    worktree is on a branch; ``None`` when detached (``HEAD`` at a
    bare commit). The picker UI displays the short name; full refs
    are recoverable via ``git rev-parse`` if a caller needs them.
    """

    path: Path
    branch: str | None


def list_worktrees(cwd: Path) -> list[Worktree]:
    """Return the worktrees git knows about, or ``[]`` if not a git repo.

    Calls ``git worktree list --porcelain`` (machine-parseable, stable
    across git versions). Empty list on:

    - ``git`` binary missing (``FileNotFoundError``)
    - not a git repo (``git`` exits non-zero)
    - any other subprocess error (logged at WARNING)

    The empty-list contract lets the picker UI degrade to "current
    directory only" without the caller distinguishing failure modes.
    """
    try:
        proc = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
    except FileNotFoundError:
        logger.debug("git: binary not on PATH; no worktrees")
        return []
    except subprocess.CalledProcessError as exc:
        # Exit code 128 = "not a git repository" — common outside repos.
        logger.debug(
            "git: worktree list failed (rc=%s): %s",
            exc.returncode, (exc.stderr or "").strip(),
        )
        return []
    except subprocess.TimeoutExpired:
        logger.warning("git: worktree list timed out after 5s")
        return []

    return _parse_porcelain(proc.stdout)


def _parse_porcelain(output: str) -> list[Worktree]:
    """Parse ``git worktree list --porcelain`` output into Worktree records.

    Porcelain format (git 2.x):

    ::

        worktree /path/to/main
        HEAD <sha>
        branch refs/heads/main

        worktree /path/to/other
        HEAD <sha>
        branch refs/heads/feature/x

        (blank line separates entries; detached entries omit ``branch``)
    """
    worktrees: list[Worktree] = []
    current_path: Path | None = None
    current_branch: str | None = None
    for line in output.splitlines():
        if not line:
            # Blank line: end of the current entry (if any).
            if current_path is not None:
                worktrees.append(Worktree(current_path, current_branch))
            current_path, current_branch = None, None
            continue
        if line.startswith("worktree "):
            current_path = Path(line[len("worktree "):])
        elif line.startswith("branch "):
            ref = line[len("branch "):]
            # refs/heads/main → main; refs/heads/feature/x → feature/x.
            prefix = "refs/heads/"
            current_branch = ref[len(prefix):] if ref.startswith(prefix) else ref
    # No trailing blank line: emit the last entry too.
    if current_path is not None:
        worktrees.append(Worktree(current_path, current_branch))
    return worktrees


def find_worktree_for_path(
    path: Path, worktrees: list[Worktree],
) -> Worktree | None:
    """Return the worktree whose ``path`` is ``path`` or an ancestor of it.

    ``None`` when ``path`` is outside every known worktree (e.g. the
    session predates a worktree move, or the user is running cothis
    outside any git repo). Used by the SessionList enrichment (#234
    AC #3) to label each session with its worktree's branch.

    Ties on equal-length ancestors resolve by iteration order — git's
    porcelain output is ordered by creation, so the most-recently
    created worktree with a matching prefix wins. In practice worktrees
    are mutually exclusive (a path belongs to exactly one), so ties
    don't occur.
    """
    for wt in worktrees:
        try:
            path.relative_to(wt.path)
        except ValueError:
            continue
        return wt
    return None
