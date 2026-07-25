"""Tests for ``cothis.git`` — the git worktree list wrapper (#234).

Covers the porcelain parser directly (deterministic, no git binary
needed) + the ``list_worktrees`` integration via a stubbed subprocess
(the real binary isn't guaranteed on CI runners + can't be seeded
with a known state).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cothis.git import (
    Worktree,
    _parse_porcelain,
    find_worktree_for_path,
    list_worktrees,
)


def test_parse_porcelain_returns_paths_and_short_branches() -> None:
    """Two worktrees, one on ``main`` + one on ``feature/x``."""
    output = (
        "worktree /repo/main\n"
        "HEAD abc123\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /repo/other\n"
        "HEAD def456\n"
        "branch refs/heads/feature/x\n"
    )
    result = _parse_porcelain(output)
    assert result == [
        Worktree(Path("/repo/main"), "main"),
        Worktree(Path("/repo/other"), "feature/x"),
    ]


def test_parse_porcelain_handles_detached_head() -> None:
    """A worktree with no ``branch`` line (HEAD detached) → branch=None."""
    output = (
        "worktree /repo/detached\n"
        "HEAD abc123\n"
        "detached\n"
    )
    result = _parse_porcelain(output)
    assert result == [Worktree(Path("/repo/detached"), None)]


def test_parse_porcelain_handles_no_trailing_blank_line() -> None:
    """The last entry still emits even when the output has no trailing newline."""
    output = "worktree /repo/main\nHEAD abc\nbranch refs/heads/main"
    result = _parse_porcelain(output)
    assert len(result) == 1
    assert result[0].path == Path("/repo/main")
    assert result[0].branch == "main"


def test_parse_porcelain_handles_unknown_branch_ref_format() -> None:
    """A branch ref that isn't ``refs/heads/...`` is kept verbatim."""
    output = "worktree /repo/main\nHEAD abc\nbranch refs/tags/v1.0"
    result = _parse_porcelain(output)
    assert result == [Worktree(Path("/repo/main"), "refs/tags/v1.0")]


def test_parse_porcelain_empty_input_returns_empty_list() -> None:
    assert _parse_porcelain("") == []


def test_list_worktrees_returns_empty_when_not_a_repo(tmp_path: Path) -> None:
    """A directory without a ``.git`` → ``git`` exits 128 → empty list (no raise)."""
    result = list_worktrees(tmp_path)
    assert result == []


def test_list_worktrees_returns_empty_when_git_binary_missing(
    tmp_path: Path,
) -> None:
    """FileNotFoundError on a missing ``git`` binary → empty list (no raise)."""
    def _raise(*args, **kwargs):  # noqa: ANN002, ANN003
        raise FileNotFoundError("git not found")

    with patch("cothis.git.subprocess.run", side_effect=_raise):
        result = list_worktrees(tmp_path)
    assert result == []


def test_list_worktrees_parses_stubbed_output(tmp_path: Path) -> None:
    """End-to-end: subprocess.run returns a fake porcelain payload → Worktree list."""
    fake_output = (
        "worktree /repo/main\nHEAD abc\nbranch refs/heads/main\n\n"
        "worktree /repo/other\nHEAD def\nbranch refs/heads/y\n"
    )

    class _FakeCompleted:
        stdout = fake_output
        stderr = ""
        returncode = 0

    with patch(
        "cothis.git.subprocess.run", return_value=_FakeCompleted(),
    ):
        result = list_worktrees(tmp_path)

    assert result == [
        Worktree(Path("/repo/main"), "main"),
        Worktree(Path("/repo/other"), "y"),
    ]


def test_list_worktrees_real_repo_returns_at_least_one_entry(
    tmp_path: Path,
) -> None:
    """Integration smoke against a real ``git init`` repo + a worktree add.

    Skipped if ``git`` binary isn't available on PATH — CI runners
    always have it, dev machines usually do, but the test still
    degrades cleanly when absent.
    """
    pytest.importorskip("subprocess")
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git binary not available")

    main_repo = tmp_path / "main"
    main_repo.mkdir()
    import subprocess

    subprocess.run(
        ["git", "init"], cwd=main_repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=main_repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=main_repo, check=True, capture_output=True,
    )
    (main_repo / "README.md").write_text("hi", encoding="utf-8")
    subprocess.run(
        ["git", "add", "."], cwd=main_repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=main_repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "worktree", "add", str(tmp_path / "wt2"), "-b", "feature"],
        cwd=main_repo, check=True, capture_output=True,
    )

    result = list_worktrees(main_repo)
    paths = {wt.path for wt in result}
    assert main_repo in paths
    assert (tmp_path / "wt2") in paths
    branches = {wt.path: wt.branch for wt in result}
    # The main repo's branch may be ``master`` or ``main`` depending on
    # git defaults — verify the worktree's explicit branch at least.
    assert branches[tmp_path / "wt2"] == "feature"


# ---------------------------------------------------------------------
# find_worktree_for_path — used by SessionList enrichment (#234 AC #3)
# ---------------------------------------------------------------------


def test_find_worktree_for_path_returns_matching_worktree() -> None:
    """A path inside a worktree returns that worktree."""
    worktrees = [
        Worktree(Path("/repo/main"), "main"),
        Worktree(Path("/repo/other"), "feature"),
    ]
    result = find_worktree_for_path(Path("/repo/main/src"), worktrees)
    assert result == worktrees[0]


def test_find_worktree_for_path_returns_worktree_for_exact_match() -> None:
    """A path equal to a worktree's root returns that worktree."""
    worktrees = [Worktree(Path("/repo/main"), "main")]
    assert find_worktree_for_path(Path("/repo/main"), worktrees) == worktrees[0]


def test_find_worktree_for_path_returns_none_outside_any_worktree() -> None:
    """A path outside every worktree → None."""
    worktrees = [Worktree(Path("/repo/main"), "main")]
    assert find_worktree_for_path(Path("/elsewhere"), worktrees) is None


def test_find_worktree_for_path_empty_list_returns_none() -> None:
    """No worktrees → None for any path (degrades to plain label)."""
    assert find_worktree_for_path(Path("/anywhere"), []) is None
