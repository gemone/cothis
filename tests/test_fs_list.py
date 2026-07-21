"""Tests for ``cothis.tools.fs.list`` — directory listing with filters.

Replaces ``fs.dir``. Stdlib backend (fd gated backend lands as follow-up).
Tests force the stdlib path via ``_have=lambda _: False``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cothis.tools.fs._hygiene import workdir_context
from cothis.tools.fs.list import _list as fs_list

if TYPE_CHECKING:
    from pathlib import Path

if TYPE_CHECKING:
    from typing import Any


def test_list_returns_structured_entries(tmp_path: Path) -> None:
    """``fs.list`` returns ``[{name, type}]`` for each entry."""
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    (tmp_path / "subdir").mkdir()

    with workdir_context(tmp_path):
        result = fs_list(path=".")
    names = sorted(e["name"] for e in result)
    assert "a.py" in names
    assert "b.txt" in names
    assert "subdir" in names
    types = {e["name"]: e["type"] for e in result}
    assert types["a.py"] == "file"
    assert types["subdir"] == "dir"


def test_list_pattern_glob_filter(tmp_path: Path) -> None:
    """``pattern="*.py"`` filters to matching names only."""
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")

    with workdir_context(tmp_path):
        result = fs_list(path=".", pattern="*.py")
    names = [e["name"] for e in result]
    assert "a.py" in names
    assert "b.txt" not in names


def test_list_type_filter(tmp_path: Path) -> None:
    """``type="file"`` returns files only; ``type="dir"`` dirs only."""
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "subdir").mkdir()

    with workdir_context(tmp_path):
        files = fs_list(path=".", type="file")
        dirs = fs_list(path=".", type="dir")
    file_names = [e["name"] for e in files]
    dir_names = [e["name"] for e in dirs]
    assert "a.py" in file_names
    assert "subdir" not in file_names
    assert "subdir" in dir_names
    assert "a.py" not in dir_names


def test_list_recursive(tmp_path: Path) -> None:
    """``recursive=True`` includes nested paths."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.py").write_text("x")

    with workdir_context(tmp_path):
        result = fs_list(path=".", recursive=True)
    names = [e["name"] for e in result]
    assert "sub/nested.py" in names


def test_list_all_shows_dotfiles(tmp_path: Path) -> None:
    """``all=True`` shows dotfiles; noise dirs still excluded."""
    (tmp_path / ".env").write_text("x")
    (tmp_path / "visible.py").write_text("y")

    with workdir_context(tmp_path):
        default_result = fs_list(path=".")
        all_result = fs_list(path=".", all=True)
    default_names = [e["name"] for e in default_result]
    all_names = [e["name"] for e in all_result]
    assert ".env" not in default_names
    assert ".env" in all_names
    assert "visible.py" in all_names


def test_list_missing_path_returns_error(tmp_path: Path) -> None:
    """Missing path → ``"Error: no such directory: ..."``."""
    with workdir_context(tmp_path):
        result = fs_list(path="nonexistent")
    assert isinstance(result, str)
    assert "Error" in result
    assert "nonexistent" in result


def test_list_noise_dirs_always_excluded(tmp_path: Path) -> None:
    """``.git``, ``__pycache__``, ``node_modules`` excluded even under ``all=True``."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "visible.py").write_text("x")

    with workdir_context(tmp_path):
        result = fs_list(path=".", all=True)
    names = [e["name"] for e in result]
    assert ".git" not in names
    assert "visible.py" in names


def test_list_truncates_past_cap(tmp_path: Path) -> None:
    """Past 500 entries, returns the actual dropped count (#116).

    The walker stops materialising entries at ``_MAX_DIR_ENTRIES``;
    a second pass counts the remaining qualifying entries without
    building their dicts. ``truncated`` is the count of dropped
    entries (10 here), not a ``-1`` sentinel.
    """
    for i in range(510):
        (tmp_path / f"f{i:03d}.txt").write_text("x")

    with workdir_context(tmp_path):
        result = fs_list(path=".")
    assert isinstance(result, dict)
    assert "entries" in result
    assert len(result["entries"]) == 500
    assert "truncated" in result
    assert result["truncated"] == 10


def test_list_truncation_count_respects_filters(tmp_path: Path) -> None:
    """The truncated count only includes entries that pass the filters.

    Seeds 500 ``.txt`` files (under cap) + 50 ``.log`` files (under cap)
    + 30 more ``.txt`` files (over cap). With ``pattern="*.txt"`` the
    walker materialises the first 500 ``.txt``, then the second pass
    counts the remaining 30 ``.txt`` — the 50 ``.log`` don't count.
    """
    for i in range(500):
        (tmp_path / f"a{i:03d}.txt").write_text("x")
    for i in range(50):
        (tmp_path / f"b{i:03d}.log").write_text("x")
    for i in range(30):
        (tmp_path / f"c{i:03d}.txt").write_text("x")

    with workdir_context(tmp_path):
        result = fs_list(path=".", pattern="*.txt")
    assert isinstance(result, dict)
    assert len(result["entries"]) == 500
    assert result["truncated"] == 30


def test_list_truncation_recursive_walks_cap_and_counts(
    tmp_path: Path,
) -> None:
    """Recursive mode: cap fires mid-walk, drain counts nested extras.

    Seeds 510 top-level files + 2 subdirs with 25 files each (562
    total qualifying entries: 510 + 50 + 2). The walker order is
    filesystem-dependent (``rglob("*")`` yields directories + files
    in entry order), so the test asserts the order-invariant count:
    cap materialises exactly 500, drain counts the remaining 62.
    """
    for i in range(510):
        (tmp_path / f"top{i:03d}.txt").write_text("x")
    sub1 = tmp_path / "sub1"
    sub2 = tmp_path / "sub2"
    sub1.mkdir()
    sub2.mkdir()
    for i in range(25):
        (sub1 / f"n{i:03d}.txt").write_text("x")
        (sub2 / f"n{i:03d}.txt").write_text("x")

    # Total qualifying entries: 510 top files + 50 nested files + 2
    # dirs = 562. Walker order is fs-dependent; the cap-vs-remainder
    # split (500 + 62) holds regardless.

    with workdir_context(tmp_path):
        result = fs_list(path=".", recursive=True)
    assert isinstance(result, dict)
    assert len(result["entries"]) == 500
    # 562 qualifying entries total (510 top + 50 nested + 2 dirs);
    # cap is 500, so the drain counts 62 regardless of walker order.
    assert result["truncated"] == 62


def test_list_truncation_count_excludes_gitignored(
    tmp_path: Path,
) -> None:
    """The drain pass honours gitignore — ignored files aren't counted.

    Seeds 500 visible files + 20 ``.log`` files (ignored via
    ``.gitignore``) + 10 extra visible files. The drain counts only
    the 10 visible extras; the 20 ignored files don't show up in
    ``truncated``.
    """
    (tmp_path / ".gitignore").write_text("*.log\n")
    for i in range(500):
        (tmp_path / f"v{i:03d}.txt").write_text("x")
    for i in range(20):
        (tmp_path / f"ignored{i:03d}.log").write_text("x")
    for i in range(10):
        (tmp_path / f"extra{i:03d}.txt").write_text("x")

    with workdir_context(tmp_path):
        result = fs_list(path=".")
    assert isinstance(result, dict)
    assert len(result["entries"]) == 500
    assert result["truncated"] == 10
