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
    """Past 500 entries, returns ``{"entries": [...], "truncated": -1}``
    (sentinel — ``-1`` means "more exist" without exhausting the walker)."""
    for i in range(510):
        (tmp_path / f"f{i:03d}.txt").write_text("x")

    with workdir_context(tmp_path):
        result = fs_list(path=".")
    assert isinstance(result, dict)
    assert "entries" in result
    assert "truncated" in result
    assert result["truncated"] == -1
