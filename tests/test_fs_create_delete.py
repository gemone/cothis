"""Tests for ``fs.create`` + ``fs.delete`` tools (#206).

``fs.create(path, content)`` writes a new file, rejecting if it exists.
``fs.delete(path)`` removes a file, erroring if not found. Both reuse
the ``_hygiene`` boundary checks (WORKDIR, ``_resolve_under``,
``_MAX_BYTES``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cothis.tools.fs._hygiene import WORKDIR, PathBoundaryError, workdir_context

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------
# fs.create
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_create_writes_new_file(tmp_path: Path) -> None:
    """create writes content to a new file."""
    from cothis.tools.fs.create import _create

    with workdir_context(tmp_path):
        result = await _create(path="hello.txt", content="hello world\n")
    assert (tmp_path / "hello.txt").read_text() == "hello world\n"
    assert "created" in result.lower()
    assert "hello.txt" in result


@pytest.mark.asyncio
async def test_fs_create_rejects_existing(tmp_path: Path) -> None:
    """create errors when the file already exists."""
    from cothis.tools.fs.create import _create

    (tmp_path / "exists.txt").write_text("old content\n")
    with workdir_context(tmp_path):
        result = await _create(path="exists.txt", content="new\n")
    assert "already exists" in result.lower() or "error" in result.lower()
    # Original content unchanged.
    assert (tmp_path / "exists.txt").read_text() == "old content\n"


@pytest.mark.asyncio
async def test_fs_create_rejects_path_escape(tmp_path: Path) -> None:
    """create rejects ``..`` escapes."""
    from cothis.tools.fs.create import _create

    with workdir_context(tmp_path):
        result = await _create(path="../../../etc/passwd", content="x\n")
    assert "boundary" in result.lower() or "escape" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_fs_create_returns_line_count(tmp_path: Path) -> None:
    """Return message includes line count."""
    from cothis.tools.fs.create import _create

    with workdir_context(tmp_path):
        result = await _create(path="multi.py", content="line1\nline2\nline3\n")
    assert "3" in result


@pytest.mark.asyncio
async def test_fs_create_empty_content(tmp_path: Path) -> None:
    """create with empty content produces an empty file."""
    from cothis.tools.fs.create import _create

    with workdir_context(tmp_path):
        result = await _create(path="empty.txt", content="")
    assert (tmp_path / "empty.txt").read_text() == ""


@pytest.mark.asyncio
async def test_fs_create_in_subdirectory(tmp_path: Path) -> None:
    """create works when the parent directory exists."""
    from cothis.tools.fs.create import _create

    (tmp_path / "subdir").mkdir()
    with workdir_context(tmp_path):
        result = await _create(path="subdir/new.py", content="x = 1\n")
    assert (tmp_path / "subdir" / "new.py").read_text() == "x = 1\n"


@pytest.mark.asyncio
async def test_fs_create_rejects_missing_parent_dir(tmp_path: Path) -> None:
    """create errors if the parent directory doesn't exist."""
    from cothis.tools.fs.create import _create

    with workdir_context(tmp_path):
        result = await _create(path="nonexistent/file.txt", content="x\n")
    assert "error" in result.lower() or "not found" in result.lower() or "directory" in result.lower()


# ---------------------------------------------------------------------
# fs.delete
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_delete_removes_file(tmp_path: Path) -> None:
    """delete removes an existing file."""
    from cothis.tools.fs.delete import _delete

    (tmp_path / "gone.txt").write_text("bye\n")
    with workdir_context(tmp_path):
        result = await _delete(path="gone.txt")
    assert not (tmp_path / "gone.txt").exists()
    assert "deleted" in result.lower()
    assert "gone.txt" in result


@pytest.mark.asyncio
async def test_fs_delete_errors_on_missing(tmp_path: Path) -> None:
    """delete errors when the file doesn't exist."""
    from cothis.tools.fs.delete import _delete

    with workdir_context(tmp_path):
        result = await _delete(path="never_existed.txt")
    assert "not found" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_fs_delete_rejects_path_escape(tmp_path: Path) -> None:
    """delete rejects ``..`` escapes."""
    from cothis.tools.fs.delete import _delete

    with workdir_context(tmp_path):
        result = await _delete(path="../../../etc/passwd")
    assert "boundary" in result.lower() or "escape" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------
# Schema descriptions (audit standard)
# ---------------------------------------------------------------------


def test_fs_create_description_has_example() -> None:
    """create description includes a concrete example."""
    from cothis.tools import schema_for
    from cothis.tools.fs.create import _create

    desc = schema_for(_create).get("description", "")
    assert "fs.create(" in desc


def test_fs_delete_description_has_example() -> None:
    """delete description includes a concrete example."""
    from cothis.tools import schema_for
    from cothis.tools.fs.delete import _delete

    desc = schema_for(_delete).get("description", "")
    assert "fs.delete(" in desc
