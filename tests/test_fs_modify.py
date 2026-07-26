"""Tests for ``fs.modify`` tool (#207).

Line-range anchored edit: replaces lines ``start_line`` through
``end_line`` (1-based, inclusive) with ``content``. Content can be
multi-line (expands), single-line, or empty (deletes those lines).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cothis.tools.fs._hygiene import workdir_context

if TYPE_CHECKING:
    from pathlib import Path


def _make_file(tmp_path: Path, name: str, lines: int) -> Path:
    """Create a file with N numbered lines for testing."""
    f = tmp_path / name
    f.write_text("\n".join(f"line{i}" for i in range(1, lines + 1)) + "\n")
    return f


# ---------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_modify_replace_single_line(tmp_path: Path) -> None:
    """start_line == end_line → replace one line."""
    from cothis.tools.fs.modify import _modify

    _make_file(tmp_path, "f.py", 5)
    with workdir_context(tmp_path):
        result = await _modify(path="f.py", start_line=3, end_line=3, content="replaced\n")
    lines = (tmp_path / "f.py").read_text().splitlines()
    assert lines[2] == "replaced"
    assert lines[0] == "line1"
    assert lines[4] == "line5"
    assert "updated" in result.lower()


@pytest.mark.asyncio
async def test_fs_modify_replace_range(tmp_path: Path) -> None:
    """start_line < end_line → replace multiple lines."""
    from cothis.tools.fs.modify import _modify

    _make_file(tmp_path, "f.py", 10)
    with workdir_context(tmp_path):
        await _modify(path="f.py", start_line=3, end_line=5, content="a\nb\nc\n")
    lines = (tmp_path / "f.py").read_text().splitlines()
    assert lines[2] == "a"
    assert lines[3] == "b"
    assert lines[4] == "c"
    assert lines[5] == "line6"


@pytest.mark.asyncio
async def test_fs_modify_expand_file(tmp_path: Path) -> None:
    """Replace 1 line with 3 → file grows."""
    from cothis.tools.fs.modify import _modify

    _make_file(tmp_path, "f.py", 5)
    with workdir_context(tmp_path):
        result = await _modify(path="f.py", start_line=2, end_line=2, content="x\ny\nz\n")
    lines = (tmp_path / "f.py").read_text().splitlines()
    assert len(lines) == 7  # 5 - 1 + 3
    assert "7" in result  # file now 7 lines


@pytest.mark.asyncio
async def test_fs_modify_shrink_with_empty_content(tmp_path: Path) -> None:
    """Empty content → delete the lines (file shrinks)."""
    from cothis.tools.fs.modify import _modify

    _make_file(tmp_path, "f.py", 10)
    with workdir_context(tmp_path):
        result = await _modify(path="f.py", start_line=4, end_line=6, content="")
    lines = (tmp_path / "f.py").read_text().splitlines()
    assert len(lines) == 7  # 10 - 3 deleted
    assert lines[0] == "line1"
    assert lines[3] == "line7"


@pytest.mark.asyncio
async def test_fs_modify_full_file_replace(tmp_path: Path) -> None:
    """start=1, end=last → full content replace."""
    from cothis.tools.fs.modify import _modify

    _make_file(tmp_path, "f.py", 5)
    with workdir_context(tmp_path):
        await _modify(path="f.py", start_line=1, end_line=5, content="brand new\nfile\n")
    lines = (tmp_path / "f.py").read_text().splitlines()
    assert lines == ["brand new", "file"]


@pytest.mark.asyncio
async def test_fs_modify_return_message(tmp_path: Path) -> None:
    """Return message shows range + new line count."""
    from cothis.tools.fs.modify import _modify

    _make_file(tmp_path, "f.py", 10)
    with workdir_context(tmp_path):
        result = await _modify(path="f.py", start_line=3, end_line=5, content="a\nb\nc\nd\n")
    assert "3" in result and "5" in result  # original range
    assert "f.py" in result
    assert "now" in result.lower()  # "file now N lines"


# ---------------------------------------------------------------------
# Edge cases — errors
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_modify_start_greater_than_end(tmp_path: Path) -> None:
    """start_line > end_line → error."""
    from cothis.tools.fs.modify import _modify

    _make_file(tmp_path, "f.py", 10)
    with workdir_context(tmp_path):
        result = await _modify(path="f.py", start_line=5, end_line=3, content="x\n")
    assert "error" in result.lower() or "must be" in result.lower()


@pytest.mark.asyncio
async def test_fs_modify_start_past_eof(tmp_path: Path) -> None:
    """start_line past EOF → error."""
    from cothis.tools.fs.modify import _modify

    _make_file(tmp_path, "f.py", 5)
    with workdir_context(tmp_path):
        result = await _modify(path="f.py", start_line=10, end_line=12, content="x\n")
    assert "error" in result.lower() or "out of range" in result.lower() or "eof" in result.lower()


@pytest.mark.asyncio
async def test_fs_modify_file_not_found(tmp_path: Path) -> None:
    """File doesn't exist → error."""
    from cothis.tools.fs.modify import _modify

    with workdir_context(tmp_path):
        result = await _modify(path="missing.py", start_line=1, end_line=1, content="x\n")
    assert "not found" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_fs_modify_path_escape(tmp_path: Path) -> None:
    """Path escape → error."""
    from cothis.tools.fs.modify import _modify

    with workdir_context(tmp_path):
        result = await _modify(path="../../../etc/passwd", start_line=1, end_line=1, content="x\n")
    assert "error" in result.lower()


# ---------------------------------------------------------------------
# Trailing newline invariant (#215)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_modify_content_without_trailing_newline(tmp_path: Path) -> None:
    """Content missing a trailing newline must not merge with the next preserved line."""
    from cothis.tools.fs.modify import _modify

    _make_file(tmp_path, "f.py", 3)
    with workdir_context(tmp_path):
        await _modify(path="f.py", start_line=2, end_line=2, content="NEW")
    text = (tmp_path / "f.py").read_text()
    assert text == "line1\nNEW\nline3\n"


@pytest.mark.asyncio
async def test_fs_modify_multiline_content_without_trailing_newline(tmp_path: Path) -> None:
    """Multi-line content whose final line lacks a newline must not merge."""
    from cothis.tools.fs.modify import _modify

    _make_file(tmp_path, "f.py", 5)
    with workdir_context(tmp_path):
        await _modify(path="f.py", start_line=2, end_line=3, content="a\nb\nc")
    text = (tmp_path / "f.py").read_text()
    assert text == "line1\na\nb\nc\nline4\nline5\n"


@pytest.mark.asyncio
async def test_fs_modify_content_with_trailing_newline_unchanged(tmp_path: Path) -> None:
    """Content already ending with a newline must not gain an extra one."""
    from cothis.tools.fs.modify import _modify

    _make_file(tmp_path, "f.py", 3)
    with workdir_context(tmp_path):
        await _modify(path="f.py", start_line=2, end_line=2, content="NEW\n")
    text = (tmp_path / "f.py").read_text()
    assert text == "line1\nNEW\nline3\n"


# ---------------------------------------------------------------------
# Schema description
# ---------------------------------------------------------------------


def test_fs_modify_description_has_example() -> None:
    """Description includes a concrete example."""
    from cothis.tools import schema_for
    from cothis.tools.fs.modify import _modify

    desc = schema_for(_modify).get("description", "")
    assert "fs.modify(" in desc


@pytest.mark.asyncio
async def test_fs_modify_leaves_no_temp_files_after_write(tmp_path: Path) -> None:
    """AC #351: atomic write leaves no ``*.tmp`` orphan after success.

    The write-temp-then-rename pattern creates a sibling temp file
    during the write. After the rename succeeds, the temp should be
    gone — if the cleanup path is broken (e.g., rename succeeds but
    the temp isn't the same inode), orphans accumulate in the user's
    source tree.

    Regression guard: after a successful ``fs.modify``, the file's
    directory should contain only the original file (no ``.tmp``).
    """
    from cothis.tools.fs.modify import _modify

    f = _make_file(tmp_path, "target.py", 5)
    dir_before = set(p.name for p in tmp_path.iterdir())

    with workdir_context(tmp_path):
        await _modify(path="target.py", start_line=2, end_line=3, content="replaced\n")

    dir_after = set(p.name for p in tmp_path.iterdir())
    # No new files should exist (temp file cleaned up by os.replace).
    assert dir_after == dir_before, (
        f"directory should be unchanged after atomic write; "
        f"new files: {dir_after - dir_before}"
    )
    # The modification landed.
    assert "replaced" in f.read_text()


@pytest.mark.asyncio
async def test_fs_modify_preserves_file_permissions(tmp_path: Path) -> None:
    """AC #351 follow-up: atomic write preserves the original file's permissions.

    ``mkstemp`` creates with 0o600; without a ``chmod`` before rename,
    a file that was 0o755 (executable) silently becomes 0o600 after
    modify. The fix captures the original mode + applies it to the temp.
    """
    import os
    import sys

    from cothis.tools.fs.modify import _modify

    f = _make_file(tmp_path, "script.sh", 3)
    os.chmod(f, 0o755)
    # Windows ignores the execute bit in chmod — capture the ACTUAL
    # mode the OS accepted, then verify the modify preserves it.
    import stat as stat_mod

    original_mode = stat_mod.S_IMODE(f.stat().st_mode)

    with workdir_context(tmp_path):
        await _modify(path="script.sh", start_line=1, end_line=1, content="#!/bin/bash\necho hi\n")

    mode = stat_mod.S_IMODE(f.stat().st_mode)
    assert mode == original_mode, (
        f"permissions should be preserved (was 0o{original_mode:o}); "
        f"got 0o{mode:o}"
    )


@pytest.mark.asyncio
async def test_fs_modify_atomic_write_cleans_up_temp_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #351: if the rename crashes mid-write, the temp is cleaned up + original is intact.

    The ``except BaseException`` block in ``_atomic_write_text`` calls
    ``os.unlink(tmp_path)`` on any failure. Without it, a crashed
    ``os.replace`` would leave a ``*.tmp`` orphan in the user's source
    tree. This test simulates the crash + verifies both the cleanup
    AND the original file's content survived.
    """
    import os as os_mod

    from cothis.tools.fs.modify import _modify

    f = _make_file(tmp_path, "target.py", 5)
    original_content = f.read_text()

    # Crash during the rename step.
    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash during os.replace")

    monkeypatch.setattr(os_mod, "replace", boom)

    with workdir_context(tmp_path):
        with pytest.raises(RuntimeError, match="simulated crash"):
            await _modify(path="target.py", start_line=1, end_line=1, content="replaced\n")

    # Original file is untouched (the rename never succeeded).
    assert f.read_text() == original_content, (
        "original content should survive a crashed rename"
    )
    # No temp file orphaned in the directory.
    files = list(tmp_path.iterdir())
    assert all(not p.name.endswith(".tmp") for p in files), (
        f"temp file should be cleaned up after crash; found: "
        f"{[p.name for p in files if p.name.endswith('.tmp')]}"
    )
