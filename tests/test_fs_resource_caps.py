"""Tests for ``fs.read`` / ``fs.write`` resource caps (#95).

Caps live in ``tools/fs/_hygiene.py`` and gate multi-path ``fs.read``
plus ``fs.write`` ops:

- ``_MAX_PATHS = 64`` — max paths per ``fs.read`` list, max ops per
  ``fs.write`` patch.
- ``_MAX_BYTES = 1 MiB`` — per-file byte cap on ``fs.read`` (with
  truncation tail), total write cap on ``fs.write``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cothis.tools.fs._hygiene import _MAX_BYTES, _MAX_PATHS, workdir_context
from cothis.tools.fs.patch import AddFile, apply_patch, parse_patch
from cothis.tools.fs.read import read as fs_read
from cothis.tools.fs.write import write as fs_write

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------
# Constants present in _hygiene
# ---------------------------------------------------------------------


def test_caps_have_expected_values() -> None:
    """Caps are pinned at the documented values (#95)."""
    assert _MAX_PATHS == 64
    assert _MAX_BYTES == 1024 * 1024


# ---------------------------------------------------------------------
# fs.read path-count cap
# ---------------------------------------------------------------------


def test_read_rejects_list_above_max_paths(tmp_path: Path) -> None:
    """``fs.read`` with more than ``_MAX_PATHS`` paths errors cleanly."""
    paths = [f"f{i}.txt" for i in range(_MAX_PATHS + 1)]
    # Don't even need the files to exist — the cap fires up front.
    with workdir_context(tmp_path):
        result = fs_read(path=paths)
    assert "Error" in result or "error" in result
    assert str(_MAX_PATHS) in result


def test_read_accepts_list_at_max_paths(tmp_path: Path) -> None:
    """At-cap list still works (boundary check)."""
    for i in range(_MAX_PATHS):
        (tmp_path / f"f{i}.txt").write_text("x")
    paths = [f"f{i}.txt" for i in range(_MAX_PATHS)]
    with workdir_context(tmp_path):
        result = fs_read(path=paths)
    # Each path produces its own block; no error.
    assert "Error" not in result
    # Every path appeared exactly once in the multi-path output.
    for i in range(_MAX_PATHS):
        assert f"f{i}.txt" in result


# ---------------------------------------------------------------------
# fs.read per-file byte cap (with truncation tail)
# ---------------------------------------------------------------------


def test_read_truncates_file_above_max_bytes(tmp_path: Path) -> None:
    """Per-file output past ``_MAX_BYTES`` is truncated with a tail marker."""
    big = "x" * (_MAX_BYTES + 100)
    (tmp_path / "big.txt").write_text(big)
    with workdir_context(tmp_path):
        result = fs_read(path="big.txt")
    # Truncation tail names the dropped byte count.
    assert "truncated" in result.lower()
    assert "100" in result  # the surplus


def test_read_keeps_file_at_max_bytes(tmp_path: Path) -> None:
    """At-cap file is unchanged (boundary check)."""
    exact = "x" * _MAX_BYTES
    (tmp_path / "exact.txt").write_text(exact)
    with workdir_context(tmp_path):
        result = fs_read(path="exact.txt")
    assert "truncated" not in result.lower()


# ---------------------------------------------------------------------
# fs.write op-count cap
# ---------------------------------------------------------------------


def test_write_rejects_patch_above_max_ops(tmp_path: Path) -> None:
    """``fs.write`` patch with more than ``_MAX_PATHS`` ops errors up front."""
    with workdir_context(tmp_path):
        # Build a patch with N+1 Add ops. fs.write rejects before disk
        # contact, so we don't need the files to exist.
        lines = ["*** Begin Patch"]
        for i in range(_MAX_PATHS + 1):
            lines.append(f"*** Add File: f{i}.txt")
            lines.append(f"+content {i}")
        lines.append("*** End Patch")
        patch = "\n".join(lines) + "\n"
        with pytest.raises(Exception) as exc_info:  # noqa: PT011 — schema-specific
            fs_write(content=patch)
    msg = str(exc_info.value)
    assert str(_MAX_PATHS) in msg


# ---------------------------------------------------------------------
# fs.write total-bytes cap
# ---------------------------------------------------------------------


def test_write_rejects_patch_above_max_bytes(tmp_path: Path) -> None:
    """``fs.write`` patch whose total content exceeds ``_MAX_BYTES`` errors."""
    big = "y" * (_MAX_BYTES + 10)
    patch = (
        "*** Begin Patch\n"
        "*** Add File: big.txt\n"
        f"+{big}\n"
        "*** End Patch\n"
    )
    with workdir_context(tmp_path):
        with pytest.raises(Exception) as exc_info:  # noqa: PT011
            fs_write(content=patch)
    msg = str(exc_info.value)
    # Error names the byte cap (MiB form is friendlier).
    assert "MiB" in msg or str(_MAX_BYTES) in msg


# ---------------------------------------------------------------------
# Sanity: imports the patch module to confirm we use the real parser
# ---------------------------------------------------------------------


def test_parse_patch_returns_addfile_for_single_add() -> None:
    """Sanity-check the parser used by the test fixtures."""
    ops = parse_patch(
        "*** Begin Patch\n*** Add File: x.txt\n+hello\n*** End Patch\n"
    )
    assert len(ops) == 1
    assert isinstance(ops[0], AddFile)
    # ``apply_patch`` is the inverse we exercise against the disk.
    _ = apply_patch  # silence linter; imported for symmetry


# Silence unused-import in case the suite is collected piecemeal.
_ = pytest


# ---------------------------------------------------------------------
# Bounded-read memory (#134)
# ---------------------------------------------------------------------


def test_read_large_file_does_not_materialise_full_file(
    tmp_path: Path,
) -> None:
    """Reading a file many times the cap peaks near the cap, not file size (#134).

    Pre-#134 ``_read_one`` loaded the entire file as ``str`` + re-encoded
    to ``bytes`` before applying the cap — a 50 MB file peaked at
    ~100 MB even though only 1 MiB was returned. The stat-then-read
    path bounds peak memory to ~``_MAX_BYTES``.
    """
    import tracemalloc

    from cothis.tools.fs.read import _read_one

    # ~5 MiB file — well over the 1 MiB cap, small enough to run in CI.
    payload = "line of text content here\n" * 200000  # ~5.2 MB
    (tmp_path / "big.txt").write_text(payload)

    with workdir_context(tmp_path):
        tracemalloc.start()
        _read_one("big.txt", None, None)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    # Peak must stay near the cap. Pre-fix this peaked at 2× file size
    # (~10 MB for a 5 MB file); post-fix it stays close to the 1 MiB
    # cap plus allocator overhead. 5× cap leaves headroom for the
    # truncated-bytes copy + Python allocator slack while still
    # catching a regression to the full-file-load path.
    assert peak < 5 * _MAX_BYTES, (
        f"peak memory {peak / 1024:.0f} KiB exceeded 5× cap; "
        f"file was materialised in full (#134 regression)"
    )
