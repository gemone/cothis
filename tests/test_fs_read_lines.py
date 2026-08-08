"""Tests for ``fs.read`` line-cap + streaming.

The line-range path (files ≤ ``_MAX_BYTES``) now:
- caps output at ``_MAX_LINES`` so a ≤1 MiB file with tens of thousands of
  short lines can't dump them all into one read;
- streams the file and stops at the requested window / the cap, instead of
  ``read_text().splitlines()`` on the whole file;
- emits an actionable continuation notice naming the resume ``start_line``
  (and the exact remaining count when ``end_line`` was given).

The byte-cap fast path (``size > _MAX_BYTES``) and its
``"… (truncated, N more bytes)"`` notice are untouched — see
``test_fs_read_truncation.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cothis.tools.fs._hygiene import _MAX_BYTES, _MAX_LINES, workdir_context
from cothis.tools.fs.read import _read_one, read

if TYPE_CHECKING:
    from pathlib import Path


def _write_lines(path: Path, n: int, *, prefix: str = "L") -> None:
    """Write *n* numbered lines (``L1`` … ``Ln``), each under ~6 bytes."""
    path.write_text(
        "".join(f"{prefix}{i}\n" for i in range(1, n + 1)), encoding="utf-8"
    )


def test_line_cap_fires_on_many_short_lines(tmp_path: Path) -> None:
    """A ≤1 MiB file with > _MAX_LINES lines caps at _MAX_LINES + a notice.

    Previously this emitted all 5000 lines into one read. The byte cap never
    fired (the file is ~25 KB); only the line cap bounds it.
    """
    f = tmp_path / "many.txt"
    _write_lines(f, 5000)  # ~25 KB, well under _MAX_BYTES
    assert f.stat().st_size < _MAX_BYTES

    with workdir_context(tmp_path):
        out = _read_one("many.txt", None, None)

    lines = out.splitlines()
    # Exactly _MAX_LINES numbered lines, then one continuation-notice line.
    assert len(lines) == _MAX_LINES + 1
    # Last numbered line is the cap; the notice follows.
    assert lines[_MAX_LINES - 1] == f"{_MAX_LINES}\tL{_MAX_LINES}"
    assert "more lines in file" in lines[_MAX_LINES]
    assert f"start_line={_MAX_LINES + 1}" in lines[_MAX_LINES]


def test_windowed_read_does_not_materialise_tail(tmp_path: Path) -> None:
    """A small window on a large file returns only that window, no notice."""
    f = tmp_path / "big.txt"
    _write_lines(f, 5000)

    with workdir_context(tmp_path):
        out = _read_one("big.txt", 5, 10)

    lines = out.splitlines()
    # Lines 5–10 only — proves the reader stopped at end_line, not EOF.
    # Width is set by the last shown line (10 → 2 digits), so single-digit
    # numbers are right-padded with a space (unchanged behaviour).
    assert lines == [" 5\tL5", " 6\tL6", " 7\tL7", " 8\tL8", " 9\tL9", "10\tL10"]
    assert "more lines" not in out


def test_window_larger_than_cap_emits_exact_remaining(tmp_path: Path) -> None:
    """A window wider than the cap emits the cap + the in-range remainder."""
    f = tmp_path / "wide.txt"
    _write_lines(f, 5000)

    with workdir_context(tmp_path):
        out = _read_one("wide.txt", 1, 5000)

    lines = out.splitlines()
    assert len(lines) == _MAX_LINES + 1
    # The notice reports the exact lines left in the requested window.
    assert "more line(s) in range" in lines[_MAX_LINES]
    assert f"{5000 - _MAX_LINES} more line(s)" in lines[_MAX_LINES]
    assert f"start_line={_MAX_LINES + 1}" in lines[_MAX_LINES]


def test_continuation_offset_resumes_correctly(tmp_path: Path) -> None:
    """Reading from the notice's ``start_line`` continues the numbering."""
    f = tmp_path / "cont.txt"
    _write_lines(f, 4500)

    with workdir_context(tmp_path):
        out = _read_one("cont.txt", _MAX_LINES + 1, None)

    lines = out.splitlines()
    # Resumes at _MAX_LINES+1; half of the remaining 2500 lines fit under
    # the cap, then a second notice for the rest.
    assert lines[0] == f"{_MAX_LINES + 1}\tL{_MAX_LINES + 1}"
    assert len(lines) == _MAX_LINES + 1
    assert f"start_line={2 * _MAX_LINES + 1}" in lines[_MAX_LINES]


def test_beyond_eof_error_preserved(tmp_path: Path) -> None:
    """The beyond-EOF error keeps its message + true line count."""
    f = tmp_path / "small.txt"
    _write_lines(f, 10)

    with workdir_context(tmp_path):
        out = _read_one("small.txt", 100, None)

    assert out == "Error: start_line 100 is beyond EOF (file has 10 lines)"


def test_short_file_numbered_output_unchanged(tmp_path: Path) -> None:
    """Small files still produce 1-based numbered output with no notice."""
    f = tmp_path / "tiny.txt"
    f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    with workdir_context(tmp_path):
        out = _read_one("tiny.txt", None, None)

    assert out.splitlines() == ["1\talpha", "2\tbeta", "3\tgamma"]
    assert "more lines" not in out


def test_public_read_tool_caps_and_continues(tmp_path: Path) -> None:
    """The ``read`` tool surface (not just ``_read_one``) honours the cap."""
    f = tmp_path / "tool.txt"
    _write_lines(f, 5000)

    with workdir_context(tmp_path):
        out = read(path="tool.txt")

    assert f"{_MAX_LINES}\tL{_MAX_LINES}" in out
    assert f"L{_MAX_LINES + 1}\t" not in out  # line past the cap absent
    assert f"start_line={_MAX_LINES + 1}" in out


def test_inverted_range_yields_empty_not_error(tmp_path: Path) -> None:
    """``end_line`` before ``start_line`` returns "" (unchanged behaviour)."""
    f = tmp_path / "inv.txt"
    _write_lines(f, 10)

    with workdir_context(tmp_path):
        out = _read_one("inv.txt", 8, 5)

    assert out == ""
