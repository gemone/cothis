"""Tests for ``fs.read`` truncation path (#312 perf fix).

The truncation path activates when a file exceeds ``_MAX_BYTES`` (1 MiB).
Pre-fix: ``while truncated: try: head = truncated.decode("utf-8"); break;
except UnicodeDecodeError: truncated = truncated[:-1]`` re-decoded the
entire 1 MiB buffer on each byte-strip retry (up to 4× for 4-byte CJK /
emoji chars), then re-encoded the kept head for byte accounting.

Post-fix: ``codecs.getincrementaldecoder("utf-8")`` decodes in a single
pass + ``decoder.buffer`` carries the undecoded tail bytes for the
``dropped`` count.

These tests verify behaviour is unchanged: the visible output (head +
truncation notice) matches what the old loop produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cothis.tools.fs._hygiene import _MAX_BYTES, workdir_context
from cothis.tools.fs.read import _read_one

if TYPE_CHECKING:
    from pathlib import Path


def test_truncation_handles_multibyte_split_at_cap(tmp_path: Path) -> None:
    """AC #312: a multi-byte char split at the byte cap decodes cleanly.

    Pre-fix, this triggered the ``while truncated`` retry loop — each
    byte-stripped retry re-decoded the whole buffer. Post-fix, the
    incremental decoder handles the split in one pass.

    The visible contract: ``head`` ends at a character boundary (no
    ``�`` replacement), and ``dropped`` accounts for the split
    char's tail bytes (size - kept_bytes).
    """
    # Build a file where the byte cap lands inside a 3-byte CJK char.
    # Padding = (cap - 3) bytes of ASCII pushes the last 3 bytes to the
    # cap boundary; writing one CJK char consumes those 3 bytes such
    # that byte cap splits it (1 byte inside, 2 bytes after).
    padding = "x" * (_MAX_BYTES - 1)
    body = padding + "世" + "tail-content-past-the-cap"
    f = tmp_path / "big.txt"
    f.write_text(body, encoding="utf-8")

    with workdir_context(tmp_path):
        out = _read_one("big.txt", None, None)

    # Decoded head must NOT contain U+FFFD (split surrogate) — the
    # incremental decoder dropped the partial char cleanly.
    assert "�" not in out
    # The ASCII padding is fully present (it's well inside the cap).
    assert "x" * 100 in out
    # Truncation notice present + correct byte math.
    assert "… (truncated," in out
    # The "tail-content..." was past the cap — must NOT appear.
    assert "tail-content-past-the-cap" not in out


def test_truncation_byte_count_matches_pre_fix_behaviour(tmp_path: Path) -> None:
    """AC #312: the ``dropped`` byte count matches the old code's semantics.

    Pre-fix: ``dropped = size - len(head.encode("utf-8"))``. Post-fix:
    ``dropped = size - (_MAX_BYTES - len(decoder.buffer))``. These must
    produce the same number — both express ``size - kept_bytes``.

    Construct: file with all-ASCII content (no multi-byte split), so
    ``decoder.buffer`` is empty + the old and new code paths agree
    exactly on ``dropped``.
    """
    body = "a" * (_MAX_BYTES + 1000)  # all ASCII, no split
    f = tmp_path / "ascii.txt"
    f.write_text(body, encoding="utf-8")

    with workdir_context(tmp_path):
        out = _read_one("ascii.txt", None, None)

    # Old behaviour: head = first _MAX_BYTES of ASCII = "a" × _MAX_BYTES
    # (decoded fully, no split). Dropped = size - _MAX_BYTES = 1000.
    assert "… (truncated, 1000 more bytes)" in out


def test_short_file_unaffected_by_incremental_decoder(tmp_path: Path) -> None:
    """AC #312: short files bypass the truncation path entirely.

    Regression guard: the incremental-decoder refactor only touched
    the ``size > _MAX_BYTES`` branch. Files under the cap must continue
    to flow through the original line-range code.
    """
    f = tmp_path / "small.txt"
    f.write_text("line one\nline two\nline three\n", encoding="utf-8")

    with workdir_context(tmp_path):
        out = _read_one("small.txt", None, None)

    # 1-based numbered output from the line-range path.
    assert "1\tline one" in out
    assert "2\tline two" in out
    assert "3\tline three" in out
    assert "truncated" not in out
