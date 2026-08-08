"""Tests for the ``fs.edit`` unique-anchor string-replace tool.

Unique-anchor string search-and-replace edit. ``old_string`` must be
unique unless ``replace_all=True``. Match is byte-exact (substring
search via ``str.count`` / ``str.replace``), including line endings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cothis.tools.fs._hygiene import workdir_context

if TYPE_CHECKING:
    from pathlib import Path


def _make_file(tmp_path: Path, name: str, content: str) -> Path:
    """Create a file with the given text content for testing.

    Writes via ``write_bytes`` (binary) rather than ``write_text`` so the
    file's line endings are exactly the LF bytes in ``content`` on EVERY
    platform — ``write_text`` is text-mode and translates ``\\n`` to
    ``\\r\\n`` on Windows, which would make any multi-line ``old_string``
    fail the byte-exact substring match. ``fs.edit`` itself is line-ending-
    preserving; only the test fixture needs to be platform-stable.
    """
    f = tmp_path / name
    f.write_bytes(content.encode("utf-8"))
    return f


# ---------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_edit_unique_single_match(tmp_path: Path) -> None:
    """One occurrence of old_string → replaced; return message carries count + path."""
    from cothis.tools.fs.edit import _edit

    _make_file(tmp_path, "f.py", "alpha\nbeta\ngamma\n")
    with workdir_context(tmp_path):
        result = await _edit(path="f.py", old_string="beta", new_string="BETA")
    assert "updated" in result.lower()
    assert "1 replacement" in result, result
    assert "f.py" in result
    assert "file now" in result.lower()
    assert (tmp_path / "f.py").read_text() == "alpha\nBETA\ngamma\n"


@pytest.mark.asyncio
async def test_fs_edit_replace_all_multiple(tmp_path: Path) -> None:
    """replace_all=True → every occurrence replaced; return shows count."""
    from cothis.tools.fs.edit import _edit

    _make_file(tmp_path, "f.py", "x x x\n")
    with workdir_context(tmp_path):
        result = await _edit(
            path="f.py", old_string="x", new_string="Y", replace_all=True,
        )
    assert "3 replacements" in result, result
    assert (tmp_path / "f.py").read_text() == "Y Y Y\n"


@pytest.mark.asyncio
async def test_fs_edit_multiline_block_replace(tmp_path: Path) -> None:
    """old_string spanning multiple lines → whole block swapped in one replacement."""
    from cothis.tools.fs.edit import _edit

    original = "header\nold line 1\nold line 2\nold line 3\nfooter\n"
    _make_file(tmp_path, "f.py", original)
    with workdir_context(tmp_path):
        result = await _edit(
            path="f.py",
            old_string="old line 1\nold line 2\nold line 3\n",
            new_string="brand\nnew\nblock\n",
        )
    assert "1 replacement" in result
    assert (tmp_path / "f.py").read_text() == "header\nbrand\nnew\nblock\nfooter\n"


@pytest.mark.asyncio
async def test_fs_edit_empty_new_string_deletes(tmp_path: Path) -> None:
    """Empty new_string is a valid deletion (only old_string is gated)."""
    from cothis.tools.fs.edit import _edit

    _make_file(tmp_path, "f.py", "keep this\nremove me\nkeep this too\n")
    with workdir_context(tmp_path):
        result = await _edit(
            path="f.py", old_string="remove me\n", new_string="",
        )
    assert "1 replacement" in result
    assert (tmp_path / "f.py").read_text() == "keep this\nkeep this too\n"


# ---------------------------------------------------------------------
# Errors — match semantics
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_edit_no_match_error(tmp_path: Path) -> None:
    """old_string absent → error regardless of replace_all (no silent no-op)."""
    from cothis.tools.fs.edit import _edit

    _make_file(tmp_path, "f.py", "alpha\nbeta\n")
    original = (tmp_path / "f.py").read_bytes()
    with workdir_context(tmp_path):
        result = await _edit(path="f.py", old_string="zeta", new_string="Z")
    assert "old_string not found" in result
    assert "f.py" in result
    # File untouched.
    assert (tmp_path / "f.py").read_bytes() == original


@pytest.mark.asyncio
async def test_fs_edit_no_match_error_even_with_replace_all(tmp_path: Path) -> None:
    """replace_all=True with 0 matches still errors (not a license to no-op)."""
    from cothis.tools.fs.edit import _edit

    _make_file(tmp_path, "f.py", "alpha\nbeta\n")
    with workdir_context(tmp_path):
        result = await _edit(
            path="f.py", old_string="zeta", new_string="Z", replace_all=True,
        )
    assert "old_string not found" in result


@pytest.mark.asyncio
async def test_fs_edit_ambiguous_match_error(tmp_path: Path) -> None:
    """n>1 with replace_all=False → 'not unique' error listing the count."""
    from cothis.tools.fs.edit import _edit

    _make_file(tmp_path, "f.py", "dup\nunique\ndup\n")
    original = (tmp_path / "f.py").read_bytes()
    with workdir_context(tmp_path):
        result = await _edit(path="f.py", old_string="dup", new_string="X")
    assert "not unique" in result
    assert "2 matches" in result, result
    assert "replace_all=True" in result
    # File byte-unchanged.
    assert (tmp_path / "f.py").read_bytes() == original


@pytest.mark.asyncio
async def test_fs_edit_empty_old_string_rejected(tmp_path: Path) -> None:
    """Empty old_string short-circuits before count (str.count('') = len+1)."""
    from cothis.tools.fs.edit import _edit

    _make_file(tmp_path, "f.py", "alpha\nbeta\n")
    original = (tmp_path / "f.py").read_bytes()
    with workdir_context(tmp_path):
        result = await _edit(path="f.py", old_string="", new_string="X")
    assert "must not be empty" in result.lower()
    assert (tmp_path / "f.py").read_bytes() == original


# ---------------------------------------------------------------------
# CRLF / LF preservation (#371)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_edit_preserves_crlf_on_untouched_lines(tmp_path: Path) -> None:
    """#371: replacing one substring of a CRLF file keeps CRLF on every line."""
    from cothis.tools.fs.edit import _edit

    f = tmp_path / "crlf.txt"
    f.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    with workdir_context(tmp_path):
        result = await _edit(path="crlf.txt", old_string="beta", new_string="NEW")
    assert "updated" in result.lower()
    assert f.read_bytes() == b"alpha\r\nNEW\r\ngamma\r\n"


@pytest.mark.asyncio
async def test_fs_edit_lf_file_stays_lf(tmp_path: Path) -> None:
    """#371: LF files stay LF (no CRLF regression)."""
    from cothis.tools.fs.edit import _edit

    f = tmp_path / "lf.txt"
    f.write_bytes(b"alpha\nbeta\ngamma\n")
    with workdir_context(tmp_path):
        await _edit(path="lf.txt", old_string="beta", new_string="NEW")
    assert f.read_bytes() == b"alpha\nNEW\ngamma\n"


@pytest.mark.asyncio
async def test_fs_edit_crlf_anchor_must_match_byte_exact(tmp_path: Path) -> None:
    """Byte-exact match: passing LF anchor to a CRLF file → no-match error."""
    from cothis.tools.fs.edit import _edit

    f = tmp_path / "crlf.txt"
    f.write_bytes(b"alpha\r\nbeta\r\n")
    with workdir_context(tmp_path):
        result = await _edit(
            path="crlf.txt", old_string="alpha\nbeta", new_string="X",
        )
    assert "old_string not found" in result
    assert f.read_bytes() == b"alpha\r\nbeta\r\n"


# ---------------------------------------------------------------------
# Errors — path / size
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_edit_file_not_found(tmp_path: Path) -> None:
    """Missing file → error."""
    from cothis.tools.fs.edit import _edit

    with workdir_context(tmp_path):
        result = await _edit(
            path="missing.py", old_string="x", new_string="y",
        )
    assert "file not found" in result.lower()
    assert "missing.py" in result


@pytest.mark.asyncio
async def test_fs_edit_path_escape_rejected(tmp_path: Path) -> None:
    """Path escape → boundary error."""
    from cothis.tools.fs.edit import _edit

    with workdir_context(tmp_path):
        result = await _edit(
            path="../../../etc/passwd", old_string="x", new_string="y",
        )
    assert "error" in result.lower()
    assert "outside cwd" in result.lower() or "resolves outside" in result.lower()


@pytest.mark.asyncio
async def test_fs_edit_absolute_path_rejected(tmp_path: Path) -> None:
    """Absolute path → boundary error."""
    from cothis.tools.fs.edit import _edit

    with workdir_context(tmp_path):
        result = await _edit(
            path="/etc/passwd", old_string="x", new_string="y",
        )
    assert "error" in result.lower()
    assert "absolute" in result.lower()


@pytest.mark.asyncio
async def test_fs_edit_refuses_oversized_file(tmp_path: Path) -> None:
    """#419: stat-first guard refuses files > _MAX_BYTES; file left unchanged."""
    from cothis.tools.fs._hygiene import _MAX_BYTES
    from cothis.tools.fs.edit import _edit

    f = tmp_path / "big.txt"
    payload = b"x" * (_MAX_BYTES + 1)
    f.write_bytes(payload)
    with workdir_context(tmp_path):
        result = await _edit(path="big.txt", old_string="x", new_string="y")
    assert "too large" in result.lower(), result
    # File NOT modified (no truncation, no rewrite).
    assert f.read_bytes() == payload


# ---------------------------------------------------------------------
# Atomic-write invariants (#351)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_edit_leaves_no_temp_files_after_write(tmp_path: Path) -> None:
    """#351: atomic write leaves no ``*.tmp`` orphan after success."""
    from cothis.tools.fs.edit import _edit

    _make_file(tmp_path, "target.py", "alpha\nbeta\n")
    dir_before = set(p.name for p in tmp_path.iterdir())

    with workdir_context(tmp_path):
        await _edit(path="target.py", old_string="beta", new_string="BETA")

    dir_after = set(p.name for p in tmp_path.iterdir())
    assert dir_after == dir_before, (
        f"directory should be unchanged after atomic write; "
        f"new files: {dir_after - dir_before}"
    )
    assert "BETA" in (tmp_path / "target.py").read_text()


@pytest.mark.asyncio
async def test_fs_edit_atomic_write_cleans_up_temp_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#351: if the rename crashes, temp is cleaned up + original content survives."""
    import os as os_mod

    from cothis.tools.fs.edit import _edit

    f = _make_file(tmp_path, "target.py", "alpha\nbeta\n")
    original_content = f.read_bytes()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash during os.replace")

    monkeypatch.setattr(os_mod, "replace", boom)

    with workdir_context(tmp_path):
        with pytest.raises(RuntimeError, match="simulated crash"):
            await _edit(path="target.py", old_string="beta", new_string="BETA")

    # Original survived.
    assert f.read_bytes() == original_content
    # No .tmp orphan.
    files = list(tmp_path.iterdir())
    assert all(not p.name.endswith(".tmp") for p in files), (
        f"temp file should be cleaned up after crash; found: "
        f"{[p.name for p in files if p.name.endswith('.tmp')]}"
    )


@pytest.mark.asyncio
async def test_fs_edit_preserves_file_permissions(tmp_path: Path) -> None:
    """#351 follow-up: atomic write preserves the original file's permissions."""
    import os
    import stat as stat_mod

    from cothis.tools.fs.edit import _edit

    f = _make_file(tmp_path, "script.sh", "alpha\nbeta\n")
    os.chmod(f, 0o755)
    # Windows ignores the execute bit in chmod — capture the ACTUAL mode
    # the OS accepted, then verify edit preserves it.
    original_mode = stat_mod.S_IMODE(f.stat().st_mode)

    with workdir_context(tmp_path):
        await _edit(path="script.sh", old_string="beta", new_string="BETA")

    mode = stat_mod.S_IMODE(f.stat().st_mode)
    assert mode == original_mode, (
        f"permissions should be preserved (was 0o{original_mode:o}); "
        f"got 0o{mode:o}"
    )


# ---------------------------------------------------------------------
# Schema / description audit smoke
# ---------------------------------------------------------------------


def test_fs_edit_schema_shape() -> None:
    """``schema_for(_edit)`` exposes the right name + typed args."""
    from cothis.tools import schema_for
    from cothis.tools.fs.edit import _edit

    schema = schema_for(_edit)
    assert schema["name"] == "fs.edit"
    props = schema["input_schema"]["properties"]
    assert set(props) == {"path", "old_string", "new_string", "replace_all"}
    assert props["path"]["type"] == "string"
    assert props["old_string"]["type"] == "string"
    assert props["new_string"]["type"] == "string"
    assert props["replace_all"]["type"] == "boolean"
    assert schema["input_schema"]["required"] == ["path", "old_string", "new_string"]


def test_fs_edit_description_meets_audit_floor() -> None:
    """Description satisfies the 4-point audit floor inline."""
    from cothis.tools import schema_for
    from cothis.tools.fs.edit import _edit

    desc = schema_for(_edit)["description"]
    assert len(desc) >= 120
    assert "edit(" in desc
    assert ("Example" in desc) or ("::" in desc)
    assert ("→" in desc) or ("Returns" in desc) or ("return" in desc.lower())
