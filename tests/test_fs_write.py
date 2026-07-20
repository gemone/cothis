"""Tests for ``cothis.tools.fs.write`` — codex apply_patch writer.

`fs.write(content)` takes a single codex ``apply_patch`` document and
commits Add / Update / Delete ops to disk via the deep module in
``tools/fs/patch.py``. Pure signature change + migration; cwd boundary
lands in slice #5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from cothis.tools.fs._hygiene import workdir_context
from cothis.tools.fs.patch import PatchError
from cothis.tools.fs.write import write

if TYPE_CHECKING:
    from pathlib import Path


def test_write_add_creates_new_file(tmp_path: Path) -> None:
    """``*** Add File:`` creates the file (mkdir parents, for now)."""
    patch = """\
*** Begin Patch
*** Add File: new.txt
+hello world
*** End Patch
"""
    with workdir_context(tmp_path):
        result = write(content=patch)
    assert (tmp_path / "new.txt").read_text() == "hello world\n"
    assert "added 1" in result


def test_write_add_in_nested_subdir_mkdirs_parents(tmp_path: Path) -> None:
    """Pre-#52 behavior created intermediate dirs; #52 removed that —
    Add File into a missing parent dir is now rejected. Test name kept
    to minimise diff churn; body asserts the new boundary."""
    patch = """\
*** Begin Patch
*** Add File: a/b/c.txt
+deep
*** End Patch
"""
    with workdir_context(tmp_path):
        with pytest.raises(PatchError, match="parent"):
            write(content=patch)
    assert not (tmp_path / "a").exists()


def test_write_update_splices_pre_image(tmp_path: Path) -> None:
    """``*** Update File:`` replaces the pre-image block; file untouched
    on pre-image miss (PatchError raised before commit)."""
    (tmp_path / "app.py").write_text('def greet():\n    old = "hi"\n', encoding="utf-8")
    patch = """\
*** Begin Patch
*** Update File: app.py
@@ def greet():
-    old = "hi"
+    new = "hello"
*** End Patch
"""
    with workdir_context(tmp_path):
        write(content=patch)
    assert (tmp_path / "app.py").read_text() == 'def greet():\n    new = "hello"\n'


def test_write_delete_removes_file(tmp_path: Path) -> None:
    """``*** Delete File:`` removes the file from disk."""
    (tmp_path / "stale.txt").write_text("bye\n", encoding="utf-8")
    patch = """\
*** Begin Patch
*** Delete File: stale.txt
*** End Patch
"""
    with workdir_context(tmp_path):
        result = write(content=patch)
    assert not (tmp_path / "stale.txt").exists()
    assert "deleted 1" in result


def test_write_add_on_existing_raises_patch_error(tmp_path: Path) -> None:
    """``Add File`` on an existing path is rejected; nothing written."""
    (tmp_path / "dup.txt").write_text("exists\n", encoding="utf-8")
    patch = """\
*** Begin Patch
*** Add File: dup.txt
+overwrite
*** End Patch
"""
    with workdir_context(tmp_path):
        with pytest.raises(PatchError, match="exist|already"):
            write(content=patch)
    # File untouched.
    assert (tmp_path / "dup.txt").read_text() == "exists\n"


def test_write_two_ops_on_one_path_raises(tmp_path: Path) -> None:
    """Two ops on the same path in one patch → PatchError; nothing written."""
    (tmp_path / "x.txt").write_text("alpha\n", encoding="utf-8")
    patch = """\
*** Begin Patch
*** Update File: x.txt
@@
-alpha
+beta
*** Delete File: x.txt
*** End Patch
"""
    with workdir_context(tmp_path):
        with pytest.raises(PatchError, match="one op|more than"):
            write(content=patch)
    # File untouched.
    assert (tmp_path / "x.txt").read_text() == "alpha\n"


def test_write_malformed_patch_raises_before_disk_write(tmp_path: Path) -> None:
    """Malformed patch (missing End Patch) → PatchError; no file touched."""
    patch = "*** Begin Patch\n*** Add File: bad.txt\n+content"  # no End Patch
    with workdir_context(tmp_path):
        with pytest.raises(PatchError):
            write(content=patch)
    assert not (tmp_path / "bad.txt").exists()


def test_write_multi_file_atomic_on_error(tmp_path: Path) -> None:
    """A patch that fails mid-way (pre-image miss on second file) leaves
    the first file's intended change NOT applied — in-memory error path
    rolls back everything."""
    (tmp_path / "keep.txt").write_text("orig\n", encoding="utf-8")
    # Second op references a file that doesn't exist → Update fails.
    patch = """\
*** Begin Patch
*** Update File: keep.txt
@@
-orig
+modified
*** Update File: ghost.txt
@@
-nope
+yes
*** End Patch
"""
    with workdir_context(tmp_path):
        with pytest.raises(PatchError):
            write(content=patch)
    # keep.txt untouched (atomic rollback).
    assert (tmp_path / "keep.txt").read_text() == "orig\n"


def test_write_summary_counts_all_three_op_kinds(tmp_path: Path) -> None:
    """A patch touching all three op kinds returns a summary with counts."""
    (tmp_path / "u.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "d.txt").write_text("b\n", encoding="utf-8")
    patch = """\
*** Begin Patch
*** Add File: n.txt
+new
*** Update File: u.txt
@@
-a
+A
*** Delete File: d.txt
*** End Patch
"""
    with workdir_context(tmp_path):
        result = write(content=patch)
    assert "added 1" in result
    assert "updated 1" in result
    assert "deleted 1" in result


# ---------------------------------------------------------------------
# cwd boundary enforcement (slice #5 / #52)
# ---------------------------------------------------------------------


def test_write_rejects_absolute_path(tmp_path: Path) -> None:
    """Absolute path in patch → PatchError; nothing written."""
    patch = """\
*** Begin Patch
*** Add File: /etc/cothis-test-target
+evil
*** End Patch
"""
    with workdir_context(tmp_path):
        with pytest.raises(PatchError, match="absolute"):
            write(content=patch)


def test_write_rejects_parent_traversal(tmp_path: Path) -> None:
    """``..`` that escapes cwd → PatchError; nothing written."""
    (tmp_path / "real.txt").write_text("orig\n", encoding="utf-8")
    patch = """\
*** Begin Patch
*** Update File: ../secret
+x
*** End Patch
"""
    with workdir_context(tmp_path):
        with pytest.raises(PatchError, match="cwd|outside"):
            write(content=patch)


def test_write_rejects_symlink_escape(tmp_path: Path) -> None:
    """Symlink pointing outside cwd → PatchError; nothing written."""
    outside = tmp_path.parent / "outside-target"
    outside.write_text("outside\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(outside)

    patch = """\
*** Begin Patch
*** Update File: link
+x
*** End Patch
"""
    with workdir_context(tmp_path):
        with pytest.raises(PatchError):
            write(content=patch)


def test_write_add_missing_parent_dir_rejected(tmp_path: Path) -> None:
    """Add File targeting a path whose parent dir doesn't exist →
    PatchError naming the missing parent; no directory created."""
    patch = """\
*** Begin Patch
*** Add File: newdir/a.py
+x
*** End Patch
"""
    with workdir_context(tmp_path):
        with pytest.raises(PatchError, match="parent"):
            write(content=patch)
    # Parent not created.
    assert not (tmp_path / "newdir").exists()


def test_write_add_into_existing_subdir_ok(tmp_path: Path) -> None:
    """Add File into an existing subdir → creates file (no regression)."""
    (tmp_path / "existing").mkdir()
    patch = """\
*** Begin Patch
*** Add File: existing/a.py
+x
*** End Patch
"""
    with workdir_context(tmp_path):
        write(content=patch)
    assert (tmp_path / "existing" / "a.py").read_text() == "x\n"


def test_write_boundary_check_runs_before_any_disk_write(tmp_path: Path) -> None:
    """A boundary violation in a multi-op patch leaves NO files on disk —
    pre-flight check runs entirely before commit."""
    (tmp_path / "safe.txt").write_text("orig\n", encoding="utf-8")
    patch = """\
*** Begin Patch
*** Update File: safe.txt
@@
-orig
+modified
*** Add File: /etc/cothis-evil
+evil
*** End Patch
"""
    with workdir_context(tmp_path):
        with pytest.raises(PatchError):
            write(content=patch)
    # safe.txt untouched (pre-flight raised before commit).
    assert (tmp_path / "safe.txt").read_text() == "orig\n"
