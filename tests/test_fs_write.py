"""Tests for ``cothis.tools.fs.write`` — codex apply_patch writer.

`fs.write(content)` takes a single codex ``apply_patch`` document and
commits Add / Update / Delete ops to disk via the deep module in
``tools/fs/patch.py``. Pure signature change + migration; cwd boundary
lands in slice #5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cothis.tools.fs._hygiene import workdir_context
from cothis.tools.fs.patch import PatchError
from cothis.tools.fs.write import write


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
    assert "added new.txt" in result


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
    assert "deleted stale.txt" in result


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
    assert "added n.txt" in result
    assert "updated u.txt" in result
    assert "deleted d.txt" in result


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


# ---------------------------------------------------------------------
# atomic commit + rollback (slice #6 / #53)
# ---------------------------------------------------------------------


def test_write_returns_file_list_with_verbs(tmp_path: Path) -> None:
    """Return value lists each affected file with its verb — not just counts."""
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
    # Each file listed with its verb.
    assert "added n.txt" in result
    assert "updated u.txt" in result
    assert "deleted d.txt" in result
    # No diff preview (information duplication with the patch already in model context).
    assert "+new" not in result
    assert "-a" not in result


def test_write_rollback_on_oserror_mid_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OSError on the Nth target's write rolls back prior commits in reverse."""
    import cothis.tools.fs.write as write_mod

    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    patch = """\
*** Begin Patch
*** Update File: a.txt
@@
-alpha
+ALPHA
*** Update File: b.txt
@@
-beta
+BETA
*** Add File: c.txt
+gamma
*** End Patch
"""
    # Inject OSError on the second write_text call (b.txt's commit).
    real_write_text = Path.write_text
    call_count = {"n": 0}

    def failing_write_text(self: Path, data: str, **kw: str | None) -> int:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("simulated disk full")
        return real_write_text(self, data, **kw)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    with workdir_context(tmp_path):
        with pytest.raises(OSError, match="disk full"):
            write(content=patch)

    # Rollback: a.txt's content restored, c.txt (added then rolled back) gone.
    assert (tmp_path / "a.txt").read_text() == "alpha\n"
    assert (tmp_path / "b.txt").read_text() == "beta\n"
    assert not (tmp_path / "c.txt").exists()


def test_write_rollback_failure_logs_and_reraises_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """If rollback itself fails on a target, logger.error records it; the
    *primary* OSError is what propagates (rollback error doesn't mask it)."""
    import cothis.tools.fs.write as write_mod

    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    patch = """\
*** Begin Patch
*** Update File: a.txt
@@
-alpha
+ALPHA
*** Add File: b.txt
+beta
*** End Patch
"""
    real_write_text = Path.write_text
    real_write_bytes = Path.write_bytes
    primary_raised = {"n": 0}

    def write_then_fail(self: Path, data: str, **kw: str | None) -> int:
        primary_raised["n"] += 1
        # b.txt commit (second call) raises primary OSError.
        if primary_raised["n"] == 2:
            raise OSError("primary disk error")
        return real_write_text(self, data, **kw)

    def rollback_fail(self: Path, data: bytes) -> int:
        # Rollback path: always fail.
        raise OSError("rollback failure")

    monkeypatch.setattr(Path, "write_text", write_then_fail)
    monkeypatch.setattr(Path, "write_bytes", rollback_fail)

    with workdir_context(tmp_path):
        with pytest.raises(OSError, match="primary disk error"):
            with caplog.at_level("ERROR", logger=write_mod.__name__):
                write(content=patch)

    # Rollback failure logged.
    rollback_errors = [
        r for r in caplog.records
        if "rollback" in r.getMessage().lower()
    ]
    assert rollback_errors, "rollback failure must be logged at ERROR"
