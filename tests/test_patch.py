"""Tests for ``cothis.tools.fs.patch`` — v4 diff format parser + applier.

The patch deep module is pure: dict-in/dict-out, no filesystem access.
Tests cover the three op kinds (Add / Update / Delete), multi-hunk
Update, trailing-whitespace-tolerant matching, and error cases that
must surface file + line so the LLM can self-correct.
"""

from __future__ import annotations

import pytest

from cothis.tools.fs.patch import (
    AddFile,
    ApplyError,
    DeleteFile,
    ParseError,
    UpdateFile,
    apply_patch,
    parse_patch,
)

# ---------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------


def test_parse_empty_patch_returns_no_ops() -> None:
    ops = parse_patch("*** Begin Patch\n*** End Patch")
    assert ops == []


def test_parse_add_file_single_line() -> None:
    text = """\
*** Begin Patch
*** Add File: hello.txt
+hello world
*** End Patch
"""
    ops = parse_patch(text)
    assert len(ops) == 1
    assert isinstance(ops[0], AddFile)
    assert ops[0].path == "hello.txt"
    assert ops[0].content == "hello world\n"


def test_parse_add_file_multi_line() -> None:
    text = """\
*** Begin Patch
*** Add File: hello.txt
+line one
+line two
+line three
*** End Patch
"""
    ops = parse_patch(text)
    assert ops[0].content == "line one\nline two\nline three\n"


def test_parse_delete_file() -> None:
    text = """\
*** Begin Patch
*** Delete File: stale.txt
*** End Patch
"""
    ops = parse_patch(text)
    assert len(ops) == 1
    assert isinstance(ops[0], DeleteFile)
    assert ops[0].path == "stale.txt"


def test_parse_update_file_single_hunk() -> None:
    text = """\
*** Begin Patch
*** Update File: app.py
@@ def greet():
-old = "hi"
+new = "hello"
*** End Patch
"""
    ops = parse_patch(text)
    assert len(ops) == 1
    assert isinstance(ops[0], UpdateFile)
    assert ops[0].path == "app.py"
    assert len(ops[0].hunks) == 1
    hunk = ops[0].hunks[0]
    assert hunk.context == ["def greet():"]
    assert hunk.removes == ['old = "hi"']
    assert hunk.adds == ['new = "hello"']


def test_parse_update_file_multi_hunk() -> None:
    text = """\
*** Begin Patch
*** Update File: app.py
@@ def greet():
-old = "hi"
+new = "hello"
@@ def farewell():
-old = "bye"
+new = "goodbye"
*** End Patch
"""
    ops = parse_patch(text)
    assert len(ops[0].hunks) == 2


def test_parse_multiple_ops_in_one_patch() -> None:
    text = """\
*** Begin Patch
*** Add File: a.txt
+alpha
*** Update File: b.txt
@@ ctx
-beta
+gamma
*** Delete File: c.txt
*** End Patch
"""
    ops = parse_patch(text)
    assert len(ops) == 3
    assert isinstance(ops[0], AddFile)
    assert isinstance(ops[1], UpdateFile)
    assert isinstance(ops[2], DeleteFile)


def test_parse_rejects_missing_begin() -> None:
    with pytest.raises(ParseError) as exc:
        parse_patch("*** Add File: x.txt\n+content")
    assert "Begin Patch" in str(exc.value)


def test_parse_rejects_missing_end() -> None:
    with pytest.raises(ParseError) as exc:
        parse_patch("*** Begin Patch\n*** Add File: x.txt\n+content")
    assert "End Patch" in str(exc.value)


def test_parse_unknown_op_marker_raises_with_file_and_line() -> None:
    text = """\
*** Begin Patch
*** Add File: x.txt
+content
*** Rename File: y.txt -> z.txt
*** End Patch
"""
    with pytest.raises(ParseError) as exc:
        parse_patch(text)
    msg = str(exc.value)
    assert "Rename File" in msg or "unknown" in msg.lower()
    assert "line" in msg.lower()


# ---------------------------------------------------------------------
# applier
# ---------------------------------------------------------------------


def test_apply_add_file_creates_new_entry() -> None:
    ops = parse_patch("""\
*** Begin Patch
*** Add File: new.txt
+hello
*** End Patch
""")
    result = apply_patch({}, ops)
    assert result == {"new.txt": "hello\n"}


def test_apply_add_file_existing_raises_with_file_and_line() -> None:
    ops = parse_patch("""\
*** Begin Patch
*** Add File: exists.txt
+hello
*** End Patch
""")
    with pytest.raises(ApplyError) as exc:
        apply_patch({"exists.txt": "old\n"}, ops)
    msg = str(exc.value)
    assert "exists.txt" in msg
    assert "line" in msg.lower() or "Add File" in msg


def test_apply_delete_file_removes_entry() -> None:
    ops = parse_patch("""\
*** Begin Patch
*** Delete File: gone.txt
*** End Patch
""")
    result = apply_patch({"gone.txt": "x\n", "keep.txt": "y\n"}, ops)
    assert "gone.txt" not in result
    assert result["keep.txt"] == "y\n"


def test_apply_delete_missing_file_raises_with_file() -> None:
    ops = parse_patch("""\
*** Begin Patch
*** Delete File: missing.txt
*** End Patch
""")
    with pytest.raises(ApplyError) as exc:
        apply_patch({}, ops)
    assert "missing.txt" in str(exc.value)


def test_apply_update_replaces_pre_image() -> None:
    files = {"app.py": 'def greet():\n    old = "hi"\n    return old\n'}
    ops = parse_patch("""\
*** Begin Patch
*** Update File: app.py
@@ def greet():
-    old = "hi"
+    new = "hello"
*** End Patch
""")
    result = apply_patch(files, ops)
    assert result["app.py"] == 'def greet():\n    new = "hello"\n    return old\n'


def test_apply_update_multi_hunk_in_one_op() -> None:
    files = {"app.py": 'def greet():\n    a = "hi"\n\ndef farewell():\n    b = "bye"\n'}
    ops = parse_patch("""\
*** Begin Patch
*** Update File: app.py
@@ def greet():
-    a = "hi"
+    a = "hello"
@@ def farewell():
-    b = "bye"
+    b = "goodbye"
*** End Patch
""")
    result = apply_patch(files, ops)
    assert result["app.py"] == 'def greet():\n    a = "hello"\n\ndef farewell():\n    b = "goodbye"\n'


def test_apply_update_pre_image_missing_raises_with_file_and_line() -> None:
    files = {"app.py": 'def greet():\n    new = "hello"\n'}
    ops = parse_patch("""\
*** Begin Patch
*** Update File: app.py
@@ def greet():
-old = "hi"
+new = "hello"
*** End Patch
""")
    with pytest.raises(ApplyError) as exc:
        apply_patch(files, ops)
    msg = str(exc.value)
    assert "app.py" in msg
    assert "line" in msg.lower() or "context" in msg.lower()


def test_apply_update_trailing_whitespace_tolerant() -> None:
    """Pre-image with trailing whitespace still matches content with none
    (and vice versa). Editors strip trailing ws; patches shouldn't fail
    purely on that."""
    files = {"app.py": "x = 1\n"}  # no trailing ws
    ops = parse_patch("""\
*** Begin Patch
*** Update File: app.py
@@
-x = 1
+x = 2
*** End Patch
""")
    result = apply_patch(files, ops)
    assert result["app.py"] == "x = 2\n"


def test_apply_mixed_ops_in_one_patch() -> None:
    files = {"keep.txt": "alpha\n", "update.txt": "beta\n"}
    ops = parse_patch("""\
*** Begin Patch
*** Add File: new.txt
+gamma
*** Update File: update.txt
@@
-beta
+delta
*** Delete File: keep.txt
*** End Patch
""")
    result = apply_patch(files, ops)
    assert result == {"update.txt": "delta\n", "new.txt": "gamma\n"}


def test_apply_is_pure_does_not_mutate_input() -> None:
    """The applier must not mutate the input dict — callers may reuse it
    (snapshot for rollback)."""
    files = {"x.txt": "alpha\n"}
    ops = parse_patch("""\
*** Begin Patch
*** Update File: x.txt
@@
-alpha
+beta
*** End Patch
""")
    apply_patch(files, ops)
    assert files == {"x.txt": "alpha\n"}, "input dict must be unchanged"
