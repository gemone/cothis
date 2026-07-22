"""Tests for ``fs.write`` schema description carrying the apply_patch format (#190).

The model needs at least one concrete example of the codex patch
format in the tool's schema description — without it, non-Claude/GPT-4
providers guess wrong 3-4 times before succeeding. The griffe parser
extracts the first docstring paragraph as the description, so the
format tokens must appear there.
"""

from __future__ import annotations

import pytest

from cothis.tools import schema_for
from cothis.tools.fs.write import write


def _write_schema() -> dict:
    """Build the fs.write tool schema (the LLM-facing description)."""
    return schema_for(write)


def test_fs_write_description_mentions_begin_patch() -> None:
    """``*** Begin Patch`` appears in the schema description."""
    schema = _write_schema()
    desc = schema.get("description", "")
    assert "*** Begin Patch" in desc


def test_fs_write_description_mentions_end_patch() -> None:
    """``*** End Patch`` appears in the schema description."""
    schema = _write_schema()
    desc = schema.get("description", "")
    assert "*** End Patch" in desc


def test_fs_write_description_mentions_add_file_op() -> None:
    """``*** Add File:`` appears (the most common first-use case)."""
    schema = _write_schema()
    desc = schema.get("description", "")
    assert "*** Add File:" in desc


def test_fs_write_description_mentions_update_file_op() -> None:
    """``*** Update File:`` appears (second common case)."""
    schema = _write_schema()
    desc = schema.get("description", "")
    assert "*** Update File:" in desc


def test_fs_write_description_mentions_delete_file_op() -> None:
    """``*** Delete File:`` appears (third op type)."""
    schema = _write_schema()
    desc = schema.get("description", "")
    assert "*** Delete File:" in desc


def test_fs_write_arg_description_carries_format() -> None:
    """Per-arg description for ``content`` also mentions the format tokens."""
    schema = _write_schema()
    content_arg = schema.get("input_schema", {}).get("properties", {}).get("content", {})
    arg_desc = content_arg.get("description", "")
    # At least the begin/end markers should be in the arg description too.
    assert "*** Begin Patch" in arg_desc or "*** End Patch" in arg_desc


def test_fs_write_description_has_concrete_example_block() -> None:
    """The description carries a full example patch block.

    The model needs to see all four lines (Begin, Add File, +content,
    End) to reproduce the format. Spot-check by extracting the
    description and verifying it contains the four-line pattern as
    substrings.
    """
    schema = _write_schema()
    desc = schema.get("description", "")
    # Look for the canonical "add a new file" example block.
    assert "*** Begin Patch" in desc
    assert "*** Add File: " in desc  # space + path after
    assert "+hello" in desc or "+" in desc  # the + content marker
    assert "*** End Patch" in desc
