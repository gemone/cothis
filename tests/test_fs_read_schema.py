"""Tests for ``fs.read`` schema description (#196).

Same audit standard as #190/#192/#194: the model-facing description
must include the line-numbered output format, multi-path header
shape, per-file error behavior, and a concrete example.
"""

from __future__ import annotations

from cothis.tools import schema_for
from cothis.tools.fs.read import read


def _schema() -> dict:
    return schema_for(read)


def test_description_mentions_line_numbers() -> None:
    """Description tells the model lines are numbered."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "line number" in lowered or "numbered" in lowered


def test_description_mentions_tab_separator() -> None:
    """Description mentions tab-separated prefix."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "tab" in lowered


def test_description_mentions_multi_path_header() -> None:
    """Description mentions the ``=== <path> ===`` header shape."""
    desc = _schema().get("description", "")
    assert "===" in desc


def test_description_mentions_per_file_error_resilience() -> None:
    """Description mentions missing-file behavior in multi-path calls."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "error" in lowered and ("missing" in lowered or "not found" in lowered)


def test_description_has_example_invocation() -> None:
    """Description includes a concrete ``fs.read(path=...)`` example."""
    desc = _schema().get("description", "")
    assert "fs.read(path=" in desc


def test_description_mentions_line_range_args() -> None:
    """Description mentions ``start_line`` / ``end_line``."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "start_line" in lowered and "end_line" in lowered
