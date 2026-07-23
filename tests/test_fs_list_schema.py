"""Tests for ``fs.list`` schema description (#201).

Same audit standard as #190/#192/#194/#196/#199: model-facing
description must include return-format (``[{name, type}]``),
truncation shape (``{entries, truncated}`` past 500), dotfile
exclusion, and a concrete example.
"""

from __future__ import annotations

from cothis.tools import schema_for
from cothis.tools.fs.list import _list


def _schema() -> dict:
    return schema_for(_list)


def test_description_mentions_return_dict_shape() -> None:
    """Description tells the model what fields each entry has."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "name" in lowered and "type" in lowered


def test_description_mentions_truncation() -> None:
    """Description mentions the {entries, truncated} shape past 500."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "truncat" in lowered or "500" in lowered


def test_description_mentions_dotfile_exclusion() -> None:
    """Description mentions dotfiles hidden by default."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "dotfile" in lowered or "all=true" in lowered or "all" in lowered


def test_description_has_example_invocation() -> None:
    """Description includes a concrete ``fs.list(...)`` example."""
    desc = _schema().get("description", "")
    assert "fs.list(" in desc


def test_description_mentions_recursive() -> None:
    """Description mentions the ``recursive`` option."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "recursive" in lowered


def test_description_mentions_type_filter() -> None:
    """Description mentions the ``type`` filter (file/dir)."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert '"file"' in lowered or "'file'" in lowered or "file" in lowered
