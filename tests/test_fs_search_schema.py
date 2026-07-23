"""Tests for ``fs.search`` schema description (#199).

Same audit standard as #190/#192/#194/#196: model-facing description
must include the return-format (``[{file, line, text}]``), field
semantics (1-based line, matched-line text), sensitive-file exclusion,
and a concrete example.
"""

from __future__ import annotations

from cothis.tools import schema_for
from cothis.tools.fs.search import _search


def _schema() -> dict:
    return schema_for(_search)


def test_description_mentions_return_dict_shape() -> None:
    """Description tells the model what fields each result has."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "file" in lowered and "line" in lowered and "text" in lowered


def test_description_mentions_line_is_1_based() -> None:
    """Description mentions ``line`` is 1-based (matches fs.read numbering)."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "1-based" in lowered or "1 based" in lowered or "one-based" in lowered


def test_description_mentions_sensitive_file_exclusion() -> None:
    """Description mentions credentials / .env / keys always excluded."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "sensitive" in lowered or ".env" in lowered or "credential" in lowered


def test_description_has_example_invocation() -> None:
    """Description includes a concrete ``fs.search(pattern=...)`` example."""
    desc = _schema().get("description", "")
    assert "fs.search(" in desc


def test_description_mentions_regex_pattern() -> None:
    """Description mentions the ``pattern`` arg is a regex."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "regex" in lowered or "pattern" in lowered


def test_description_mentions_glob_filter() -> None:
    """Description mentions the ``glob`` filename filter."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "glob" in lowered
