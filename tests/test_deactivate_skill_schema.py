"""Tests for ``deactivate_skill`` schema description (#192).

The model-facing description must be jargon-free and include a
concrete example — same standard as #190 (fs.write). Verifies the
description carries actionable effect + when-to-call guidance,
without internal-only phrases ("tagged blocks", "Delete strategy",
"queued UPDATE").
"""

from __future__ import annotations

from cothis.skills import deactivate_skill
from cothis.tools import schema_for


def _schema() -> dict:
    return schema_for(deactivate_skill)


def test_description_mentions_context_exclusion() -> None:
    """Description tells the model what the effect is (instructions excluded)."""
    desc = _schema().get("description", "")
    assert "context" in desc.lower()


def test_description_has_example_invocation() -> None:
    """Description includes a concrete ``deactivate_skill(name=...)`` example."""
    desc = _schema().get("description", "")
    assert "deactivate_skill(name=" in desc


def test_description_mentions_return_confirmation() -> None:
    """Description mentions what the tool returns on success."""
    desc = _schema().get("description", "")
    assert "archived" in desc.lower()


def test_description_no_jargon_tagged_blocks() -> None:
    """Description omits internal-only phrase ``tagged blocks``."""
    desc = _schema().get("description", "")
    assert "tagged blocks" not in desc.lower()


def test_description_no_jargon_delete_strategy() -> None:
    """Description omits internal-only phrase ``Delete strategy``."""
    desc = _schema().get("description", "")
    assert "delete strategy" not in desc.lower()


def test_description_no_jargon_queued_update() -> None:
    """Description omits internal-only phrase ``queued UPDATE``."""
    desc = _schema().get("description", "")
    assert "queued update" not in desc.lower()


def test_description_mentions_when_to_use() -> None:
    """Description includes guidance on when to call (context relevance)."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "no longer" in lowered or "not needed" in lowered or "irrelevant" in lowered
