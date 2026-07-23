"""Tests for ``load_skill`` schema description (#194).

Same audit standard as #190 (fs.write) and #192 (deactivate_skill):
description must include concrete return-format signal, an example,
and idempotency notice.
"""

from __future__ import annotations

from cothis.skills import load_skill
from cothis.tools import schema_for


def _schema() -> dict:
    return schema_for(load_skill)


def test_description_mentions_skill_content_xml() -> None:
    """Description tells the model what wrap to expect (``<skill_content>``)."""
    desc = _schema().get("description", "")
    assert "skill_content" in desc.lower()


def test_description_has_example_invocation() -> None:
    """Description includes a concrete ``load_skill(name=...)`` example."""
    desc = _schema().get("description", "")
    assert "load_skill(name=" in desc


def test_description_mentions_idempotency() -> None:
    """Description mentions repeated-call behavior (already-active notice)."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "already active" in lowered or "repeat" in lowered or "idempot" in lowered


def test_description_mentions_available_skills_catalog() -> None:
    """Description references the catalog the model picks from."""
    desc = _schema().get("description", "")
    assert "available_skills" in desc.lower() or "catalog" in desc.lower()


def test_description_mentions_when_to_use() -> None:
    """Description includes guidance on when to call."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "need" in lowered or "when" in lowered or "detailed" in lowered


def test_description_mentions_resources() -> None:
    """Description mentions resource files (part of the return shape)."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "resource" in lowered
