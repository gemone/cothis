"""``cothis.skills`` — Agent Skills discovery and catalog rendering.

Public re-exports for callers (``cothis.agent`` and tests). Submodule
layout: :mod:`cothis.skills.discovery` carries the layer scanner, YAML
parser, shadow logic, size cap, and catalog renderer in one cohesive
module — the catalog is a pure function of the discovered list, so it
lives next to the data shape it consumes.
"""

from __future__ import annotations

from cothis.skills.discovery import SkillRecord, discover_skills, format_catalog

__all__ = ["SkillRecord", "discover_skills", "format_catalog"]
