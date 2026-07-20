"""``cothis.skills`` — Agent Skills discovery and catalog assembly.

Agent Skills are on-demand knowledge bundles the model can load during a
turn. A skill is a directory containing a ``SKILL.md`` file whose YAML
frontmatter carries ``name`` and ``description`` (the catalog entry the
model sees) and whose body carries the skill's instructions (delivered
to the model on ``load_skill``).

Layers (lowest precedence first — project shadows user):

- ``~/.agents/skills/`` (user-agents; cross-tool convention)
- ``~/.cothis/skills/`` (user-cothis; cothis-specific user skills)
- ``<cwd>/.agents/skills/`` (project; checked into the repo)

Discovery is lenient: a skill with unparseable YAML or an empty
description is skipped with a logged warning (the catalog is for the
model; a malformed entry would mislead it). ``name`` missing defaults
to the directory name. ``name`` ≠ directory name warns but loads.

The catalog is a pure function of the discovered list: see
:func:`format_catalog`.
"""

from __future__ import annotations

from cothis.skills.catalog import format_catalog
from cothis.skills.discovery import SkillRecord, discover_skills

__all__ = ["SkillRecord", "discover_skills", "format_catalog"]
