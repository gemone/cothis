"""``cothis.skills.catalog`` — render the ``<available_skills>`` block.

Pure function of the discovered list. The block is appended to the
``system`` parameter when at least one skill is discovered; omitted
entirely otherwise (no token cost when skills are absent).

The block carries:

- a short usage header naming ``load_skill`` / ``deactive_skill`` so the
  model knows the activation protocol;
- one row per skill (``name`` + ``description``), sorted by name (the
  discovery sort order is preserved).

Layer information is intentionally NOT surfaced to the model — it's an
implementation detail. The model only needs names it can call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cothis.skills.discovery import SkillRecord

_HEADER = (
    "<available_skills>\n"
    "The following Agent Skills are available. Activate one with the "
    "`load_skill(name)` tool to inject its full body into the conversation; "
    "release it with `deactive_skill(name)` when its work is done. Each "
    "skill's body replaces the catalog entry while loaded; the model "
    "should not echo skill content back to the user.\n\n"
    "Skills:"
)


def format_catalog(skills: list[SkillRecord]) -> str | None:
    """Render the catalog block, or ``None`` when ``skills`` is empty.

    The function is pure: callers can memoise on the discovered list,
    and the input list is never mutated.
    """
    if not skills:
        return None
    lines = [_HEADER]
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.description}")
    lines.append("</available_skills>")
    return "\n".join(lines)
