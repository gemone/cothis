"""``cothis.skills`` — Agent Skills discovery + catalog (#68).

Discovers Agent Skills from three layers (project > user-cothis >
user-agents), parses each ``SKILL.md`` leniently, and renders the
``<available_skills>`` catalog block for the system prompt. Also
provides the ``load_skill`` tool for skill activation (#158).

Layers (highest precedence first):
- **Project**: ``.agents/skills/`` (relative to cwd)
- **User-cothis**: ``$COTHIS_HOME/skills/`` (default ``~/.cothis/skills/``)
- **User-agents**: ``~/.agents/skills/``

Cross-layer name conflicts are resolved by shadowing (higher
precedence wins; a ``WARNING`` is logged naming both sources).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from cothis.slash import SlashContext
from cothis.slash import register as slash_register
from cothis.tools.core import tool

if TYPE_CHECKING:
    from cothis.session import Session

logger = logging.getLogger(__name__)

_SKILL_FILE = "SKILL.md"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    """One discovered Agent Skill."""

    name: str
    description: str
    body: str
    source: Path


def discover_skills(
    cwd: Path,
    *,
    cothis_home: Path | None = None,
    user_agents: Path | None = None,
) -> list[Skill]:
    """Discover skills from three layers, resolving cross-layer shadows.

    Returns a list of :class:`Skill` sorted by name. Skills in
    higher-precedence layers shadow skills with the same name in
    lower-precedence layers (a ``WARNING`` names both sources).
    """
    if cothis_home is None:
        cothis_home = Path(
            __import__("os").environ.get("COTHIS_HOME")
            or Path.home() / ".cothis"
        )
    if user_agents is None:
        user_agents = Path.home() / ".agents"

    layers = [
        ("project", cwd / ".agents" / "skills"),
        ("user-cothis", cothis_home / "skills"),
        ("user-agents", user_agents / "skills"),
    ]

    by_name: dict[str, Skill] = {}
    seen_in: dict[str, str] = {}

    for layer_name, layer_dir in layers:
        if not layer_dir.is_dir():
            continue
        for skill_dir in sorted(layer_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / _SKILL_FILE
            if not skill_file.is_file():
                continue
            skill = _parse_skill_md(skill_file)
            if skill is None:
                continue
            if skill.name in by_name:
                logger.warning(
                    "skills: %r in layer %r is shadowed by %r in layer %r",
                    skill.name, layer_name, skill.name, seen_in[skill.name],
                )
                continue
            by_name[skill.name] = skill
            seen_in[skill.name] = layer_name

    return sorted(by_name.values(), key=lambda s: s.name)


def _parse_skill_md(path: Path) -> Skill | None:
    """Parse a ``SKILL.md`` file leniently.

    Returns ``None`` (with a log line) on:
    - Decode failure (neither UTF-8 nor locale-decodable; skip + log).
      Never uses ``errors='replace'`` — silent U+FFFD injection into
      the system prompt violates the same floor ``agent._read_text``
      enforces for AGENTS.md content.
    - Broken YAML frontmatter (skip + log)
    - Empty ``description`` (skip + log)
    Missing ``name`` defaults to the directory name.
    ``name`` ≠ directory → warn + load.
    """
    import locale

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(
                encoding=locale.getpreferredencoding(False),
            )
        except (UnicodeDecodeError, LookupError):
            logger.warning(
                "skills: %s is neither valid UTF-8 nor %s-decodable; skipped.",
                path, locale.getpreferredencoding(False),
            )
            return None
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        logger.warning(
            "skills: %s has no YAML frontmatter; skipped.", path
        )
        return None

    raw_yaml = match.group(1)
    body = text[match.end():]

    try:
        meta = yaml.safe_load(raw_yaml)
    except yaml.YAMLError:
        logger.warning(
            "skills: %s has broken YAML frontmatter; skipped.", path
        )
        return None

    if not isinstance(meta, dict):
        logger.warning(
            "skills: %s frontmatter is not a mapping; skipped.", path
        )
        return None

    dir_name = path.parent.name
    name = meta.get("name") or dir_name
    if meta.get("name") and meta["name"] != dir_name:
        logger.warning(
            "skills: %s name %r ≠ directory %r; loaded with declared name.",
            path, meta["name"], dir_name,
        )

    description = meta.get("description", "")
    if not description or not str(description).strip():
        logger.warning(
            "skills: %s has empty description; skipped.", path
        )
        return None

    return Skill(
        name=str(name),
        description=str(description).strip(),
        body=body.strip(),
        source=path,
    )


def format_catalog(skills: list[Skill]) -> str | None:
    """Render the ``<available_skills>`` catalog block.

    Pure function of the discovered list. Returns ``None`` when the
    list is empty (the caller omits the system-prompt block entirely).
    """
    if not skills:
        return None
    lines = ["<available_skills>"]
    for skill in skills:
        lines.append(f"  - {skill.name}: {skill.description}")
    lines.append("</available_skills>")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# load_skill tool (#158)
# ---------------------------------------------------------------------


@tool(name="load_skill", inject_session=True, skill_marker=True)
def load_skill(name: str, _session: Any) -> str:
    """Activate a skill by name and load its content.

    Use this when you need the detailed instructions from a skill
    listed in ``<available_skills>``. The skill body is returned
    wrapped in ``<skill_content>`` tags. Repeated calls for an
    already-active skill return a notice instead of reloading.

    Args:
        name: The skill name (as shown in the catalog).
    """
    catalog = discover_skills(Path.cwd())
    by_name = {s.name: s for s in catalog}

    if name not in by_name:
        return f"Error: unknown skill {name!r}. Available: {', '.join(sorted(by_name)) or '(none)'}."

    if _session is not None and _session.is_skill_active(name):
        return f"Skill {name!r} is already active."

    skill = by_name[name]
    if _session is not None:
        _session._activate_skill(name)

    parts = [f"<skill_content name={name!r}>\n{skill.body}\n</skill_content>"]

    resources_dir = skill.source.parent
    resource_files = sorted(
        f.relative_to(resources_dir).as_posix()
        for f in resources_dir.rglob("*")
        if f.is_file() and f.name != _SKILL_FILE
    )
    if resource_files:
        parts.append(
            "<skill_resources>\n"
            + "\n".join(f"  - {r}" for r in resource_files)
            + "\n</skill_resources>"
        )

    return "\n\n".join(parts)


async def reload_skills_handler(ctx: SlashContext, args: str) -> str:
    """Slash handler for ``/reload-skills`` (#172).

    Re-runs ``discover_skills`` from the session's cwd (or
    ``Path.cwd()`` when no session is attached) and returns a
    one-line summary naming the count and skill names. The next
    agent run picks up the fresh catalog automatically via
    ``_assemble_system`` — this handler does not mutate any cached
    state, it just validates discovery + reports.

    Discovery failures are caught and surfaced to the user as an
    error summary; the handler never raises.
    """
    cwd = Path.cwd()
    if ctx.session is not None:
        cwd = getattr(ctx.session, "cwd", cwd)
        if not isinstance(cwd, Path):
            cwd = Path.cwd()
    try:
        skills = discover_skills(cwd)
    except Exception as exc:  # noqa: BLE001 — best-effort; surface to user
        logger.warning("skills: /reload-skills discovery failed: %s", exc)
        return f"Reload failed: {exc}"
    names = sorted(s.name for s in skills)
    if not names:
        return "Reloaded skills catalog: 0 skills discovered."
    return (
        f"Reloaded skills catalog: {len(names)} skill(s) discovered: "
        + ", ".join(names)
    )


def register_slash_commands() -> None:
    """Register skills-related slash commands (currently ``/reload-skills``).

    Idempotent: safe to call multiple times (the registry overwrites on
    re-registration). Not auto-called at module import — the REPL (or
    whoever wires slash dispatch) calls this explicitly so importing
    ``cothis.skills`` stays side-effect-free.
    """
    slash_register(
        "reload-skills", reload_skills_handler,
        summary="Re-run skill discovery; list discovered skills.",
    )


@tool(name="deactivate_skill", inject_session=True, skill_marker=True)
def deactivate_skill(name: str, _session: Any) -> str:
    """Retire an active skill: its tagged blocks archive (Delete strategy).

    Use this when a skill is no longer needed. Future ``tool_use`` /
    ``tool_result`` blocks for the named skill write ``state='archived'``
    at persist time so context assembly can skip them on later turns.
    Historical rows are covered separately (queued UPDATE).

    Args:
        name: The skill name (as shown in ``<available_skills>``).
    """
    if _session is None:
        return "Error: deactivate_skill requires a live session to record archival state."

    catalog = discover_skills(Path.cwd())
    by_name = {s.name: s for s in catalog}

    if name not in by_name:
        return (
            f"Error: unknown skill {name!r}. "
            f"Available: {', '.join(sorted(by_name)) or '(none)'}."
        )

    if _session.is_skill_archived(name):
        return f"Skill {name!r} is already archived."

    if not _session.is_skill_active(name):
        return f"Skill {name!r} is not currently active; nothing to deactivate."

    _session._deactivate_skill(name)
    return (
        f"Skill {name!r} archived. Future blocks for this skill will be "
        f"marked state='archived'; the model will not see them in "
        f"subsequent turns."
    )
