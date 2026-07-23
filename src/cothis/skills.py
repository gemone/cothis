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
    # ``delete`` (default) archives tagged blocks on deactivate; ``summarize``
    # declares intent to summarise before archiving. The Summarize strategy
    # is not yet implemented — ``deactivate_skill`` falls back to Delete
    # with a logged WARNING (#170).
    deactivation: str = "delete"


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

    raw_deactivation = str(meta.get("deactivation") or "delete").strip().lower()
    if raw_deactivation not in ("delete", "summarize"):
        logger.warning(
            "skills: %s has unknown deactivation %r; defaulting to 'delete'.",
            path, raw_deactivation,
        )
        raw_deactivation = "delete"

    return Skill(
        name=str(name),
        description=str(description).strip(),
        body=body.strip(),
        source=path,
        deactivation=raw_deactivation,
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


_LOAD_DESCRIPTION = """Activate a skill by name; the skill body is returned for use.

Call this when you need the detailed instructions of a skill listed
in ``<available_skills>``. The skill body comes back wrapped in a
``<skill_content>`` XML block; resource files (if any) are listed in
a sibling ``<skill_resources>`` block. Repeated calls for an
already-active skill return a short notice instead of reloading.

Example::

    load_skill(name='git-commit')
    → "<skill_content name='git-commit'>\\n...body...\\n</skill_content>"
"""


@tool(
    name="load_skill",
    inject_session=True,
    skill_marker=True,
    description=_LOAD_DESCRIPTION,
)
def load_skill(name: str, _session: Any) -> str:
    """Activate a skill, returning its body wrapped in ``<skill_content>``.

    Effect: ``_session._active_skills.add(name)`` (idempotent — repeat
    calls return a notice rather than re-injecting the body). The
    catalog is discovered fresh on each call via ``discover_skills``
    (3-layer: project > user-cothis > user-agents), so newly installed
    skills are pickable without restarting the session.

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
    """Slash handler for ``/reload-skills`` (#172, #74 final AC).

    Re-runs ``discover_skills`` from the session's cwd (or
    ``Path.cwd()`` when no session is attached) and returns a
    one-line summary naming the count and skill names. The next
    agent run picks up the fresh catalog automatically via
    ``_assemble_system``.

    Active skills no longer present in the new discovery (vanished
    from disk) are archived via ``Session._deactivate_skill`` and a
    WARNING is logged per vanished skill. The archival composes with
    Half A + B + the in-memory walk from #167/#168/#169.

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

    discovered_names = {s.name for s in skills}
    vanished: list[str] = []
    if ctx.session is not None:
        for name in sorted(ctx.session.active_skills):
            if name not in discovered_names:
                logger.warning(
                    "skills: /reload-skills: %r vanished from disk; "
                    "archiving (was active).",
                    name,
                )
                ctx.session._deactivate_skill(name)
                vanished.append(name)

    names = sorted(discovered_names)
    parts: list[str] = []
    if not names:
        parts.append("Reloaded skills catalog: 0 skills discovered.")
    else:
        parts.append(
            f"Reloaded skills catalog: {len(names)} skill(s) discovered: "
            + ", ".join(names)
        )
    if vanished:
        parts.append(
            f"Archived {len(vanished)} vanished skill(s): "
            + ", ".join(vanished)
        )
    return " ".join(parts)


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


_DEACTIVATE_DESCRIPTION = """Remove a previously-activated skill from future context.

After this call, the skill's instructions are excluded from
subsequent turns — the model will no longer see them. Use this
when a loaded skill is no longer relevant and context space matters.

Example::

    deactivate_skill(name='git-commit')
    → "Skill 'git-commit' archived."

Returns a confirmation on success, a notice if the skill was
already archived or was never activated.
"""


@tool(
    name="deactivate_skill",
    inject_session=True,
    skill_marker=True,
    description=_DEACTIVATE_DESCRIPTION,
)
def deactivate_skill(name: str, _session: Any) -> str:
    """Retire an active skill so context assembly excludes its blocks.

    Effect: ``_cothis_state='archived'`` propagates to the skill's
    tagged blocks (Half A at enqueue time, Half B via queued UPDATE on
    historical rows, in-memory walk on the current mirror). The
    projection layer (``_request_messages``) filters blocks with
    ``_cothis_state='archived'`` so the model never sees them on later
    turns. The skill stays on disk and in the catalog; only its blocks
    are hidden. Re-activation via ``load_skill`` produces a new visible
    epoch.

    Args:
        name: The skill name (as shown in ``<available_skills>``).
    """
    if _session is None:
        return "Error: deactivate_skill requires a live session to record archival state."

    # Order matters (#180): session state first, catalog last. A skill
    # activated earlier in the session may have been removed from disk
    # mid-session (deleted directory, ``git pull`` removed it, symlink
    # target gone). It's still in ``_active_skills`` — deactivating it
    # must succeed so the tagged blocks archive. The catalog check is
    # the fallback for names the session has never seen.
    if _session.is_skill_archived(name):
        return f"Skill {name!r} is already archived."

    if _session.is_skill_active(name):
        skill = _deactivate_active_skill(name, _session)
        return skill

    catalog = discover_skills(Path.cwd())
    by_name = {s.name: s for s in catalog}
    if name not in by_name:
        return (
            f"Error: unknown skill {name!r}. "
            f"Available: {', '.join(sorted(by_name)) or '(none)'}."
        )
    return f"Skill {name!r} is not currently active; nothing to deactivate."


def _deactivate_active_skill(name: str, _session: Any) -> str:
    """Run the archival + build the result message for an active skill.

    Factored out so the order-of-checks in ``deactivate_skill`` stays
    flat. Reads the deactivation declaration if the skill is still on
    disk (Summarize fallback from #170); missing-from-disk skills use
    the Delete strategy directly (no declaration to read).
    """
    deactivation = "delete"
    summary_note = ""
    catalog = discover_skills(Path.cwd())
    by_name = {s.name: s for s in catalog}
    if name in by_name:
        deactivation = by_name[name].deactivation
        if deactivation == "summarize":
            logger.warning(
                "skills: %r declares deactivation: summarize, which is not "
                "yet implemented; falling back to Delete strategy.",
                name,
            )
            summary_note = (
                f" (Note: {name!r} declares deactivation: summarize; the "
                f"Summarize strategy is not yet implemented — used Delete "
                f"fallback.)"
            )
    _session._deactivate_skill(name)
    return (
        f"Skill {name!r} archived. Future blocks for this skill will be "
        f"marked state='archived'; the model will not see them in "
        f"subsequent turns.{summary_note}"
    )
