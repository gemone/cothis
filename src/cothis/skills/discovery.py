"""``cothis.skills.discovery`` — scan layer dirs for ``SKILL.md`` bundles.

Three layer directories (lowest precedence first; project shadows user):

- ``user-agents``: ``~/.agents/skills/`` (cross-tool convention)
- ``user-cothis``: ``~/.cothis/skills/`` (cothis-specific user skills)
- ``project``: ``<cwd>/.agents/skills/`` (checked into the repo)

Discovery is **lenient by design** — a single broken ``SKILL.md`` never
crashes the agent. Failure modes and their handling:

- Missing ``SKILL.md`` in a directory: silently skipped (lets users
  stage partial skills or hold asset folders).
- Unparseable YAML frontmatter: skipped with a logged WARNING naming
  the file.
- Missing ``description`` (or empty): skipped with WARNING. The catalog
  is for the model; an undescribed skill would mislead it.
- Missing ``name``: defaults to the directory name (no warning).
- ``name`` ≠ directory name: WARNING but the skill still loads.

Same-name skills across layers: the higher-precedence layer wins (project
> user-cothis > user-agents); a WARNING names the shadow so users don't
silently lose access to the lower copy.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Layer order, lowest precedence first. The catalog walks in reverse
# (project first) so shadow-WARNINGs fire on the lower-precedence loser.
_LAYER_ORDER: tuple[str, ...] = ("user-agents", "user-cothis", "project")
_SKILL_FILE = "SKILL.md"
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)\Z", re.DOTALL
)


@dataclass(frozen=True)
class SkillRecord:
    """One discovered skill — the data ``format_catalog`` consumes.

    ``layer`` is the discovery layer (``user-agents`` / ``user-cothis`` /
    ``project``) so the catalog and shadow logic can name the source.
    ``path`` is the ``SKILL.md`` location so ``load_skill`` can read the
    body on demand without re-scanning. ``body`` is cached at discovery
    time so a single filesystem pass suffices.
    """

    name: str
    description: str
    path: Path
    body: str
    layer: str


def discover_skills() -> list[SkillRecord]:
    """Scan the three layer dirs; return skills ordered by name.

    Project shadows user-cothis shadows user-agents. Same-name shadows
    fire a WARNING. Within a layer, duplicate names are not possible
    (each skill lives in its own directory).
    """
    layer_dirs = _layer_dirs()
    # Walk highest-precedence first so the first-seen wins per name; the
    # later (lower-precedence) losers are logged as shadows.
    records: dict[str, SkillRecord] = {}
    for layer in reversed(_LAYER_ORDER):
        layer_dir = layer_dirs.get(layer)
        if layer_dir is None:
            continue
        for record in _scan_layer(layer_dir, layer):
            if record.name in records:
                winner = records[record.name]
                logger.warning(
                    "Skill %r in layer %r shadows the same name in layer %r "
                    "(%s); the lower-precedence copy is hidden from the catalog.",
                    record.name,
                    winner.layer,
                    layer,
                    record.path,
                )
                continue
            records[record.name] = record
    return sorted(records.values(), key=lambda r: r.name)


def _layer_dirs() -> dict[str, Path]:
    """Resolve the three layer dirs, honouring ``COTHIS_AGENTS_USER_GLOBAL``.

    Mirrors :func:`cothis.agent._load_agents_md`'s env handling so the
    two systems agree on what "user" means. Unknown / disabled layers
    are simply omitted from the returned dict.
    """
    home = Path.home()
    cothis_home = Path(
        os.environ.get("COTHIS_HOME") or (home / ".cothis")
    ).expanduser()
    user_global = os.environ.get("COTHIS_AGENTS_USER_GLOBAL", "1")
    user_global = user_global.lower() not in ("0", "false", "no", "off")

    layer_dirs: dict[str, Path] = {"project": Path.cwd() / ".agents" / "skills"}
    if user_global:
        layer_dirs["user-cothis"] = cothis_home / "skills"
        layer_dirs["user-agents"] = home / ".agents" / "skills"
    return layer_dirs


def _scan_layer(layer_dir: Path, layer: str) -> list[SkillRecord]:
    """Return one ``SkillRecord`` per ``SKILL.md`` under ``layer_dir``.

    Failures (missing dir, unparseable YAML, empty description) log and
    continue — never raise.
    """
    if not layer_dir.is_dir():
        return []
    out: list[SkillRecord] = []
    for entry in sorted(layer_dir.iterdir()):
        if not entry.is_dir():
            continue
        skill_md = entry / _SKILL_FILE
        if not skill_md.is_file():
            continue
        record = _parse_skill_md(skill_md, layer)
        if record is not None:
            out.append(record)
    return out


def _parse_skill_md(path: Path, layer: str) -> SkillRecord | None:
    """Parse one ``SKILL.md``; return ``None`` + WARN on any failure.

    Frontmatter is YAML between ``---`` fences. ``name`` defaults to the
    parent directory name; ``description`` must be non-empty.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Skill %s unreadable (%s); skipped.", path, exc)
        return None
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        # No frontmatter at all — treat the whole file as the body and
        # default name from the directory.
        frontmatter: dict[str, str] = {}
        body = text
    else:
        raw_yaml = match.group("yaml")
        body = match.group("body") or ""
        try:
            loaded = yaml.safe_load(raw_yaml) or {}
        except yaml.YAMLError as exc:
            logger.warning(
                "Skill %s has unparseable YAML frontmatter (%s); skipped.",
                path, exc,
            )
            return None
        if not isinstance(loaded, dict):
            logger.warning(
                "Skill %s frontmatter is not a mapping (got %s); skipped.",
                path, type(loaded).__name__,
            )
            return None
        frontmatter = loaded

    dir_name = path.parent.name
    name = frontmatter.get("name") or dir_name
    if not isinstance(name, str) or not name.strip():
        logger.warning(
            "Skill %s has empty/non-string name after defaulting; skipped.", path
        )
        return None
    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        logger.warning(
            "Skill %s has empty/non-string description; skipped (catalog needs a description).",
            path,
        )
        return None

    if frontmatter.get("name") is not None and frontmatter["name"] != dir_name:
        logger.warning(
            "Skill %s declares name %r but lives in directory %r; loading under the declared name.",
            path, frontmatter["name"], dir_name,
        )

    return SkillRecord(
        name=name.strip(),
        description=description.strip(),
        path=path,
        body=body,
        layer=layer,
    )
