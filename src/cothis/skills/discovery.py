"""``cothis.skills`` — Agent Skills discovery and catalog rendering.

Discovery scans three layer dirs (lowest precedence first; project
shadows user): ``~/.agents/skills/`` (user-agents), ``~/.cothis/skills/``
(user-cothis), ``<cwd>/.agents/skills/`` (project). A skill is a
directory with a ``SKILL.md`` whose YAML frontmatter declares ``name``
and ``description``.

Failures are lenient — a single broken ``SKILL.md`` never crashes the
agent. The catalog renders an ``<available_skills>`` block that names
each discovered skill. Catalog text is XML-escaped before insertion so
a malicious ``name`` / ``description`` cannot break out of the block.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import yaml

logger = logging.getLogger(__name__)

_LAYER_ORDER: tuple[str, ...] = ("user-agents", "user-cothis", "project")
_SKILL_FILE = "SKILL.md"
# Cap a SKILL.md at 1 MiB so a planted multi-GB file can't OOM the agent
# at startup. Real skills are a few KB.
_MAX_SKILL_BYTES = 1 * 1024 * 1024
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)\Z", re.DOTALL
)

_CATALOG_HEADER = (
    "<available_skills>\n"
    "The following Agent Skills are available in this project. Each entry is a "
    "directory under `.agents/skills/`, `~/.cothis/skills/`, or "
    "`~/.agents/skills/` containing a `SKILL.md`. Refer to a skill by its "
    "name when the user's task matches its description.\n\n"
    "Skills:"
)


@dataclass(frozen=True)
class SkillRecord:
    """One discovered skill."""

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


def format_catalog(skills: list[SkillRecord]) -> str | None:
    """Render the ``<available_skills>`` block, or ``None`` when empty.

    Catalog text is XML-escaped: a malicious ``name`` or ``description``
    containing ``</available_skills>`` cannot break out of the block.
    Pure function — callers can memoise on the discovered list.
    """
    if not skills:
        return None
    lines = [_CATALOG_HEADER]
    for skill in skills:
        lines.append(f"- {escape(skill.name)}: {escape(skill.description)}")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _layer_dirs() -> dict[str, Path]:
    """Resolve the three layer dirs, honouring ``COTHIS_AGENTS_USER_GLOBAL``.

    ``HOME`` is honoured when set so tests (and Windows users with a
    non-standard layout) can steer the user-agents layer without
    patching :func:`pathlib.Path.home` — which on Windows ignores
    ``HOME`` entirely (uses ``USERPROFILE``).
    """
    home_env = os.environ.get("HOME")
    home = Path(home_env).expanduser() if home_env else Path.home()
    cothis_home = Path(
        os.environ.get("COTHIS_HOME") or (Path.home() / ".cothis")
    ).expanduser()
    user_global = os.environ.get("COTHIS_AGENTS_USER_GLOBAL", "1")
    user_global = user_global.lower() not in ("0", "false", "no", "off")

    layer_dirs: dict[str, Path] = {"project": Path.cwd() / ".agents" / "skills"}
    if user_global:
        layer_dirs["user-cothis"] = cothis_home / "skills"
        layer_dirs["user-agents"] = home / ".agents" / "skills"
    return layer_dirs


def _scan_layer(layer_dir: Path, layer: str) -> list[SkillRecord]:
    """Return one ``SkillRecord`` per ``SKILL.md`` under ``layer_dir``."""
    if not layer_dir.is_dir():
        return []
    out: list[SkillRecord] = []
    for entry in sorted(layer_dir.iterdir()):
        if not entry.is_dir() or entry.is_symlink():
            continue
        skill_md = entry / _SKILL_FILE
        if not skill_md.is_file() or skill_md.is_symlink():
            continue
        record = _parse_skill_md(skill_md, layer)
        if record is not None:
            out.append(record)
    return out


def _parse_skill_md(path: Path, layer: str) -> SkillRecord | None:
    """Parse one ``SKILL.md``; return ``None`` + WARN on any failure."""
    try:
        # Cap reads at _MAX_SKILL_BYTES so a planted multi-GB file can't
        # OOM the agent at startup.
        with path.open("r", encoding="utf-8") as f:
            text = f.read(_MAX_SKILL_BYTES + 1)
        if len(text) > _MAX_SKILL_BYTES:
            logger.warning(
                "Skill %s exceeds %d bytes; skipped (size cap).",
                path, _MAX_SKILL_BYTES,
            )
            return None
    except (OSError, UnicodeDecodeError, RecursionError) as exc:
        logger.warning("Skill %s unreadable (%s); skipped.", path, exc)
        return None
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        frontmatter: dict[str, str] = {}
        body = text
    else:
        raw_yaml = match.group("yaml")
        body = match.group("body") or ""
        try:
            loaded = yaml.safe_load(raw_yaml) or {}
        except (yaml.YAMLError, RecursionError) as exc:
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
