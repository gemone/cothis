"""``cothis.tools.fs.search`` — content search across files.

Regex-based content search returning ``[{file, line, text}]`` entries.
Uses the stdlib ``re.compile`` walker; gated ripgrep backend deferred.

Security mitigations: per-file size cap, per-line length cap, total
files-scanned cap, and a sensitive-name denylist that blocks search
of credentials / keys / tokens regardless of hygiene flags.
"""

from __future__ import annotations

import fnmatch
import logging
import re
import time
from pathlib import Path
from typing import Any

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import (
    _IGNORED_DIRS,
    WORKDIR,
    _load_gitignore,
    _resolve_under,
)

logger = logging.getLogger(__name__)

_MAX_PATTERN_LEN = 256
_MAX_FILE_BYTES = 1_048_576  # 1 MiB — larger files are skipped entirely.
_MAX_LINE_LEN = 4096  # skip long lines (ReDoS / log noise).
_MAX_FILES_SCANNED = 5000  # total work cap — bounds traversal even with 0 matches.
# cothis: wall-clock cap on the whole call (#111). The deadline is
# checked at the outer (per-file) and inner (per-line) loop
# boundaries; on hit, the call returns partial results.
_DEADLINE_SECONDS = 5.0

# Denylist of filename patterns that carry secrets. Checked independently
# of the dotfile rule — non-dot secrets like ``credentials.json``,
# ``id_rsa``, ``.npmrc`` must never be searched regardless of ``all``.
_SENSITIVE_NAMES = frozenset(
    {
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        ".env",
        ".npmrc",
        ".pypirc",
        ".netrc",
        ".git-credentials",
    }
)
_SENSITIVE_PATTERNS = (
    "credentials*",
    "secrets.*",
    "auth.*",
    ".env*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.keystore",
)


def _is_sensitive(name: str) -> bool:
    """True if the filename matches a known secret-bearing pattern."""
    if name in _SENSITIVE_NAMES:
        return True
    return any(fnmatch.fnmatchcase(name, pat) for pat in _SENSITIVE_PATTERNS)


_SEARCH_DESCRIPTION = """Search file contents for a regex pattern.

Returns a list of ``{file, line, text}`` dicts — ``file`` is the path
relative to ``path``, ``line`` is the 1-based line number (matches
``fs.read`` numbering so follow-up reads land on the right line),
``text`` is the full matched line (trailing newline stripped).

Sensitive files (credentials, private keys, ``.env``, tokens) are
always excluded regardless of ``glob`` — searching for them silently
returns zero results.

Example::

    fs.search(pattern='TODO', glob='*.py')
    → [{"file": "src/app.py", "line": 42, "text": "# TODO: refactor"}]
"""


@tool("fs.search", description=_SEARCH_DESCRIPTION)
def _search(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    max_results: int = 50,
) -> list[dict[str, str]] | str:
    """Search file contents for a regex pattern.

    Returns ``[{file, line, text}]`` — ``file`` relative to ``path``,
    ``line`` is 1-based, ``text`` is the matched line (trailing newline
    stripped). Hygiene (dotfiles, gitignore, noise dirs) matches
    ``fs.list``. Sensitive files (credentials, private keys, tokens)
    are always excluded.

    Args:
        pattern: Regex to search in file contents.
        path: Directory to search. Relative to cwd.
        glob: Filename glob filter (e.g. ``"*.py"``).
        max_results: Cap on returned matches (default 50).

    Returns:
        List of ``{"file": <rel>, "line": <int>, "text": <str>}`` entries,
        or ``"Error: ..."`` on invalid regex or missing path.
    """
    if len(pattern) > _MAX_PATTERN_LEN:
        return f"Error: pattern exceeds {_MAX_PATTERN_LEN} chars"
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"

    cwd = WORKDIR.get() or Path.cwd()
    try:
        root = _resolve_under(path, cwd)
    except Exception:
        return f"Error: path outside cwd boundary: {path}"
    if not root.exists():
        return f"Error: no such path: {path}"
    if not root.is_dir():
        return f"Error: not a directory: {path}"

    logger.debug("fs.search: stdlib backend")

    gitignore = _load_gitignore(root)
    results: list[dict[str, str]] = []
    files_scanned = 0
    deadline = time.perf_counter() + _DEADLINE_SECONDS
    deadline_hit = False

    for p in root.rglob("*"):
        if len(results) >= max_results or files_scanned >= _MAX_FILES_SCANNED:
            break
        if time.perf_counter() > deadline:
            deadline_hit = True
            break
        if not p.is_file():
            continue
        if _is_sensitive(p.name):
            continue
        if glob and not fnmatch.fnmatchcase(p.name, glob):
            continue
        rel = p.relative_to(root)
        rel_str = rel.as_posix()
        if any(part in _IGNORED_DIRS for part in rel.parts):
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        if gitignore is not None and gitignore.match_file(rel_str):
            continue
        try:
            if p.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        files_scanned += 1
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if len(results) >= max_results:
                        break
                    if time.perf_counter() > deadline:
                        deadline_hit = True
                        break
                    if len(line) > _MAX_LINE_LEN:
                        continue
                    if regex.search(line):
                        results.append(
                            {"file": rel_str, "line": str(i), "text": line.rstrip("\n")}
                        )
        except OSError:
            continue

    if deadline_hit:
        logger.warning(
            "fs.search: wall-clock cap %.1fs hit; returning %d result(s) "
            "found before the deadline (ReDoS or huge tree).",
            _DEADLINE_SECONDS, len(results),
        )

    return results
