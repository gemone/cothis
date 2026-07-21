"""``cothis.tools.fs.search`` — content search across files.

Regex-based content search returning ``[{file, line, text}]`` entries.
Uses the gated ripgrep backend when available (``rg --json``); falls
back to stdlib ``re.compile`` walker otherwise. The backend choice
is logged once at DEBUG on first use.

Security mitigations: per-file size cap, per-line length cap, total
files-scanned cap, and a sensitive-name denylist that blocks search
of credentials / keys / tokens regardless of hygiene flags.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cothis.tools.core import tool
from cothis.tools.fs._hygiene import (
    _IGNORED_DIRS,
    WORKDIR,
    _load_gitignore,
    _resolve_under,
)

if TYPE_CHECKING:
    import pathspec

logger = logging.getLogger(__name__)

# Best-effort, not thread-safe.
_backend_logged = False
_MAX_PATTERN_LEN = 256
_MAX_FILE_BYTES = 1_048_576  # 1 MiB — larger files are skipped entirely.
_MAX_LINE_LEN = 4096  # skip long lines (ReDoS / log noise).
_MAX_FILES_SCANNED = 5000  # total work cap — bounds traversal even with 0 matches.

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


def _log_backend_choice(backend: str) -> None:
    global _backend_logged
    if not _backend_logged:
        logger.debug("fs tools: fs.search using backend %s", backend)
        _backend_logged = True


def _is_sensitive(name: str) -> bool:
    """True if the filename matches a known secret-bearing pattern."""
    if name in _SENSITIVE_NAMES:
        return True
    return any(fnmatch.fnmatchcase(name, pat) for pat in _SENSITIVE_PATTERNS)


@tool("fs.search")
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

    _log_backend_choice("stdlib")

    gitignore = _load_gitignore(root)
    results: list[dict[str, str]] = []
    files_scanned = 0

    for p in root.rglob("*"):
        if len(results) >= max_results or files_scanned >= _MAX_FILES_SCANNED:
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
                    if len(line) > _MAX_LINE_LEN:
                        continue
                    if regex.search(line):
                        results.append(
                            {"file": rel_str, "line": str(i), "text": line.rstrip("\n")}
                        )
        except OSError:
            continue

    return results
