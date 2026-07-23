"""Audit production source for text-boundary violations (#219).

Enforces the 3 rules from AGENTS.md § "Text boundary guard":

1. **Decode bytes strict by default** — no ``errors='replace'`` on
   paths whose output reaches the model or user. ``errors='ignore'``
   is allowed only with an inline ``# text-boundary: allow`` marker
   (e.g. binary-safe regex search).
2. **Emit Unicode-native by default** — no ``ensure_ascii=True`` without
   an inline ``# text-boundary: allow`` justification (token-cost hit).
3. **No hardcoded ``\\n`` on file-content mutation paths** — appending
   ``+ "\\n"`` to replacement content (the #96 / #215 pattern) is
   forbidden in ``tools/fs/*``. Use ``splitlines(keepends=True)`` +
   ``"".join(...)`` instead.

The allowlist is a same-line comment marker so the rationale sits next
to the call site, not in a separate file.
"""

from __future__ import annotations

import re
import tokenize
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).parent.parent / "src" / "cothis"

# Files exempted from rule 3 (not file-content mutation paths).
# ``format.py`` legitimately emits ``"\\n"`` in output separators
# (CSV row terminator, TSV column separator) — that's emitting format
# syntax, not mutating file content.
_RULE3_SCOPED_MODULES = ("tools/fs/",)

_ALLOW_MARKER = "text-boundary: allow"


def _python_files() -> list[Path]:
    return sorted(_SRC_ROOT.rglob("*.py"))


def _line_has_allow_marker(line: str) -> bool:
    return _ALLOW_MARKER in line


# ---------------------------------------------------------------------
# Tokenization approach: walk string literals, avoid comment false positives.
# Regex over raw source would catch strings inside comments / docstrings.
# ---------------------------------------------------------------------


def _string_literals(path: Path) -> list[tuple[int, str, str]]:
    """Return (line_no, literal_value, full_source_line) for each string literal.

    Skips multi-line string tokens (module/class/function docstrings)
    so documentation that *mentions* a forbidden pattern as a warning
    isn't flagged.
    """
    src = path.read_text(encoding="utf-8")
    out: list[tuple[int, str, str]] = []
    try:
        tokens = list(tokenize.generate_tokens(iter(src.splitlines(keepends=True)).__next__))
    except tokenize.TokenError:
        return out
    lines = src.splitlines(keepends=True)
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        # Skip multi-line strings (docstrings) — tok.start[0] != tok.end[0].
        if tok.start[0] != tok.end[0]:
            continue
        lineno = tok.start[0]
        if lineno == 0 or lineno > len(lines):
            continue
        out.append((lineno, tok.string, lines[lineno - 1]))
    return out


def _errors_replace_sites(path: Path) -> list[str]:
    """Rule 1: ``errors='replace'`` / ``errors="replace"`` on a string-literal token."""
    bad: list[str] = []
    for lineno, literal, line in _string_literals(path):
        if "replace" not in literal:
            continue
        # match errors='replace' or errors="replace" (with optional spaces)
        if not re.search(r"errors\s*=\s*(['\"]\s*replace\s*['\"])", literal):
            continue
        if _line_has_allow_marker(line):
            continue
        bad.append(f"{path}:{lineno} errors='replace'")
    return bad


def _ensure_ascii_true_sites(path: Path) -> list[str]:
    """Rule 2: ``ensure_ascii=True`` without inline justification."""
    bad: list[str] = []
    for lineno, literal, line in _string_literals(path):
        if "ensure_ascii" not in literal:
            continue
        if not re.search(r"ensure_ascii\s*=\s*True", literal):
            continue
        if _line_has_allow_marker(line):
            continue
        bad.append(f"{path}:{lineno} ensure_ascii=True")
    return bad


def _hardcoded_newline_append_sites(path: Path) -> list[str]:
    """Rule 3: ``+ "\\n"`` literal on file-content mutation paths.

    Scoped to ``tools/fs/`` modules per the issue (the motivating bug
    was patch.py's replacement-line terminator, now deleted). Matches
    the explicit append pattern ``... + "\\n"`` (or single-quoted) —
    a bare ``"\\n"`` argument to another call (e.g. ``count("\\n")``)
    is not an append and is not flagged.
    """
    bad: list[str] = []
    # Contiguous append pattern: identifier-or-close-paren, spaces, +, spaces, "\n"
    append_pat = re.compile(r"\+\s*(['\"]\\\\n['\"]|['\"]\\n['\"])")
    for lineno, _literal, line in _string_literals(path):
        if _line_has_allow_marker(line):
            continue
        if append_pat.search(line):
            bad.append(f"{path}:{lineno} hardcoded '\\n' append")
    return bad


# ---------------------------------------------------------------------
# Parametrized audit — every production source file is checked against
# every rule. Empty violation list = pass.
# ---------------------------------------------------------------------


_PYFILES = _python_files()


@pytest.mark.parametrize(
    "path",
    _PYFILES,
    ids=[str(p.relative_to(_SRC_ROOT.parent)) for p in _PYFILES],
)
def test_no_errors_replace(path: Path) -> None:
    """Rule 1 — no ``errors='replace'`` without inline allow."""
    violations = _errors_replace_sites(path)
    assert not violations, (
        "errors='replace' silently injects U+FFFD on decode failure (#166). "
        "Use errors='strict' (default) or add `# text-boundary: allow` "
        "with rationale. Violations:\n" + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "path",
    _PYFILES,
    ids=[str(p.relative_to(_SRC_ROOT.parent)) for p in _PYFILES],
)
def test_no_ensure_ascii_true(path: Path) -> None:
    """Rule 2 — no ``ensure_ascii=True`` without inline allow."""
    violations = _ensure_ascii_true_sites(path)
    assert not violations, (
        "ensure_ascii=True escapes every non-ASCII codepoint (~1.5x token "
        "cost under BPE, #108). Use ensure_ascii=False (the project default) "
        "or add `# text-boundary: allow` with rationale. "
        "Violations:\n" + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "path",
    [p for p in _PYFILES if any(seg in str(p) for seg in _RULE3_SCOPED_MODULES)],
    ids=[
        str(p.relative_to(_SRC_ROOT.parent))
        for p in _PYFILES
        if any(seg in str(p) for seg in _RULE3_SCOPED_MODULES)
    ],
)
def test_no_hardcoded_newline_in_fs_tools(path: Path) -> None:
    """Rule 3 — no ``+ '\\n'`` literal in ``tools/fs/*`` (file mutation paths)."""
    violations = _hardcoded_newline_append_sites(path)
    assert not violations, (
        "Appending '\\n' to file-content mutation lines reintroduces the "
        "#96 / #215 class of bug (mixed line endings, spurious trailing "
        "newline). Use splitlines(keepends=True) + ''.join(...). "
        "Violations:\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------
# Self-test: the allowlist marker actually suppresses a violation.
# ---------------------------------------------------------------------


def test_allowlist_marker_suppresses_errors_replace() -> None:
    """If someone marks a line ``# text-boundary: allow``, the audit skips it."""
    fake_line = '    with open(p, encoding="utf-8", errors="replace") as f:  # text-boundary: allow'
    assert _line_has_allow_marker(fake_line)


def test_audit_catches_naked_errors_replace() -> None:
    """Without the marker, the regex flags the call."""
    fake_line = '    with open(p, encoding="utf-8", errors="replace") as f:'
    assert not _line_has_allow_marker(fake_line)
    assert re.search(
        r"errors\s*=\s*(['\"]\s*replace\s*['\"])", fake_line
    )
