"""Startup latency budget + import-cost audit (#220).

Two parts:

1. **Runtime measurement** — 3-run median wall time of
   ``python -c 'import cothis.cli; cothis.cli.app([\"--help\"])'``
   minus a ``python -c 'pass'`` baseline, asserted against a
   per-platform ceiling. Ceilings are conservative (1.5× measured
   baseline); tighten once stable CI data accumulates.

2. **Static import audit** — scan ``cothis/__init__.py``,
   ``cothis/cli.py``, ``cothis/agent.py`` for top-level third-party
   imports. Each must carry a ``# cost: ~Nms`` comment or be deferred
   under ``TYPE_CHECKING`` / inside a function body. Catches the
   regression mode that #45 / #81 / #118 each fixed reactively.
"""

from __future__ import annotations

import ast
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
_SRC_ROOT = _REPO_ROOT / "src" / "cothis"

# Third-party packages the project depends on (pyproject.toml + their
# transitive module names). Stdlib imports are exempt.
_THIRD_PARTY_MODULES = frozenset({
    "any_llm", "anthropic", "click", "filelock", "griffe", "mcp",
    "pathspec", "prompt_toolkit", "pydantic", "rich", "typer", "yaml",
})

_COST_MARKER = re.compile(r"#\s*cost:\s*~(\d+)\s*ms")

# Files whose top-level imports are in the ``cothis --help`` startup
# path. New files enter this list only when they're imported by one of
# these three (transitively, during startup).
_STARTUP_PATH_FILES = (
    _SRC_ROOT / "__init__.py",
    _SRC_ROOT / "cli.py",
    _SRC_ROOT / "agent.py",
)

# Per-platform wall-time ceilings (ms). Baselines measured 2026-07-24:
#   Linux ~390ms median (3-run, local), Darwin ~500ms (estimate),
#   Windows ~840ms (from #45 post-deferral data point).
# Ceiling = baseline × 1.5 (conservative; issue's target is baseline+50ms
# but CI variance needs more headroom until we have stable data).
_CEILINGS_MS = {
    "Linux": 600,
    "Darwin": 750,
    "Windows": 1300,
}


# =====================================================================
# Part 1 — Runtime measurement
# =====================================================================


def _run_subprocess_ms(code: str) -> float:
    """Run ``python -c <code>`` and return wall time in ms."""
    import time

    t = time.perf_counter()
    subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=str(_REPO_ROOT),
        env={
            **os.environ,
            # Ensure the subprocess imports THIS checkout, not an
            # installed copy. uv-run sets PYTHONPATH already, but be
            # explicit so a bare pytest invocation matches.
            "PYTHONPATH": str(_REPO_ROOT / "src"),
        },
    )
    return (time.perf_counter() - t) * 1000


def _median_of_3(code: str) -> float:
    runs = [_run_subprocess_ms(code) for _ in range(3)]
    return statistics.median(runs)


_HELP_CODE = (
    "import sys; sys.argv = ['cothis', '--help'];\n"
    "from cothis.cli import app;\n"
    "try: app()\n"
    "except SystemExit: pass\n"
)
_PASS_CODE = "pass"


@pytest.mark.skipif(
    platform.system() not in _CEILINGS_MS,
    reason=f"no baseline recorded for {platform.system()}",
)
def test_cothis_help_under_startup_budget() -> None:
    """Median startup overhead must stay under the platform ceiling."""
    help_ms = _median_of_3(_HELP_CODE)
    pass_ms = _median_of_3(_PASS_CODE)
    overhead = help_ms - pass_ms

    system = platform.system()
    ceiling = _CEILINGS_MS[system]
    assert overhead <= ceiling, (
        f"startup overhead {overhead:.0f}ms > ceiling {ceiling}ms "
        f"on {system} (help={help_ms:.0f}ms, pass={pass_ms:.0f}ms). "
        "See AGENTS.md § Startup latency budget. If this is a CI flake, "
        "re-run; if it persists, the baseline needs a bump or a new "
        "import is a real regression."
    )


# =====================================================================
# Part 2 — Static import audit
# =====================================================================


def _full_lines(path: Path) -> dict[int, str]:
    return {i + 1: line for i, line in enumerate(path.read_text(encoding="utf-8").splitlines())}


def _is_deferred(tree: ast.Module, target_lineno: int) -> bool:
    """True if the import at ``target_lineno`` is inside ``if TYPE_CHECKING``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            is_type_checking = (
                (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                or (
                    isinstance(test, ast.Constant)
                    and test.value is False
                )
            )
            if not is_type_checking:
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Import | ast.ImportFrom):
                    if child.lineno == target_lineno:
                        return True
    return False


@pytest.mark.parametrize(
    "path",
    _STARTUP_PATH_FILES,
    ids=[str(p.relative_to(_SRC_ROOT.parent)) for p in _STARTUP_PATH_FILES],
)
def test_no_unjustified_third_party_imports(path: Path) -> None:
    """Every top-level third-party import needs a ``# cost: ~Nms`` marker.

    The marker keeps the cost visible at the call site — when a new
    import adds 50ms, the reviewer sees it in the diff. Deferring under
    ``TYPE_CHECKING`` (the #45 / #81 / #118 pattern) bypasses the
    requirement entirely.
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = _full_lines(path)

    violations: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        module_name = _import_root(node)
        if module_name not in _THIRD_PARTY_MODULES:
            continue
        if _is_deferred(tree, node.lineno):
            continue
        line = lines.get(node.lineno, "")
        if not _COST_MARKER.search(line):
            violations.append(
                f"{path.relative_to(_SRC_ROOT.parent)}:{node.lineno} "
                f"top-level third-party import {module_name!r} "
                f"without `# cost: ~Nms` marker"
            )
    assert not violations, (
        "Every top-level third-party import in the startup path must "
        "either carry a `# cost: ~Nms` comment or be deferred under "
        "TYPE_CHECKING. See AGENTS.md § Startup latency budget.\n"
        + "\n".join(violations)
    )


def _import_root(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".")[0]
    return (node.module or "").split(".")[0]


# =====================================================================
# Self-test: the audit catches a naked third-party import.
# =====================================================================


def test_audit_catches_naked_third_party_import(tmp_path: Path) -> None:
    """A top-level third-party import without a marker must be flagged."""
    fake = tmp_path / "fake.py"
    fake.write_text("import pydantic\n", encoding="utf-8")
    src = fake.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = _full_lines(fake)
    naked = False
    for node in tree.body:
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        module_name = _import_root(node)
        if module_name not in _THIRD_PARTY_MODULES:
            continue
        if _is_deferred(tree, node.lineno):
            continue
        line = lines.get(node.lineno, "")
        if not _COST_MARKER.search(line):
            naked = True
    assert naked, "audit failed to flag naked `import pydantic`"


def test_audit_accepts_cost_marker(tmp_path: Path) -> None:
    """A top-level third-party import with the marker is allowed."""
    fake = tmp_path / "fake.py"
    fake.write_text("import pydantic  # cost: ~1ms\n", encoding="utf-8")
    src = fake.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = _full_lines(fake)
    naked = False
    for node in tree.body:
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        module_name = _import_root(node)
        if module_name not in _THIRD_PARTY_MODULES:
            continue
        if _is_deferred(tree, node.lineno):
            continue
        line = lines.get(node.lineno, "")
        if not _COST_MARKER.search(line):
            naked = True
    assert not naked, "audit flagged an import that has the marker"
