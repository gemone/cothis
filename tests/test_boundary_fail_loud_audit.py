"""External boundary fail-loud audit (#221).

Regression guard for the two classes of boundary-site bugs the
project has hit:

1. **Third-party non-public attribute read without a shape guard**
   (``group.tools`` on mcp SDK's ``ClientSessionGroup`` — #63).
2. **Silent registry overwrite** (``slash.register`` — #112;
   ``Agent._tool_map`` collision — #112 pattern).

Each known site is parametrised; the test verifies the guard is still
present. New boundary sites enter this registry when they're added to
the codebase — the rule is in ``AGENTS.md`` § External boundary
fail-loud, this test is the regression backstop.

A general static scan for ``obj._foo`` attribute reads is **not** the
right shape for this audit: distinguishing third-party objects from
first-party ones requires cross-module type inference that's noisy in
practice. The registry approach is explicit + cheap + catches the
actual regression mode (guard removed in a refactor).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent


# Registry of known boundary sites. Each entry pins one guard that
# must remain in place. ``kind`` selects the assertion shape:
#   - "shape_guard"     — third-party private attr read protected by
#                         an isinstance/hasattr check + a RuntimeError
#                         or WARNING naming the divergence.
#   - "registry_overwrite" — ``dict[key] = ...`` on a registry dict
#                         protected by a ``key in dict`` collision
#                         check + ``logger.warning``.
_BOUNDARY_SITES = [
    {
        "id": "mcp.group.tools",
        "file": "src/cothis/tools/mcp.py",
        "kind": "shape_guard",
        "boundary": "group.tools",
        "shape_check_regex": r"isinstance\([^)]*tools[^)]*,\s*dict\)",
        "action_regex": r"raise\s+RuntimeError|logger\.warning",
        "issue_ref": "#63",
        "adr_ref": "ADR-0005",
    },
    {
        "id": "slash.register",
        "file": "src/cothis/slash.py",
        "kind": "registry_overwrite",
        "boundary": "_entries",
        "collision_check_regex": r"if\s+name\s+in\s+_entries",
        "warning_regex": r"logger\.warning",
        "issue_ref": "#112",
    },
    {
        "id": "agent._tool_map",
        "file": "src/cothis/agent.py",
        "kind": "registry_overwrite",
        "boundary": "self._tool_map",
        "collision_check_regex": r"if\s+key\s+in\s+self\._tool_map",
        "warning_regex": r"logger\.warning",
        "issue_ref": "#112",
    },
]


@pytest.mark.parametrize(
    "site",
    _BOUNDARY_SITES,
    ids=[s["id"] for s in _BOUNDARY_SITES],
)
def test_boundary_site_keeps_its_guard(site: dict) -> None:
    """Pin each known guard in place — removal turns this test red."""
    src = (_REPO_ROOT / site["file"]).read_text(encoding="utf-8")
    assert site["boundary"] in src, (
        f"{site['file']}: boundary site {site['boundary']!r} not found — "
        f"if it moved or was renamed, update the registry in this test."
    )
    if site["kind"] == "shape_guard":
        _assert_shape_guard(src, site)
    elif site["kind"] == "registry_overwrite":
        _assert_registry_overwrite_guard(src, site)
    else:  # pragma: no cover
        raise AssertionError(f"unknown site kind: {site['kind']}")


def _assert_shape_guard(src: str, site: dict) -> None:
    """Verify an isinstance/hasattr check + an explicit raise/warning."""
    shape_re = site["shape_check_regex"]
    action_re = site["action_regex"]
    assert re.search(shape_re, src), (
        f"{site['file']}: shape check for {site['boundary']!r} "
        f"({site['issue_ref']}) was removed. A third-party private "
        f"attr read without a shape check silently breaks on SDK "
        f"upgrades. See AGENTS.md § External boundary fail-loud."
    )
    assert re.search(action_re, src), (
        f"{site['file']}: fail-loud action (RuntimeError or "
        f"logger.warning) for {site['boundary']!r} ({site['issue_ref']}) "
        f"was removed. See AGENTS.md § External boundary fail-loud."
    )


def _assert_registry_overwrite_guard(src: str, site: dict) -> None:
    """Verify a collision check + ``logger.warning`` on the overwrite path."""
    collision_re = site["collision_check_regex"]
    warning_re = site["warning_regex"]
    assert re.search(collision_re, src), (
        f"{site['file']}: collision check for registry "
        f"{site['boundary']!r} ({site['issue_ref']}) was removed. "
        f"Silent overwrite of prior registration is the bug."
    )
    assert re.search(warning_re, src), (
        f"{site['file']}: logger.warning on {site['boundary']!r} "
        f"collision ({site['issue_ref']}) was removed."
    )


# ---------------------------------------------------------------------
# Self-tests for the regex shape
# ---------------------------------------------------------------------


def test_shape_guard_regex_matches_isinstance_dict() -> None:
    """``isinstance(x, dict)`` triggers the shape_check_regex."""
    assert re.search(
        r"isinstance\([^)]*tools[^)]*,\s*dict\)",
        "if not isinstance(tools_attr, dict):",
    )


def test_collision_check_regex_matches_if_in() -> None:
    """``if key in self._tool_map:`` triggers the collision regex."""
    assert re.search(
        r"if\s+key\s+in\s+self\._tool_map",
        "    if key in self._tool_map:",
    )
