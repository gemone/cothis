"""Audit every model-facing ``@tool`` description against the standard.

See ``AGENTS.md`` § "Tool description standard" for the 4-point rule.
This test is the pragmatic floor: string-presence checks for the
standard's structural keywords. Option (a) of the issue's trade-off
(AST / LLM-judge rejected — see the section's "Why this test shape").
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

from cothis.tools.core import schema_for

if TYPE_CHECKING:
    from cothis.tools.core import Tool

_AUDITED_MODULES = (
    "cothis.tools.fs.read",
    "cothis.tools.fs.list",
    "cothis.tools.fs.search",
    "cothis.tools.fs.create",
    "cothis.tools.fs.modify",
    "cothis.tools.fs.delete",
    "cothis.skills",
)

_MIN_LEN = 120


def _discover_tools() -> list[tuple[str, Tool, str]]:
    """Collect every ``@tool``-decorated callable from audited modules."""
    out: list[tuple[str, Tool, str]] = []
    for mod_name in _AUDITED_MODULES:
        mod = importlib.import_module(mod_name)
        for attr, obj in vars(mod).items():
            if hasattr(obj, "__cothis_schema__") and hasattr(obj, "__name__"):
                schema = schema_for(obj)  # type: ignore[arg-type]
                name = schema.get("name") or obj.__name__  # type: ignore[union-attr]
                out.append((f"{mod_name}.{attr}", obj, name))  # type: ignore[arg-type]
    return out


_TOOLS = _discover_tools()


@pytest.mark.parametrize(
    "qualified_name,tool,name",
    _TOOLS,
    ids=[t[2] for t in _TOOLS],
)
def test_tool_description_meets_audit_standard(
    qualified_name: str,
    tool: Tool,
    name: str,
) -> None:
    """Floor check — AGENTS.md § Tool description standard (4 points)."""
    desc = schema_for(tool)["description"]
    assert isinstance(desc, str) and desc, f"{name}: description is empty"

    assert len(desc) >= _MIN_LEN, (
        f"{name}: description is {len(desc)} chars, need ≥ {_MIN_LEN}. "
        "See AGENTS.md § Tool description standard."
    )

    short = name.split(".")[-1]
    assert f"{short}(" in desc, (
        f"{name}: description must show a `{short}(...)` example invocation."
    )

    has_example = "Example" in desc or "::" in desc
    assert has_example, f"{name}: description must include a concrete Example block."

    has_return_signal = "→" in desc or "Returns" in desc or "return" in desc.lower()
    assert has_return_signal, (
        f"{name}: description must signal the return shape ('→ ...' or 'Returns ...')."
    )
