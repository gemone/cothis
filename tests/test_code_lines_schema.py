"""Tests for ``code.lines`` demo tool schema description (#203).

Same audit standard as #190/#192/#194/#196/#199/#201: the model-facing
description must include the return-format (``{total, blank, comment, code}``)
and a concrete example.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

from cothis.tools import schema_for

if TYPE_CHECKING:
    from typing import Any

_DEMO_TOOL = Path(__file__).resolve().parent.parent / ".agents" / "tools" / "code_lines.py"


def _load_tool() -> Any:
    """Import the demo tool from ``.agents/tools/code_lines.py``."""
    spec = importlib.util.spec_from_file_location("code_lines", _DEMO_TOOL)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.count_lines


def _schema() -> dict:
    return schema_for(_load_tool())


def test_description_mentions_return_dict_shape() -> None:
    """Description tells the model the return has 4 fields."""
    desc = _schema().get("description", "")
    lowered = desc.lower()
    assert "total" in lowered and "blank" in lowered
    assert "comment" in lowered and "code" in lowered


def test_description_has_example_invocation() -> None:
    """Description includes a concrete ``code.lines(...)`` example."""
    desc = _schema().get("description", "")
    assert "code.lines(" in desc
