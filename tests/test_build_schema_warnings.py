"""Tests for ``cothis.tools.core._build_schema`` warning surfacing (#80).

When a user-authored tool has a typo in a type hint, the previous
behaviour silently coerced the parameter to ``string`` in the LLM
schema with no log line. #80 adds a load-time WARNING so the typo is
discoverable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from cothis.tools.core import _build_schema

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _typo_hint_tool() -> Any:
    """Return a function whose ``path`` parameter has an unresolvable
    hint (``"Paht.Typo"`` — a dotted name that doesn't exist). Quoted
    annotation survives def time and only blows up at hint resolution.
    """
    def read(
        path: "Paht.Typo",  # type: ignore[name-defined, unresolved-reference]  # noqa: F821, UP037  # ty:ignore[unresolved-reference]
    ) -> str:
        """Read a file.

        Args:
            path: File to read.
        """
        return ""

    return read


def test_unresolvable_type_hint_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Hint-resolution failure emits a WARNING naming the tool + cause."""
    fn = _typo_hint_tool()
    with caplog.at_level(logging.WARNING, logger="cothis.tools.core"):
        schema = _build_schema(fn, "read", None)
    # Schema still falls back to ``string`` — the warning is additive.
    prop = schema["input_schema"]["properties"]["path"]
    assert prop["type"] == "string"
    # The warning names the tool and mentions the fallback.
    warning_text = " ".join(r.message for r in caplog.records)
    assert "read" in warning_text
    assert "type hint" in warning_text or "could not resolve" in warning_text


def test_per_parameter_schema_failure_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A parameter whose annotation TypeAdapter can't handle logs a WARNING."""

    def tool_one(
        x: "Does.Not.Exist",  # type: ignore[name-defined, unresolved-reference]  # noqa: F821, UP037  # ty:ignore[unresolved-reference]
    ) -> str:
        """Tool.

        Args:
            x: thing.
        """
        return ""

    with caplog.at_level(logging.WARNING, logger="cothis.tools.core"):
        schema = _build_schema(tool_one, "tool_one", None)
    prop = schema["input_schema"]["properties"]["x"]
    assert prop["type"] == "string"
    warning_text = " ".join(r.message for r in caplog.records)
    assert "tool_one" in warning_text
    assert "x" in warning_text or "parameter" in warning_text


def test_clean_tool_emits_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A well-typed tool doesn't trip the warning."""

    def good(x: int) -> str:
        """Good.

        Args:
            x: int.
        """
        return ""

    with caplog.at_level(logging.WARNING, logger="cothis.tools.core"):
        _build_schema(good, "good", None)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []
