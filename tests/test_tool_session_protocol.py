"""Tests for tool protocol extension — .session handler + inject_session (#157)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from cothis.tools import tool
from cothis.tools.core import ToolDef

if TYPE_CHECKING:
    from cothis.session import Session


def test_inject_session_strips_param_from_schema() -> None:
    """``_session`` parameter is absent from the LLM schema."""

    @tool(inject_session=True)
    def my_tool(name: str, _session: Any) -> str:
        """Do thing.

        Args:
            name: a name.
        """
        return "ok"

    schema = my_tool.__cothis_schema__
    props = schema["input_schema"]["properties"]
    assert "name" in props
    assert "_session" not in props
    assert schema["input_schema"]["required"] == ["name"]


def test_underscore_param_always_stripped() -> None:
    """Any param starting with ``_`` is stripped from the schema."""

    @tool
    def tool_with_private(x: str, _internal: int = 0) -> str:
        """Tool.

        Args:
            x: public.
        """
        return x

    props = tool_with_private.__cothis_schema__["input_schema"]["properties"]
    assert "x" in props
    assert "_internal" not in props


def test_inject_session_flag_set() -> None:
    """``inject_session=True`` sets the flag on ToolDef."""

    @tool(inject_session=True)
    def t(name: str, _session: Any) -> str:
        """T.

        Args:
            name: n.
        """
        return name

    assert t._inject_session is True


def test_skill_marker_flag_set() -> None:
    """``skill_marker=True`` sets the flag on ToolDef."""

    @tool(skill_marker=True)
    def t(name: str) -> str:
        """T.

        Args:
            name: n.
        """
        return name

    assert t._skill_marker is True


def test_session_decorator_registers_handler() -> None:
    """``@tool_func.session`` registers handler + implies inject_session."""
    seen: list[Any] = []

    @tool
    def loader(name: str, _session: Any) -> str:
        """Load.

        Args:
            name: skill name.
        """
        return f"loaded {name}"

    @loader.session
    def handler(session: Any, result: str, args: dict[str, Any]) -> None:
        seen.append((result, args.get("name")))

    assert loader._inject_session is True
    assert loader._session_handler is handler

    # Simulate a call through the tool body.
    loader(name="deploy", _session=None)
    # Handler hasn't run yet — it's invoked by _execute_tool, not by __call__.
    assert seen == []


def test_session_handler_receives_session_result_args() -> None:
    """Handler gets (session, result, args) when called from _execute_tool."""
    from unittest.mock import MagicMock

    import any_llm

    handler_calls: list[tuple[Any, str, dict[str, Any]]] = []

    @tool(inject_session=True)
    def load_skill(name: str, _session: Any) -> str:
        """Load skill.

        Args:
            name: skill name.
        """
        return f"<skill_content>{name}</skill_content>"

    @load_skill.session
    def on_load(session: Any, result: str, args: dict[str, Any]) -> None:
        handler_calls.append((session, result, args))

    # Build a minimal agent to drive _execute_tool.
    monkeypatch_target = MagicMock()
    monkeypatch_target.setattr = lambda *a, **kw: None

    # Can't easily build an Agent without API keys; test the protocol
    # by calling the tool + handler directly (same as _execute_tool does).
    fake_session = MagicMock()
    args = {"name": "deploy", "_session": fake_session}
    result = load_skill(**args)
    handler = load_skill._session_handler
    if handler is not None:
        handler(fake_session, result, {"name": "deploy"})

    assert len(handler_calls) == 1
    assert handler_calls[0][0] is fake_session
    assert "deploy" in handler_calls[0][1]
    assert handler_calls[0][2]["name"] == "deploy"


def test_default_tool_has_no_session_flags() -> None:
    """Plain ``@tool`` doesn't set inject_session or skill_marker."""

    @tool
    def plain(x: str) -> str:
        """Plain.

        Args:
            x: x.
        """
        return x

    assert plain._inject_session is False
    assert plain._session_handler is None
    assert plain._skill_marker is False
