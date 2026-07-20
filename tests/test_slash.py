"""Tests for ``cothis.slash`` — chat REPL slash command framework.

A small dispatch registry: name → async handler. Lines starting with
``/`` in the chat REPL go through here instead of the agent loop.
Unknown commands produce a local error listing available commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from cothis.slash import SlashContext, SlashRegistry

if TYPE_CHECKING:
    from cothis.session import Session

# ---------------------------------------------------------------------
# registry + dispatch
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_known_command_dispatches_to_handler() -> None:
    """A registered command's handler runs and its return value surfaces."""
    reg = SlashRegistry()
    seen: dict[str, object] = {}

    async def hello(ctx: SlashContext) -> str:
        seen["called"] = True
        return "hi back"

    reg.register("hello", hello, summary="say hi")
    result = await reg.dispatch("/hello")
    assert seen["called"] is True
    assert result == "hi back"


@pytest.mark.asyncio
async def test_known_command_with_args_passes_to_handler() -> None:
    """Arguments after the command name reach the handler as a single string."""
    reg = SlashRegistry()
    received: dict[str, str] = {}

    async def echo(ctx: SlashContext, args: str) -> str:
        received["args"] = args
        return f"echo: {args}"

    reg.register("echo", echo)
    result = await reg.dispatch("/echo foo bar baz")
    assert received["args"] == "foo bar baz"
    assert result == "echo: foo bar baz"


@pytest.mark.asyncio
async def test_unknown_command_lists_available() -> None:
    """Unknown ``/cmd`` produces a local error message listing registered commands."""
    reg = SlashRegistry()
    reg.register("hello", _noop_handler, summary="say hi")
    reg.register("exit", _noop_handler, summary="exit the REPL")

    result = await reg.dispatch("/bogus arg")
    assert result is not None

    assert "unknown" in result.lower() or "not found" in result.lower()
    assert "/bogus" in result
    assert "hello" in result
    assert "exit" in result
    # Not an LLM call — handler must not run.
    assert "say hi" in result  # summary text leaks into the listing


@pytest.mark.asyncio
async def test_registry_empty_lists_nothing() -> None:
    """Empty registry + unknown command → empty listing, not a crash."""
    reg = SlashRegistry()
    result = await reg.dispatch("/whatever")
    assert result is not None
    assert "unknown" in result.lower() or "no commands" in result.lower()


@pytest.mark.asyncio
async def test_help_lists_all_commands_with_summaries() -> None:
    """/help returns one line per registered command with each summary."""
    reg = SlashRegistry()
    reg.register("hello", _noop_handler, summary="say hi")
    reg.register("exit", _noop_handler, summary="exit the REPL")
    result = await reg.dispatch("/help")
    assert result is not None
    assert "/hello" in result
    assert "say hi" in result
    assert "/exit" in result
    assert "exit the REPL" in result


@pytest.mark.asyncio
async def test_handler_receives_session_context() -> None:
    """The handler context carries the session the REPL is attached to."""
    reg = SlashRegistry()
    seen: dict[str, object] = {}

    async def show_session(ctx: SlashContext) -> str:
        seen["session"] = ctx.session
        return "ok"

    reg.register("show", show_session)
    sentinel = cast("Session | None", object())  # stand-in for a real Session
    result = await reg.dispatch("/show", ctx=SlashContext(session=sentinel))
    assert seen["session"] is sentinel
    assert result == "ok"


@pytest.mark.asyncio
async def test_non_slash_input_returns_none() -> None:
    """The REPL passes every line through ``dispatch``; non-``/`` lines
    return ``None`` so the REPL knows to send it to the agent instead."""
    reg = SlashRegistry()
    reg.register("hello", _noop_handler)
    assert await reg.dispatch("hello there") is None
    assert await reg.dispatch("") is None
    assert await reg.dispatch("  /not-a-slash  ") is None  # leading whitespace


@pytest.mark.asyncio
async def test_register_overwrites_prior_handler() -> None:
    """Re-registering the same name replaces silently — handlers grow over time."""
    reg = SlashRegistry()
    calls: list[str] = []

    async def first(ctx: SlashContext) -> str:
        calls.append("first"); return "1"

    async def second(ctx: SlashContext) -> str:
        calls.append("second"); return "2"

    reg.register("cmd", first)
    reg.register("cmd", second)
    result = await reg.dispatch("/cmd")
    assert result == "2"
    assert calls == ["second"]


@pytest.mark.asyncio
async def test_handler_returning_none_prints_nothing() -> None:
    """Handler returning ``None`` → registry returns ``""`` (REPL prints nothing)."""
    reg = SlashRegistry()

    async def quiet(ctx: SlashContext) -> None:
        return None

    reg.register("quiet", quiet)
    result = await reg.dispatch("/quiet")
    assert result == ""


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


async def _noop_handler(ctx: SlashContext) -> str:
    return "noop"
