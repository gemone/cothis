"""Tests for ``cothis.tui`` (#228).

Covers the 3-pane layout + interactivity API:

- 3 panes exist + are queryable.
- ``ConversationView.append_delta(kind, text)`` routes text vs thinking.
- ``ConversationView.append_tool_call`` mounts an inline card.
- ``InputBar`` accepts multi-line text + clears on send.
- ``action_send_prompt`` echoes the user prompt into the conversation.
- ``append_assistant_delta`` + ``append_tool_call`` forward to the view.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_app_launches_with_three_panes() -> None:
    """Pilot launches CothisApp; all three panes are queryable."""
    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("SessionList") is not None
        assert app.query_one(ConversationView) is not None
        assert app.query_one("InputBar") is not None


@pytest.mark.asyncio
async def test_conversation_view_appends_text_delta() -> None:
    """``append_delta(kind='text', ...)`` accumulates into renderable."""
    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        view.append_delta("text", "hello ")
        view.append_delta("text", "world")
        await pilot.pause()
        assert "hello" in view.renderable_str
        assert "world" in view.renderable_str


@pytest.mark.asyncio
async def test_conversation_view_thinking_delta_does_not_crash() -> None:
    """``append_delta(kind='thinking', ...)`` is accepted without error."""
    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        view.append_delta("thinking", "I should consider...")
        await pilot.pause()
        assert "consider" not in view.renderable_str


@pytest.mark.asyncio
async def test_tool_call_card_mounts() -> None:
    """``append_tool_call`` creates a visible ToolCallCard widget."""
    from cothis.tui import ConversationView, CothisApp, ToolCallCard

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        card = view.append_tool_call("fs.read")
        await pilot.pause()
        assert isinstance(card, ToolCallCard)
        cards = list(view.query("ToolCallCard"))
        assert len(cards) >= 1


@pytest.mark.asyncio
async def test_input_bar_accepts_text() -> None:
    """InputBar holds a TextArea that can hold multi-line content."""
    from cothis.tui import CothisApp, InputBar

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(InputBar)
        bar.set_text("line one\nline two")
        assert bar.get_text() == "line one\nline two"


@pytest.mark.asyncio
async def test_send_prompt_echoes_into_conversation() -> None:
    """``action_send_prompt`` posts the InputBar text to ConversationView + clears."""
    from cothis.tui import ConversationView, CothisApp, InputBar

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(InputBar)
        bar.set_text("what is 2+2?")
        app.action_send_prompt()
        await pilot.pause()
        view = app.query_one(ConversationView)
        assert "what is 2+2?" in view.renderable_str
        assert bar.get_text() == ""


@pytest.mark.asyncio
async def test_send_prompt_ignores_empty_input() -> None:
    """Empty InputBar → no echo, no crash."""
    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        before = view.renderable_str
        app.action_send_prompt()
        await pilot.pause()
        assert view.renderable_str == before


@pytest.mark.asyncio
async def test_append_assistant_delta_routes_to_view() -> None:
    """``app.append_assistant_delta`` forwards to ConversationView."""
    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.append_assistant_delta("text", "answer text")
        await pilot.pause()
        view = app.query_one(ConversationView)
        assert "answer text" in view.renderable_str


@pytest.mark.asyncio
async def test_append_tool_call_via_app() -> None:
    """``app.append_tool_call`` forwards to ConversationView."""
    from cothis.tui import ConversationView, CothisApp, ToolCallCard

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        card = app.append_tool_call("fs.modify")
        await pilot.pause()
        assert isinstance(card, ToolCallCard)


@pytest.mark.asyncio
async def test_ctrl_enter_keypress_sends_prompt() -> None:
    """Ctrl+Enter binding triggers send_prompt via the actual keypress."""
    from cothis.tui import ConversationView, CothisApp, InputBar

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(InputBar)
        bar.set_text("via keypress")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        view = app.query_one(ConversationView)
        assert "via keypress" in view.renderable_str
        assert bar.get_text() == ""


@pytest.mark.asyncio
async def test_user_message_brackets_are_escaped() -> None:
    """Brackets in user text are escaped so Markdown injection is blocked."""
    from cothis.tui import ConversationView, CothisApp, InputBar

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(InputBar)
        bar.set_text("[click](javascript:alert(1))")
        app.action_send_prompt()
        await pilot.pause()
        view = app.query_one(ConversationView)
        assert "\\[click\\]" in view.renderable_str
        assert "[click]" not in view.renderable_str.replace("\\[click\\]", "")
