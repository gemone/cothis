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


@pytest.mark.asyncio
async def test_tool_call_flushes_text_segment_for_dom_order() -> None:
    """Card mount flushes the active text segment so DOM order matches event order.

    Regression guard for the Rule 3 violation on #228: without the
    flush, all text accumulated in one Markdown widget and a card
    mounted between two deltas rendered below both. With the flush,
    each card finalises the current segment; the next delta starts a
    fresh Markdown widget below the card.
    """
    from textual.widgets import Markdown

    from cothis.tui import ConversationView, CothisApp, ToolCallCard

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        view.append_delta("text", "before-card")
        await pilot.pause()
        view.append_tool_call("fs.read")
        await pilot.pause()
        view.append_delta("text", "after-card")
        await pilot.pause()

        # Two separate Markdown segments exist (one per text span).
        markdowns = list(view.query(Markdown))
        assert len(markdowns) == 2

        # The card sits between them in DOM order — verify by walking
        # immediate children of the ConversationView scroll container.
        children = list(view.children)
        positions_md = [i for i, c in enumerate(children) if isinstance(c, Markdown)]
        positions_card = [i for i, c in enumerate(children) if isinstance(c, ToolCallCard)]
        assert len(positions_card) == 1
        assert positions_md[0] < positions_card[0] < positions_md[1]


@pytest.mark.asyncio
async def test_append_delta_segmented_streaming_is_linear_not_quadratic() -> None:
    """AC #267: doubling N appends with periodic flushes ≤ 2× wall time, not 4×.

    The hot path is ``append_delta`` between tool calls (which flush the
    segment). Without Fix A (``list[str]`` accumulator) each ``+=`` was
    O(buffer_size); without Pattern 2 (segment flush on tool_call) the
    buffer grew unbounded across the whole turn. Combined, each segment
    is bounded by ``deltas_per_segment × chunk_size``, so total work is
    linear in N × chunk_size.

    The test fails on pre-fix code: ``str`` accumulator + full re-parse
    per call gives O(N²) within a segment, and segments never reset.
    """
    import time

    from cothis.tui import ConversationView, CothisApp

    async def _run(n_deltas: int, flush_every: int) -> float:
        app = CothisApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            view = app.query_one(ConversationView)
            t0 = time.perf_counter()
            for i in range(n_deltas):
                view.append_delta("text", f"line{i}\n")
                if (i + 1) % flush_every == 0:
                    view.append_tool_call("fs.read")
            await pilot.pause()
            return time.perf_counter() - t0

    # Two workloads: 200 deltas vs 400 deltas, both flushing every 20.
    # Doubling N should ≤ 2× wall time if the work is linear in N.
    t_small = await _run(200, flush_every=20)
    t_large = await _run(400, flush_every=20)
    ratio = t_large / t_small if t_small > 0 else float("inf")
    assert ratio <= 2.5, (
        f"expected ≤2.5× slowdown on 2× workload (linear); got {ratio:.2f}× "
        f"(small={t_small*1000:.1f}ms, large={t_large*1000:.1f}ms) — "
        f"buffer is accumulating O(N²) work somewhere"
    )
