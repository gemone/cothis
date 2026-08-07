"""Tests for ``cothis.tui`` (#228).

Covers the 3-pane layout + interactivity API:

- 3 panes exist + are queryable.
- ``ConversationView.append_delta(kind, text)`` routes text vs thinking.
- ``ConversationView.append_tool_call`` mounts an inline card.
- the input ``TextArea`` accepts multi-line text + clears on send.
- ``action_send_prompt`` echoes the user prompt into the conversation.
- ``append_assistant_delta`` + ``append_tool_call`` forward to the view.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
        assert app.query_one("#input") is not None


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
    """The input TextArea holds multi-line content."""
    from textual.widgets import TextArea

    from cothis.tui import CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#input", TextArea)
        bar.text = "line one\nline two"
        assert bar.text == "line one\nline two"


@pytest.mark.asyncio
async def test_send_prompt_echoes_into_conversation() -> None:
    """``action_send_prompt`` posts the input text to ConversationView + clears."""
    from textual.widgets import TextArea

    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#input", TextArea)
        bar.text = "what is 2+2?"
        await app.action_send_prompt()
        await pilot.pause()
        view = app.query_one(ConversationView)
        assert "what is 2+2?" in view.renderable_str
        assert bar.text == ""


@pytest.mark.asyncio
async def test_send_prompt_ignores_empty_input() -> None:
    """Empty input → no echo, no crash."""
    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        before = view.renderable_str
        await app.action_send_prompt()
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
    from textual.widgets import TextArea

    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#input", TextArea)
        bar.text = "via keypress"
        await pilot.press("ctrl+enter")
        await pilot.pause()
        view = app.query_one(ConversationView)
        assert "via keypress" in view.renderable_str
        assert bar.text == ""


@pytest.mark.asyncio
async def test_user_message_brackets_are_escaped() -> None:
    """Brackets in user text are escaped so Markdown injection is blocked."""
    from textual.widgets import TextArea

    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#input", TextArea)
        bar.text = "[click](javascript:alert(1))"
        await app.action_send_prompt()
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
        # The streaming segment stays a plain Static until finalised (#407);
        # finalise it so both text spans are Markdown widgets, as the idle
        # timer would after streaming settles.
        view._finalize_segment()
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
    """AC #267: doubling N appends with periodic flushes ≤ 3.5× wall time, not 4×.

    The hot path is ``append_delta`` between tool calls (which flush the
    segment). Without Fix A (``list[str]`` accumulator) each ``+=`` was
    O(buffer_size); without Pattern 2 (segment flush on tool_call) the
    buffer grew unbounded across the whole turn. Combined, each segment
    is bounded by ``deltas_per_segment × chunk_size``, so total work is
    linear in N × chunk_size.

    A 4× workload (400 vs 1600 deltas) widens the gap between a healthy
    linear implementation (~4-6×) and a true O(N²) regression (~16×), so
    the threshold has wide headroom for CI runner variance + Textual's
    per-widget layout overhead without flaking, while still catching
    quadratic accumulation.
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

    # Two workloads: 400 deltas vs 1600 deltas (4× workload), flushing
    # every 50. A 4× workload separates linear (~4-6×) from O(N²) (~16×)
    # far more clearly than 2×, so CI runner variance can't push a healthy
    # linear impl over the threshold.
    t_small = await _run(400, flush_every=50)
    t_large = await _run(1600, flush_every=50)
    ratio = t_large / t_small if t_small > 0 else float("inf")
    assert ratio <= 8.0, (
        f"expected ≤8.0× slowdown on 4× workload (linear + overhead); "
        f"got {ratio:.2f}× (small={t_small*1000:.1f}ms, large={t_large*1000:.1f}ms) — "
        f"buffer is accumulating O(N²) work somewhere"
    )


@pytest.mark.asyncio
async def test_streaming_parses_markdown_once_per_segment_not_per_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#407: Markdown is parsed ONCE per segment (at finalise), not per delta.

    The bug re-parsed the whole segment on every text delta — O(S²) in the
    segment size. After the fix, deltas accumulate into a plain Static and
    ``Markdown(source)`` runs once at finalise. This guards the quadratic
    directly: N deltas → 1 parse (pre-fix it was N).
    """
    import cothis.tui as tui_mod
    from cothis.tui import ConversationView, CothisApp

    # Intercept tui's ``Markdown(source)`` construction to count parses.
    # A factory (not an ``__init__`` patch) keeps the signature precise so ty
    # can verify it — tui only ever constructs ``Markdown(source)``.
    parses: list[None] = []
    real_markdown = tui_mod.Markdown

    def counting_markdown(markdown: str | None = None) -> tui_mod.Markdown:
        parses.append(None)
        return real_markdown(markdown)

    monkeypatch.setattr(tui_mod, "Markdown", counting_markdown)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        for i in range(500):
            view.append_delta("text", f"word{i} ")
        await pilot.pause()
        # While streaming: NO Markdown parse (plain Static holds the text).
        assert not parses, (
            f"streaming parsed Markdown {len(parses)} time(s); expected 0"
        )
        view._finalize_segment()  # idle-timer proxy
        await pilot.pause()

    assert len(parses) == 1, (
        f"expected exactly 1 Markdown parse per segment; got {len(parses)} "
        f"(pre-fix re-parsed per delta)"
    )


@pytest.mark.asyncio
async def test_streaming_one_segment_refresh_time_is_linear_not_quadratic() -> None:
    """#407 acceptance #2: total refresh time for one growing segment is O(S), not O(S²).

    A pure-text answer with no tool calls is the worst case — one segment
    growing across every delta. Pre-fix each delta re-parsed the whole segment
    (Σ O(k·d) = O(S²)); post-fix deltas accumulate O(1) and Markdown is parsed
    once at finalise (O(S)). Doubling the deltas should ≈ double the time, not
    quadruple it. A warmup + best-of-two per size absorb CI contention; the 3.5×
    threshold mirrors the #267 guard and separates post-fix (~2×) from a pre-fix
    O(S²) regression (~4×). The deterministic guard is
    ``test_streaming_parses_markdown_once_per_segment_not_per_delta``.
    """
    import time

    from cothis.tui import ConversationView, CothisApp

    async def _run(n: int) -> float:
        app = CothisApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            view = app.query_one(ConversationView)
            start = time.perf_counter()
            for i in range(n):
                view.append_delta("text", f"word{i} ")
            view._finalize_segment()
            await pilot.pause()
            return time.perf_counter() - start

    await _run(50)  # warmup: imports, widget caches, allocator

    async def _best_of_two(n: int) -> float:
        return min(await _run(n), await _run(n))

    t_small = await _best_of_two(200)
    t_large = await _best_of_two(400)
    ratio = t_large / t_small if t_small > 0 else float("inf")
    assert ratio <= 3.5, (
        f"expected ≤3.5× on 2× deltas (linear + overhead); got {ratio:.2f}× "
        f"(small={t_small * 1000:.1f}ms, large={t_large * 1000:.1f}ms) — "
        f"per-delta cost is growing with segment size (O(S²))"
    )


@pytest.mark.asyncio
async def test_streaming_defers_markdown_until_finalize() -> None:
    """#407: a streaming segment renders plain text (no Markdown) until
    finalise swaps in a single Markdown widget."""
    from textual.widgets import Markdown

    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        for chunk in ("hello ", "world"):
            view.append_delta("text", chunk)
        await pilot.pause()
        # While streaming: no Markdown yet; the text is buffered.
        assert list(view.query(Markdown)) == []
        assert view.renderable_str == "hello world"
        # Finalise (idle-timer proxy): one Markdown widget mounts.
        view._finalize_segment()
        await pilot.pause()
        assert len(list(view.query(Markdown))) == 1


@pytest.mark.asyncio
async def test_streaming_auto_follows_new_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#409: while streaming, the view follows the growing content — the
    newest text stays in view without manual scrolling (acceptance #1, #3).

    ``_STREAM_REFRESH_S`` is zeroed so every delta updates the Static
    (deterministic — no throttle timing); a tall segment forces overflow so
    the scroll range is non-zero.
    """
    import cothis.tui as tui_mod
    from cothis.tui import ConversationView, CothisApp

    monkeypatch.setattr(tui_mod, "_STREAM_REFRESH_S", 0.0)

    app = CothisApp()
    async with app.run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        for i in range(60):
            view.append_delta("text", f"line {i}\n")
            await pilot.pause()
        # Content overflowed the viewport + the view auto-pinned to the bottom.
        assert view.max_scroll_y > 0, "expected content to overflow the viewport"
        assert view.scroll_y >= view.max_scroll_y - 1, (
            f"auto-follow did not pin to the bottom; scroll_y={view.scroll_y} "
            f"max={view.max_scroll_y}"
        )


@pytest.mark.asyncio
async def test_auto_follow_does_not_yank_user_who_scrolled_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#409 acceptance #2: a user who scrolled up to read earlier output is
    NOT yanked back to the bottom when the next content arrives."""
    import cothis.tui as tui_mod
    from cothis.tui import ConversationView, CothisApp

    monkeypatch.setattr(tui_mod, "_STREAM_REFRESH_S", 0.0)

    app = CothisApp()
    async with app.run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        # Stream (deltas arrive over time, as in real use) + finalise → pinned.
        for i in range(60):
            view.append_delta("text", f"line {i}\n")
            await pilot.pause()
        view._finalize_segment()
        await pilot.pause()
        assert view.scroll_y >= view.max_scroll_y - 1  # pinned

        # User scrolls UP to read earlier output.
        view.scroll_y = 0
        await pilot.pause()
        scrolled_to = view.scroll_y

        # More content arrives while the user is reading up top.
        for i in range(60):
            view.append_delta("text", f"more {i}\n")
            await pilot.pause()
        view._finalize_segment()
        await pilot.pause()

        # NOT yanked back down — stays where the user scrolled (allowing a
        # 1-line float tolerance from layout clamping).
        assert view.scroll_y <= scrolled_to + 1, (
            f"user scrolled up to {scrolled_to} but was yanked to "
            f"scroll_y={view.scroll_y}"
        )


# ---------------------------------------------------------------------
# WS attach (#252 item 1) — caller supplies URI + bearer token; the
# app opens a client, pumps inbound frames to ConversationView, and
# exposes ``send_run_turn`` for outbound prompts. Tests drive the
# pump via a fake ``websockets.connect`` that yields scripted frames.
# ---------------------------------------------------------------------


class _FakeWS:
    """In-memory stand-in for a ``websockets.WebSocketClientProtocol``.

    Exposes ``send`` (records outbound) + async iteration (yields
    scripted inbound frames) + ``close``. No socket, no I/O.
    """

    def __init__(self, inbound: list[str]) -> None:
        self._inbound = list(inbound)
        self.sent: list[str] = []
        self.closed = False

    async def __aenter__(self) -> _FakeWS:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> _FakeWS:
        return self

    async def __anext__(self):
        if not self._inbound:
            raise StopAsyncIteration
        return self._inbound.pop(0)


@pytest.mark.asyncio
async def test_attach_ws_dispatches_assistant_delta_to_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #252 item 1: inbound ``assistant_delta`` lands in ConversationView."""
    import json as _json

    from cothis.tui import ConversationView, CothisApp

    frames = [
        _json.dumps({"type": "assistant_delta", "kind": "text", "text": "hello "}),
        _json.dumps({"type": "assistant_delta", "kind": "text", "text": "world"}),
    ]
    fake = _FakeWS(frames)

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        assert uri == "ws://fake/agent"
        assert kw.get("additional_headers") == {"Authorization": "Bearer tok"}
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        # Drain the pump task: it yields one frame per ``pilot.pause`` tick.
        for _ in range(len(frames) + 1):
            await pilot.pause()
        view = app.query_one(ConversationView)
        assert "hello" in view.renderable_str
        assert "world" in view.renderable_str
        await app.detach_ws()
        await pilot.pause()


@pytest.mark.asyncio
async def test_attach_ws_dispatches_tool_call_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #252 item 1: inbound ``tool_call_started`` mounts a ToolCallCard."""
    import json as _json

    from cothis.tui import CothisApp, ToolCallCard

    frames = [
        _json.dumps({"type": "tool_call_started", "tool": "fs.read"}),
    ]
    fake = _FakeWS(frames)

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        for _ in range(len(frames) + 1):
            await pilot.pause()
        cards = list(app.query(ToolCallCard))
        assert len(cards) == 1
        await app.detach_ws()
        await pilot.pause()


@pytest.mark.asyncio
async def test_send_run_turn_forwards_prompt_when_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #252 item 1: ``send_run_turn`` writes a JSON control message over WS."""
    import json as _json

    from cothis.tui import CothisApp

    fake = _FakeWS([])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        await pilot.pause()
        await app.send_run_turn("what is 2+2?")
        # ``send`` is async on the WS; the bytes are already on the fake's
        # ``sent`` list before the pump task drains.
        assert len(fake.sent) == 1
        assert _json.loads(fake.sent[0]) == {
            "type": "run_turn",
            "prompt": "what is 2+2?",
        }
        await app.detach_ws()
        await pilot.pause()


@pytest.mark.asyncio
async def test_detach_ws_closes_connection_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #252 item 1: detach closes the WS; double-detach is a safe no-op."""
    from cothis.tui import CothisApp

    fake = _FakeWS([])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        await pilot.pause()
        await app.detach_ws()
        assert fake.closed
        assert app._ws is None
        # Double-detach must not raise.
        await app.detach_ws()
        await pilot.pause()


@pytest.mark.asyncio
async def test_action_send_prompt_forwards_run_turn_when_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #252 item 3: ``action_send_prompt`` forwards a run_turn when WS attached.

    Local echo still renders the user's prompt; ``run_turn`` lands on
    the WS as a JSON control message with the same prompt payload.
    """
    import json as _json

    from textual.widgets import TextArea

    from cothis.tui import ConversationView, CothisApp

    fake = _FakeWS([])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        await pilot.pause()
        bar = app.query_one("#input", TextArea)
        bar.text = "what is 2+2?"
        await app.action_send_prompt()
        await pilot.pause()

        # Local echo: user prompt lands in the view.
        view = app.query_one(ConversationView)
        assert "what is 2+2?" in view.renderable_str
        # Bar cleared.
        assert bar.text == ""
        # Outbound: run_turn control message on the WS.
        assert len(fake.sent) == 1
        assert _json.loads(fake.sent[0]) == {
            "type": "run_turn",
            "prompt": "what is 2+2?",
        }
        await app.detach_ws()
        await pilot.pause()


@pytest.mark.asyncio
async def test_action_send_prompt_no_forwarding_when_not_attached() -> None:
    """AC #252 item 3: when no WS, action_send_prompt is local-echo only.

    Pre-attach behaviour preserved: no run_turn is sent anywhere (there
    is nowhere to send to); the user still sees their prompt.
    """
    from textual.widgets import TextArea

    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#input", TextArea)
        bar.text = "hello"
        await app.action_send_prompt()
        await pilot.pause()
        view = app.query_one(ConversationView)
        assert "hello" in view.renderable_str
        assert bar.text == ""
        # No WS attached → ``send_run_turn`` is a no-op. The view's
        # content matches exactly the local echo (no extra frames).
        assert app._ws is None


@pytest.mark.asyncio
async def test_tool_call_result_pointer_updates_card_status_by_call_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #252 item 4: result frame flips the matching card's badge by call_id.

    Two ``tool_call_started`` frames with different call_ids mount two
    cards. A ``tool_call_result_pointer`` frame with the second call_id
    flips ONLY the second card's status — pairing by call_id, not by
    tool name (which would be ambiguous if both ran the same tool).
    """
    import json as _json

    from cothis.tui import CothisApp, ToolCallCard

    frames = [
        _json.dumps({
            "type": "tool_call_started",
            "tool": "fs.read",
            "arguments": {"path": "a.py"},
            "call_id": "tu_first",
        }),
        _json.dumps({
            "type": "tool_call_started",
            "tool": "fs.read",
            "arguments": {"path": "b.py"},
            "call_id": "tu_second",
        }),
        _json.dumps({
            "type": "tool_call_result_pointer",
            "tool": "fs.read",
            "is_error": False,
            "duration_ms": 5,
            "pointer": "session:s:tool:tu_second",
            "call_id": "tu_second",
        }),
    ]
    fake = _FakeWS(frames)

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        for _ in range(len(frames) + 1):
            await pilot.pause()

        cards = list(app.query(ToolCallCard))
        assert len(cards) == 2
        # Pairing by call_id, not by index — find the card with tu_second.
        by_call_id = {c._call_id: c for c in cards}
        assert "tu_first" in by_call_id
        assert "tu_second" in by_call_id
        # First card untouched (still "running"); second flipped to "done".
        assert by_call_id["tu_first"]._status == "running"
        assert by_call_id["tu_second"]._status == "done"
        await app.detach_ws()
        await pilot.pause()


@pytest.mark.asyncio
async def test_tool_call_result_pointer_error_flips_card_to_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #252 item 4: ``is_error=True`` flips the card to ``failed``."""
    import json as _json

    from cothis.tui import CothisApp, ToolCallCard

    frames = [
        _json.dumps({
            "type": "tool_call_started",
            "tool": "fs.read",
            "arguments": {"path": "a.py"},
            "call_id": "tu_err",
        }),
        _json.dumps({
            "type": "tool_call_result_pointer",
            "tool": "fs.read",
            "is_error": True,
            "duration_ms": 5,
            "pointer": None,
            "call_id": "tu_err",
        }),
    ]
    fake = _FakeWS(frames)

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        for _ in range(len(frames) + 1):
            await pilot.pause()

        cards = list(app.query(ToolCallCard))
        assert len(cards) == 1
        assert cards[0]._status == "failed"
        await app.detach_ws()
        await pilot.pause()


@pytest.mark.asyncio
async def test_tool_call_result_pointer_without_call_id_is_no_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #252 item 4: result frame missing call_id is dropped (no crash).

    Guards against legacy workers that haven't been updated to emit
    ``call_id`` — the TUI must not KeyError on ``_cards_by_call_id``.
    """
    import json as _json

    from cothis.tui import CothisApp, ToolCallCard

    frames = [
        _json.dumps({
            "type": "tool_call_started",
            "tool": "fs.read",
            "arguments": {"path": "a.py"},
            "call_id": "tu_x",
        }),
        # Result frame without call_id — legacy shape.
        _json.dumps({
            "type": "tool_call_result_pointer",
            "tool": "fs.read",
            "is_error": False,
            "duration_ms": 5,
            "pointer": None,
        }),
    ]
    fake = _FakeWS(frames)

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        for _ in range(len(frames) + 1):
            await pilot.pause()

        # Card stays in "running" because the result had no call_id to pair.
        cards = list(app.query(ToolCallCard))
        assert len(cards) == 1
        assert cards[0]._status == "running"
        await app.detach_ws()
        await pilot.pause()


# ---------------------------------------------------------------------
# Session list populate (#252 item 5 — list pane half; selection
# handling is a separate slice).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_session_list_populates_from_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #252 item 5 (list): ``refresh_session_list`` shows sessions visible from cwd.

    Seeds a Storage DB with two sessions (one matching the test cwd,
    one in an unrelated directory), then calls refresh_session_list.
    Only the cwd-visible session appears in SessionList.
    """
    from textual.widgets import ListItem

    from cothis.session import Session
    from cothis.tui import CothisApp, SessionList

    db_path = tmp_path / "session.db"

    # Visible session: cwd matches test's tmp_path.
    visible = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    visible.append_message("user", [{"type": "text", "text": "in scope"}])
    visible.close()

    # Hidden session: cwd is an unrelated directory.
    hidden = Session.new(
        db_path, cwd=tmp_path / "elsewhere", model="m", flush_sync=True,
    )
    hidden.append_message("user", [{"type": "text", "text": "out of scope"}])
    hidden.close()

    monkeypatch.chdir(tmp_path)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.refresh_session_list(db_path)
        await pilot.pause()

        session_list = app.query_one(SessionList)
        items = list(session_list.query(ListItem))
        # Only the visible session shows up.
        assert len(items) == 1


@pytest.mark.asyncio
async def test_refresh_session_list_missing_db_is_no_crash(
    tmp_path: Path,
) -> None:
    """AC #252 item 5: a missing / corrupt DB is logged, not crashed on.

    The TUI must stay usable when the session DB can't be opened —
    refresh leaves the SessionList empty + a warning in the log.
    """
    from cothis.tui import CothisApp, SessionList

    bogus_path = tmp_path / "does-not-exist.db"
    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # No raise; the call logs + returns.
        app.refresh_session_list(bogus_path)
        await pilot.pause()
        # The app stays alive + queryable when storage can't be opened.
        assert app.query_one(SessionList) is not None


@pytest.mark.asyncio
async def test_session_selection_fires_hook_with_session_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #252 item 5 (selection): clicking a ListItem fires ``on_session_selected``.

    The hook receives the session id (without the ``s_`` prefix that
    Textual imposes because hex session ids can begin with a digit).
    Subclasses override the hook to wire spawn-and-attach; this test
    uses a capturing subclass to verify the call.
    """
    from textual.widgets import ListItem

    from cothis.session import Session
    from cothis.tui import CothisApp, SessionList

    class _CapturingApp(CothisApp):
        def __init__(self) -> None:
            super().__init__()
            self.captured: list[str] = []

        def on_session_selected(self, session_id: str) -> None:
            self.captured.append(session_id)

    db_path = tmp_path / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s.append_message("user", [{"type": "text", "text": "hi"}])
    sid = s.session_id
    s.close()

    monkeypatch.chdir(tmp_path)

    app = _CapturingApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.refresh_session_list(db_path)
        await pilot.pause()

        session_list = app.query_one(SessionList)
        items = list(session_list.query(ListItem))
        assert len(items) == 1
        # Trigger selection by posting a Selected message directly. The
        # user-facing way is keyboard enter on the cursor, but the test
        # harness wants the explicit message — it bypasses the focus /
        # cursor-position dance that flaked earlier pilot runs.
        first_item = items[0]
        session_list.post_message(SessionList.Selected(session_list, first_item, 0))
        await pilot.pause()

    assert app.captured == [sid], (
        f"expected on_session_selected called once with {sid!r}; "
        f"got {app.captured!r}"
    )


@pytest.mark.asyncio
async def test_refresh_session_list_enriches_label_with_worktree_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #234 #3: SessionList label includes ``branch:<name>`` when the session's cwd is in a known worktree.

    Stubs ``list_worktrees`` so the test is hermetic (no real git binary
    needed). When the stub returns a Worktree whose path is an ancestor
    of the session's cwd, the label gains ``· branch:<name>``.
    """
    from textual.widgets import Label, ListItem

    from cothis.git import Worktree
    from cothis.session import Session
    from cothis.tui import CothisApp, SessionList

    db_path = tmp_path / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s.append_message("user", [{"type": "text", "text": "hi"}])
    s.close()

    monkeypatch.chdir(tmp_path)

    def fake_list_worktrees(_cwd: Path) -> list[Worktree]:
        return [Worktree(tmp_path, "feature-branch")]

    monkeypatch.setattr("cothis.git.list_worktrees", fake_list_worktrees)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.refresh_session_list(db_path)
        await pilot.pause()

        session_list = app.query_one(SessionList)
        items = list(session_list.query(ListItem))
        assert len(items) == 1
        label_widget = items[0].query_one(Label)
        # ``Label`` inherits ``Static``; the source text lives on the
        # mangled private attr ``_Static__content``.
        label_str = str(getattr(label_widget, "_Static__content"))
        assert "branch:feature-branch" in label_str, (
            f"expected branch enrichment in label; got {label_str!r}"
        )


@pytest.mark.asyncio
async def test_refresh_session_list_skips_branch_when_no_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #234 #3: when ``list_worktrees`` returns ``[]``, label has no branch suffix.

    The cwd-only label is preserved — the TUI degrades cleanly when
    not in a git repo or git binary missing.
    """
    from textual.widgets import Label, ListItem

    from cothis.session import Session
    from cothis.tui import CothisApp, SessionList

    db_path = tmp_path / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s.append_message("user", [{"type": "text", "text": "hi"}])
    s.close()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("cothis.git.list_worktrees", lambda _cwd: [])

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.refresh_session_list(db_path)
        await pilot.pause()

        session_list = app.query_one(SessionList)
        items = list(session_list.query(ListItem))
        assert len(items) == 1
        label_widget = items[0].query_one(Label)
        label_str = str(getattr(label_widget, "_Static__content"))
        assert "branch:" not in label_str, (
            f"no branch expected when worktrees empty; got {label_str!r}"
        )


@pytest.mark.asyncio
async def test_refresh_session_list_groups_sessions_by_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #234 #5: sessions with the same cwd land adjacent in SessionList.

    Visual grouping by worktree: stable sort by cwd, with ``updated_at``
    as the within-group tiebreaker. Three sessions in two cwds end up
    as [cwd-A session 1, cwd-A session 2, cwd-B session 3] or its
    reverse — sessions in the same cwd are always adjacent.

    Pre-fix: sessions were listed in (updated_at, ...) order, so
    sessions in different cwds interleaved when their timestamps
    differed. Users running multiple sessions across worktrees had to
    scan the whole list to find related ones.
    """
    import re
    from itertools import groupby

    from textual.widgets import Label, ListItem

    from cothis.session import Session
    from cothis.session.storage import SessionRow
    from cothis.tui import CothisApp, SessionList

    db_path = tmp_path / "session.db"
    cwd_a = tmp_path / "worktree-a"
    cwd_b = tmp_path / "worktree-b"

    # Seed three sessions, alternating cwds so chronological order would
    # interleave them. The fix groups by cwd instead.
    seeded: list[SessionRow] = []
    for cwd, title in [(cwd_a, "first-in-a"), (cwd_b, "in-b"), (cwd_a, "second-in-a")]:
        s = Session.new(db_path, cwd=cwd, model="m", flush_sync=True)
        s.append_message("user", [{"type": "text", "text": title}])
        s.close()
        # Reload to capture the row's updated_at (post-flush).
        from cothis.session.storage import Storage

        storage = Storage(db_path)
        try:
            row = storage.load_session(s.session_id)
            assert row is not None
            seeded.append(row)
        finally:
            storage.close()

    # Stub ``list_sessions_in_cwd_tree`` to return all three rows —
    # otherwise the visibility filter (observer must be inside the
    # session's cwd) excludes both worktree subdirs.
    monkeypatch.setattr(
        "cothis.session.storage.Storage.list_sessions_in_cwd_tree",
        lambda self, cwd: seeded,
    )
    monkeypatch.setattr("cothis.git.list_worktrees", lambda _cwd: [])

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.refresh_session_list(db_path)
        await pilot.pause()

        session_list = app.query_one(SessionList)
        items = list(session_list.query(ListItem))
        assert len(items) == 3

        # Pull cwd out of each item's label "(path)" suffix to check grouping.
        cwds_in_list_order: list[str] = []
        for item in items:
            label_str = str(getattr(item.query_one(Label), "_Static__content"))
            match = re.search(r"\(([^)]+)\)", label_str)
            assert match is not None, f"no cwd in label {label_str!r}"
            cwds_in_list_order.append(match.group(1).split(" · ")[0])

        # Sessions with the same cwd must be adjacent.
        grouped_cwds = [k for k, _ in groupby(cwds_in_list_order)]
        assert len(grouped_cwds) == 2, (
            f"expected 2 distinct cwd groups (worktree-a, worktree-b); "
            f"got {grouped_cwds} — sessions not adjacent within their cwd"
        )


# ---------------------------------------------------------------------
# New-session binding (#234 — Ctrl-N fires ``on_new_session`` hook
# with the visible worktrees). The picker modal is a follow-up; this
# PR lands the foundation (binding + hook + dispatch).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_action_new_session_fires_hook_with_worktrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #234: ``action_new_session`` calls ``on_new_session`` with the worktree list.

    Subclass captures the call; ``list_worktrees`` is stubbed so the
    test is hermetic (no real git binary).
    """
    from cothis.git import Worktree
    from cothis.tui import CothisApp

    class _CapturingApp(CothisApp):
        def __init__(self) -> None:
            super().__init__()
            self.captured: list = []

        def on_new_session(self, worktrees: list) -> None:  # type: ignore[override]
            self.captured = worktrees

    monkeypatch.chdir(tmp_path)
    fake_worktrees = [Worktree(tmp_path, "feature-branch")]

    def fake_list_worktrees(_cwd: Path) -> list[Worktree]:
        return fake_worktrees

    monkeypatch.setattr("cothis.git.list_worktrees", fake_list_worktrees)

    app = _CapturingApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_new_session()
        await pilot.pause()

    assert app.captured == fake_worktrees


@pytest.mark.asyncio
async def test_action_new_session_passes_empty_list_when_not_in_git_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #234: when ``list_worktrees`` returns ``[]``, the hook gets an empty list.

    Degradation contract: action still fires the hook (with empty worktrees)
    so the subclass can decide how to render the no-worktrees state.
    """
    from cothis.tui import CothisApp

    class _CapturingApp(CothisApp):
        def __init__(self) -> None:
            super().__init__()
            self.captured: list | None = None

        def on_new_session(self, worktrees: list) -> None:  # type: ignore[override]
            self.captured = worktrees

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("cothis.git.list_worktrees", lambda _cwd: [])

    app = _CapturingApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_new_session()
        await pilot.pause()

    assert app.captured == []


@pytest.mark.asyncio
async def test_on_new_session_default_mounts_worktree_picker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #234 slice C: default ``on_new_session`` pushes ``WorktreePickerModal``.

    Replaces the slice-A no-op stub. Now Ctrl-N → ``action_new_session``
    → ``on_new_session`` mounts the picker (added in slice B) so the
    user sees the worktree list. Slice D will wire session creation on
    dismiss; this slice closes the "modal mounts" wiring contract.
    """
    from cothis.git import Worktree
    from cothis.tui import CothisApp, WorktreePickerModal

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.on_new_session([Worktree(Path("/repo/main"), "main")])
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, WorktreePickerModal), (
            f"expected WorktreePickerModal on top; "
            f"got {type(app.screen).__name__}"
        )

        modal.action_dismiss_modal()
        await pilot.pause()


@pytest.mark.asyncio
async def test_action_new_session_keypress_pushes_picker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #234 slice C: ``n`` keypress → ``action_new_session`` → picker mounts.

    End-to-end via the actual keypress binding (``n``, not ``ctrl+n`` —
    the binding was added in slice A). The default ``on_new_session``
    mounts ``WorktreePickerModal``; the test verifies the modal is on
    top of the screen stack after the keypress.
    """
    from cothis.git import Worktree
    from cothis.tui import CothisApp, WorktreePickerModal

    monkeypatch.setattr(
        "cothis.git.list_worktrees",
        lambda _cwd: [
            Worktree(Path("/repo/main"), "main"),
            Worktree(Path("/repo/feat"), "feature/x"),
        ],
    )

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, WorktreePickerModal)

        modal.action_dismiss_modal()
        await pilot.pause()


@pytest.mark.asyncio
async def test_on_worktree_pick_default_logs_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC #234 slice D: default ``on_worktree_pick`` logs the chosen path.

    The hook is the contract for "create a session bound to this cwd"
    — slice E (CLI integration) overrides it to call
    ``Supervisor.spawn_worker`` etc. The default impl logs so the
    wiring is observable without spawning, mirroring the other no-op
    hooks (``on_session_selected``, ``on_menu_open``) that subclasses
    override.
    """
    import logging

    from cothis.tui import CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with caplog.at_level(logging.INFO, logger="cothis.tui"):
            app.on_worktree_pick("/repo/feature-branch")
            await pilot.pause()

    assert any(
        "/repo/feature-branch" in r.getMessage() and "worktree picked" in r.getMessage()
        for r in caplog.records
    ), [
        f"expected 'worktree picked' log with the path; "
        f"got {[r.getMessage() for r in caplog.records]}"
    ]


@pytest.mark.asyncio
async def test_picker_dismiss_routes_to_on_worktree_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #234 slice D: ``WorktreePickerModal`` dismiss → ``on_worktree_pick``.

    Subclass captures the routed path via ``on_worktree_pick`` (the
    contract that slice E will override). The picker is mounted via
    the default ``on_new_session`` (slice C); picking a button
    dismisses + routes to the hook. Verifies the wiring end-to-end
    without spawning.
    """
    from textual.widgets import Button

    from cothis.git import Worktree
    from cothis.tui import CothisApp, WorktreePickerModal

    captured: list[str] = []

    class _CapturingApp(CothisApp):
        def on_worktree_pick(self, path: str) -> None:  # type: ignore[override]
            captured.append(path)

    monkeypatch.setattr(
        "cothis.git.list_worktrees",
        lambda _cwd: [
            Worktree(Path("/repo/main"), "main"),
            Worktree(Path("/repo/feat"), "feature/x"),
        ],
    )

    app = _CapturingApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Mount the picker via the default on_new_session hook.
        app.on_new_session([
            Worktree(Path("/repo/main"), "main"),
            Worktree(Path("/repo/feat"), "feature/x"),
        ])
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, WorktreePickerModal)

        # Click the second worktree's button — routes to on_worktree_pick
        # with that path (index-based ID per slice B).
        feature_button = next(
            b for b in modal.query(Button) if b.id == "wt-1"
        )
        await pilot.click(feature_button)
        await pilot.pause()

    assert captured == [str(Path("/repo/feat"))], (
        f"expected on_worktree_pick to be called with /repo/feat; "
        f"got {captured}"
    )


@pytest.mark.asyncio
async def test_picker_dismiss_cancel_does_not_call_on_worktree_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #234 slice D: cancelling the picker → ``on_worktree_pick`` NOT called.

    Cancellation (Esc / Cancel button) is distinct from picking a
    worktree. The ``on_new_session`` dismiss callback short-circuits
    on ``None`` — so the hook isn't fired with a sentinel value the
    caller would have to filter.
    """
    from textual.widgets import Button

    from cothis.git import Worktree
    from cothis.tui import CothisApp, WorktreePickerModal

    captured: list[str] = []

    class _CapturingApp(CothisApp):
        def on_worktree_pick(self, path: str) -> None:  # type: ignore[override]
            captured.append(path)

    monkeypatch.setattr(
        "cothis.git.list_worktrees",
        lambda _cwd: [Worktree(Path("/repo/main"), "main")],
    )

    app = _CapturingApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.on_new_session([Worktree(Path("/repo/main"), "main")])
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, WorktreePickerModal)

        cancel_button = next(
            b for b in modal.query(Button) if b.id == "worktree-cancel"
        )
        await pilot.click(cancel_button)
        await pilot.pause()

    assert captured == [], (
        f"on_worktree_pick should not be called on cancel; got {captured}"
    )


# ---------------------------------------------------------------------
# ask_user_request dispatch (#229 slice C) — TUI side. Worker-side
# Future blocking is Slice D; modal UI is Slice E.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_user_request_dispatches_to_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #229 slice C: inbound ``ask_user_request`` fires the hook with args.

    Subclass captures the call; verifies the hook receives ``ask_id``,
    ``prompt``, + ``choices``.
    """
    import json as _json

    from cothis.tui import CothisApp

    class _CapturingApp(CothisApp):
        def __init__(self) -> None:
            super().__init__()
            self.captured: dict = {}

        def on_ask_user_request(
            self, *, ask_id: str, prompt: str, choices: list,
        ) -> None:  # type: ignore[override]
            self.captured = {
                "ask_id": ask_id, "prompt": prompt, "choices": choices,
            }

    fake = _FakeWS([
        _json.dumps({
            "type": "ask_user_request",
            "ask_id": "ask_42",
            "prompt": "Deploy to prod?",
            "choices": ["yes", "no"],
        }),
    ])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = _CapturingApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        for _ in range(3):
            await pilot.pause()

    assert app.captured == {
        "ask_id": "ask_42", "prompt": "Deploy to prod?", "choices": ["yes", "no"],
    }


@pytest.mark.asyncio
async def test_ask_user_request_mounts_modal_and_routes_pick_to_resolve_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #229 slice F: ``ask_user_request`` → mount modal → pick → ``resolve_ask``.

    Replaces the slice-C auto-reject stub. Now the default
    ``on_ask_user_request`` pushes ``AskUserModal``; when the user
    clicks a choice button the dismiss callback fires + sends
    ``resolve_ask`` with the chosen value over the active session's WS.
    """
    import json as _json

    from textual.widgets import Button

    from cothis.tui import AskUserModal, CothisApp

    fake = _FakeWS([
        _json.dumps({
            "type": "ask_user_request",
            "ask_id": "ask_99",
            "prompt": "Continue?",
            "choices": ["y", "n"],
        }),
    ])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        for _ in range(3):
            await pilot.pause()

        modal = app.screen
        assert isinstance(modal, AskUserModal), (
            f"expected AskUserModal on top; got {type(app.screen).__name__}"
        )

        yes_button = next(
            b for b in modal.query(Button) if b.id == "choice-y"
        )
        await pilot.click(yes_button)
        await pilot.pause()

    resolve_frames = [
        _json.loads(f) for f in fake.sent
        if _json.loads(f).get("type") == "resolve_ask"
    ]
    assert len(resolve_frames) == 1, (
        f"expected 1 resolve_ask frame; got {resolve_frames}"
    )
    assert resolve_frames[0] == {
        "type": "resolve_ask", "ask_id": "ask_99", "value": "y",
    }


@pytest.mark.asyncio
async def test_ask_user_request_cancel_sends_resolve_ask_with_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #229 slice F: cancelling the modal → ``resolve_ask`` with ``value=None``.

    The agent treats ``None`` as "user declined" and the tool returns
    accordingly. This is the correct behaviour for Esc / Cancel — the
    alternative (no reply at all) would leave the agent's Future pending
    until turn timeout, which is much worse than a prompt "no".
    """
    import json as _json

    from textual.widgets import Button

    from cothis.tui import AskUserModal, CothisApp

    fake = _FakeWS([
        _json.dumps({
            "type": "ask_user_request",
            "ask_id": "ask_cancel",
            "prompt": "Deploy?",
            "choices": ["yes", "no"],
        }),
    ])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        for _ in range(3):
            await pilot.pause()

        modal = app.screen
        assert isinstance(modal, AskUserModal)

        cancel_button = next(
            b for b in modal.query(Button) if b.id == "ask-cancel"
        )
        await pilot.click(cancel_button)
        await pilot.pause()

    resolve_frames = [
        _json.loads(f) for f in fake.sent
        if _json.loads(f).get("type") == "resolve_ask"
    ]
    assert len(resolve_frames) == 1, (
        f"expected 1 resolve_ask frame; got {resolve_frames}"
    )
    assert resolve_frames[0] == {
        "type": "resolve_ask", "ask_id": "ask_cancel", "value": None,
    }


# ---------------------------------------------------------------------
# Menu binding (#235 slice A) — Ctrl-M fires ``on_menu_open`` hook.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_action_menu_fires_on_menu_open_hook() -> None:
    """AC #235 slice A: ``action_menu`` calls ``on_menu_open``."""
    from cothis.tui import CothisApp

    class _CapturingApp(CothisApp):
        def __init__(self) -> None:
            super().__init__()
            self.menu_fired = False

        def on_menu_open(self) -> None:  # type: ignore[override]
            self.menu_fired = True

    app = _CapturingApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_menu()
        await pilot.pause()

    assert app.menu_fired is True


@pytest.mark.asyncio
async def test_list_configurable_skills_returns_discovered_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #235: ``list_configurable_skills`` returns names from discover_skills."""
    from pathlib import Path as _Path

    from cothis.skills import Skill
    from cothis.tui import CothisApp

    fake_skills = [
        Skill(name="git-commit", description="d1", body="b1", source=_Path("/x")),
        Skill(name="fs-read", description="d2", body="b2", source=_Path("/y")),
    ]
    monkeypatch.setattr(
        "cothis.skills.discover_skills", lambda _cwd: fake_skills,
    )

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        names = app.list_configurable_skills()
    assert names == ["git-commit", "fs-read"]


@pytest.mark.asyncio
async def test_list_configurable_skills_empty_when_none_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #235: empty list when no skills are installed."""
    from cothis.tui import CothisApp

    monkeypatch.setattr("cothis.skills.discover_skills", lambda _cwd: [])

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.list_configurable_skills() == []


@pytest.mark.asyncio
async def test_config_menu_modal_renders_skill_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #235 slice B/C: ``action_menu`` pushes ConfigMenuModal with toggleable skill buttons."""
    from pathlib import Path as _Path

    from textual.widgets import Button

    from cothis.skills import Skill
    from cothis.tui import ConfigMenuModal, CothisApp

    fake_skills = [
        Skill(name="git-commit", description="d1", body="b1", source=_Path("/x")),
        Skill(name="fs-read", description="d2", body="b2", source=_Path("/y")),
    ]
    monkeypatch.setattr(
        "cothis.skills.discover_skills", lambda _cwd: fake_skills,
    )

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_menu()
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, ConfigMenuModal)
        buttons = list(modal.query(Button))
        button_ids = [b.id for b in buttons]
        assert "skill-git-commit" in button_ids
        assert "skill-fs-read" in button_ids
        assert "menu-done" in button_ids

        modal.action_dismiss_modal()
        await pilot.pause()


@pytest.mark.asyncio
async def test_config_menu_modal_toggle_selects_and_dismisses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #235 slice C: clicking a skill button toggles selection; Done returns the set."""
    from pathlib import Path as _Path

    from textual.widgets import Button

    from cothis.skills import Skill
    from cothis.tui import ConfigMenuModal, CothisApp

    fake_skills = [
        Skill(name="git-commit", description="d1", body="b1", source=_Path("/x")),
        Skill(name="fs-read", description="d2", body="b2", source=_Path("/y")),
    ]
    monkeypatch.setattr(
        "cothis.skills.discover_skills", lambda _cwd: fake_skills,
    )

    captured: list = []

    def on_dismiss(value: object) -> None:
        captured.append(value)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(ConfigMenuModal(["git-commit", "fs-read"]), on_dismiss)
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, ConfigMenuModal)
        assert modal._selected == set()

        git_button = next(
            b for b in modal.query(Button) if b.id == "skill-git-commit"
        )
        await pilot.click(git_button)
        await pilot.pause()
        assert "git-commit" in modal._selected

        done_button = next(
            b for b in modal.query(Button) if b.id == "menu-done"
        )
        await pilot.click(done_button)
        await pilot.pause()

    assert captured == [{"git-commit"}]


# ---------------------------------------------------------------------
# Active-session highlight (#230 slice D) — SessionList items gain
# ``active-session`` CSS class when their session becomes active.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_active_session_highlights_matching_list_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #230 slice D: the active session's ListItem gains ``active-session`` class.

    Seeds two sessions, selects one, verifies only its ListItem has
    the ``active-session`` class; the other doesn't.
    """
    from textual.widgets import ListItem

    from cothis.session import Session
    from cothis.tui import CothisApp, SessionList

    db_path = tmp_path / "session.db"

    s1 = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s1.append_message("user", [{"type": "text", "text": "one"}])
    sid1 = s1.session_id
    s1.close()

    s2 = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s2.append_message("user", [{"type": "text", "text": "two"}])
    sid2 = s2.session_id
    s2.close()

    monkeypatch.chdir(tmp_path)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.refresh_session_list(db_path)
        await pilot.pause()

        # Activate the first session.
        app.set_active_session(sid1)
        await pilot.pause()

        items = list(app.query_one(SessionList).query(ListItem))
        assert len(items) == 2
        classes = {item.id: item.classes for item in items}
        assert "active-session" in classes[f"s_{sid1}"]
        assert "active-session" not in classes[f"s_{sid2}"]

        # Switch to the second session.
        app.set_active_session(sid2)
        await pilot.pause()

        items = list(app.query_one(SessionList).query(ListItem))
        classes = {item.id: item.classes for item in items}
        assert "active-session" not in classes[f"s_{sid1}"]
        assert "active-session" in classes[f"s_{sid2}"]


# ---------------------------------------------------------------------
# Multi-session WS (#230 slice B)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attach_session_ws_stores_in_dict_and_sets_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #230 slice B: ``attach_session_ws`` stores WS in ``_ws_by_session``."""
    from cothis.tui import CothisApp

    fake = _FakeWS([])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_session_ws("session-a", "ws://fake/agent", "tok")
        await pilot.pause()

        assert "session-a" in app._ws_by_session
        assert app._ws_by_session["session-a"] is fake
        assert app._active_session_id == "session-a"
        assert "session-a" in app._ws_pump_tasks_by_session

        await app.detach_session_ws("session-a")
        await pilot.pause()

        assert "session-a" not in app._ws_by_session
        assert "session-a" not in app._ws_pump_tasks_by_session


@pytest.mark.asyncio
async def test_attach_session_ws_multiple_sessions_coexist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #230 slice B: two sessions attached simultaneously, both alive."""
    from cothis.tui import CothisApp

    fake_a = _FakeWS([])
    fake_b = _FakeWS([])

    def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake_a if "agent-a" in str(uri) else fake_b

    async def fake_connect_async(uri: str, **kw: object) -> _FakeWS:
        return fake_connect(uri, **kw)

    monkeypatch.setattr("websockets.connect", fake_connect_async)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_session_ws("session-a", "ws://fake/agent-a", "tok")
        await pilot.pause()
        await app.attach_session_ws("session-b", "ws://fake/agent-b", "tok")
        await pilot.pause()

        assert len(app._ws_by_session) == 2
        assert "session-a" in app._ws_by_session
        assert "session-b" in app._ws_by_session
        # Active session is the last-attached.
        assert app._active_session_id == "session-b"

        await app.detach_session_ws("session-a")
        await pilot.pause()
        assert "session-a" not in app._ws_by_session
        assert "session-b" in app._ws_by_session

        await app.detach_session_ws("session-b")
        await pilot.pause()


@pytest.mark.asyncio
async def test_send_run_turn_routes_to_active_session_ws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #230 slice C: ``send_run_turn`` routes to the active session's WS."""
    import json as _json

    from cothis.tui import CothisApp

    fake_a = _FakeWS([])
    fake_b = _FakeWS([])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake_a if "agent-a" in str(uri) else fake_b

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_session_ws("session-a", "ws://fake/agent-a", "tok")
        await pilot.pause()
        await app.attach_session_ws("session-b", "ws://fake/agent-b", "tok")
        await pilot.pause()

        await app.send_run_turn("hello from b")
        assert len(fake_b.sent) == 1
        assert _json.loads(fake_b.sent[0]) == {
            "type": "run_turn", "prompt": "hello from b",
        }
        assert fake_a.sent == []

        app.set_active_session("session-a")
        await pilot.pause()
        await app.send_run_turn("hello from a")
        assert len(fake_a.sent) == 1
        assert _json.loads(fake_a.sent[0]) == {
            "type": "run_turn", "prompt": "hello from a",
        }

        await app.detach_session_ws("session-a")
        await app.detach_session_ws("session-b")
        await pilot.pause()


# ---------------------------------------------------------------------
# AskUserModal (#229 slice E)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_user_modal_renders_prompt_and_choices() -> None:
    """AC #229 slice E: modal shows prompt + one button per choice."""
    from textual.widgets import Button, Label

    from cothis.tui import AskUserModal, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(AskUserModal("Deploy to prod?", ["yes", "no"]))
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, AskUserModal)

        labels = list(modal.query(Label))
        assert any("Deploy to prod?" in str(getattr(l, "_Static__content", "")) for l in labels)

        buttons = list(modal.query(Button))
        button_labels = [b.label.plain if hasattr(b.label, "plain") else str(b.label) for b in buttons]
        assert "yes" in button_labels
        assert "no" in button_labels
        assert "Cancel" in button_labels

        modal.action_dismiss_modal()
        await pilot.pause()


@pytest.mark.asyncio
async def test_ask_user_modal_choice_button_dismisses_with_value() -> None:
    """AC #229 slice E: clicking a choice button dismisses with that value."""
    from textual.widgets import Button

    from cothis.tui import AskUserModal, CothisApp

    captured: list = []

    def on_dismiss(value: str | None) -> None:
        captured.append(value)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(AskUserModal("Continue?", ["y", "n"]), on_dismiss)
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, AskUserModal)

        yes_button = next(
            b for b in modal.query(Button) if b.id == "choice-y"
        )
        await pilot.click(yes_button)
        await pilot.pause()

    assert captured == ["y"]


# ---------------------------------------------------------------------
# WorktreePickerModal (#234 slice B)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worktree_picker_renders_one_button_per_worktree_plus_cancel() -> None:
    """AC #234 slice B: modal shows branch-labeled buttons + Cancel.

    A worktree on a branch shows the branch name; a detached worktree
    falls back to its path basename. The Cancel button is always
    present so the user can bail out without creating a session.
    """
    from textual.widgets import Button

    from cothis.git import Worktree
    from cothis.tui import CothisApp, WorktreePickerModal

    worktrees = [
        Worktree(path=Path("/repo/main"), branch="main"),
        Worktree(path=Path("/repo/feature-x"), branch="feature/x"),
        Worktree(path=Path("/repo/detached"), branch=None),
    ]

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(WorktreePickerModal(worktrees))
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, WorktreePickerModal)

        buttons = list(modal.query(Button))
        labels = [
            b.label.plain if hasattr(b.label, "plain") else str(b.label)
            for b in buttons
        ]
        # Branch buttons show branch name; detached shows path basename.
        assert "main" in labels
        assert "feature/x" in labels
        assert "detached" in labels
        assert "Cancel" in labels
        assert "Current directory" in labels
        # One button per worktree + 1 Current-directory fallback + 1 Cancel.
        assert len(buttons) == len(worktrees) + 2

        modal.action_dismiss_modal()
        await pilot.pause()


@pytest.mark.asyncio
async def test_worktree_picker_button_dismisses_with_path_str() -> None:
    """AC #234 slice B: clicking a worktree button dismisses with its path.

    The dismiss value is the path as a string — what the caller stuffs
    into the new session's ``cwd``. Index-based IDs (paths contain
    ``/`` which Textual IDs reject) so the lookup is by button position.
    """
    from textual.widgets import Button

    from cothis.git import Worktree
    from cothis.tui import CothisApp, WorktreePickerModal

    worktrees = [
        Worktree(path=Path("/repo/main"), branch="main"),
        Worktree(path=Path("/repo/feature"), branch="feature/y"),
    ]
    captured: list[str | None] = []

    def on_dismiss(value: str | None) -> None:
        captured.append(value)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(WorktreePickerModal(worktrees), on_dismiss)
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, WorktreePickerModal)

        # Click the second button (feature/y) — verifies index-based ID
        # routing works for non-first entries.
        feature_button = next(
            b for b in modal.query(Button) if b.id == "wt-1"
        )
        await pilot.click(feature_button)
        await pilot.pause()

    assert captured == [str(Path("/repo/feature"))]


@pytest.mark.asyncio
async def test_worktree_picker_cancel_dismisses_with_none() -> None:
    """AC #234 slice B: Cancel button dismisses with ``None``.

    ``None`` is the "user cancelled, no new session" signal — callers
    treat it distinctly from any path string.
    """
    from textual.widgets import Button

    from cothis.git import Worktree
    from cothis.tui import CothisApp, WorktreePickerModal

    worktrees = [Worktree(path=Path("/repo/main"), branch="main")]
    captured: list[str | None] = []

    def on_dismiss(value: str | None) -> None:
        captured.append(value)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(WorktreePickerModal(worktrees), on_dismiss)
        await pilot.pause()

        modal = app.screen
        cancel_button = next(
            b for b in modal.query(Button) if b.id == "worktree-cancel"
        )
        await pilot.click(cancel_button)
        await pilot.pause()

    assert captured == [None]


@pytest.mark.asyncio
async def test_worktree_picker_current_dir_dismisses_with_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``Current directory`` button dismisses with ``str(Path.cwd())``.

    Fallback path for non-git cwds (no worktrees) + a quick "just use
    here" option when worktrees ARE available. The dismiss value is
    ``Path.cwd()`` as a string — same shape as the worktree paths, so
    ``on_worktree_pick`` handles both uniformly.
    """
    from textual.widgets import Button

    from cothis.git import Worktree
    from cothis.tui import CothisApp, WorktreePickerModal

    monkeypatch.chdir(tmp_path)
    captured: list[str | None] = []

    def on_dismiss(value: str | None) -> None:
        captured.append(value)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Even with worktrees present, the cwd button shows — it's a
        # first-class fallback, not just for the empty-list case.
        app.push_screen(
            WorktreePickerModal([Worktree(Path("/repo/main"), "main")]),
            on_dismiss,
        )
        await pilot.pause()

        modal = app.screen
        cwd_button = next(
            b for b in modal.query(Button) if b.id == "worktree-cwd"
        )
        await pilot.click(cwd_button)
        await pilot.pause()

    assert captured == [str(tmp_path)], (
        f"cwd button should dismiss with str(Path.cwd()); got {captured}"
    )


@pytest.mark.asyncio
async def test_worktree_picker_empty_list_renders_only_cancel() -> None:
    """AC #234 slice B: empty worktree list → just the Cancel button.

    Edge case: not a git repo, or git binary missing — ``list_worktrees``
    returns ``[]``. The modal still mounts (no crash) so the user sees
    an explicit "nothing to pick" rather than a silent no-op. The label
    changes to a hint that suggests ``git worktree add`` (cothis only
    discovers worktrees, it doesn't create them — out of scope per #234).
    """
    from textual.widgets import Button, Label

    from cothis.tui import CothisApp, WorktreePickerModal

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(WorktreePickerModal([]))
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, WorktreePickerModal)

        buttons = list(modal.query(Button))
        # Empty list: just the current-directory fallback + Cancel.
        assert len(buttons) == 2
        button_ids = {b.id for b in buttons}
        assert button_ids == {"worktree-cwd", "worktree-cancel"}

        # Empty-list label points the user at the fix outside cothis.
        # Plain-text comparison via ``str(content)`` — Textual's Label
        # stores a ``Text`` object; str() gives the rendered string.
        labels = list(modal.query(Label))
        assert labels, "expected at least one Label in the modal"
        label_text = str(getattr(labels[0], "_Static__content", ""))
        assert "No worktrees found" in label_text, (
            f"empty-list label should mention 'No worktrees found'; got {label_text!r}"
        )
        assert "git worktree add" in label_text, (
            f"empty-list label should suggest the fix; got {label_text!r}"
        )


# ---------------------------------------------------------------------
# Skill selection persistence (#235 slice D)
# ---------------------------------------------------------------------


def test_save_and_load_skill_selection_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #235 slice D: save → load round-trips the selected set."""
    from cothis.skills import load_skill_selection, save_skill_selection

    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))

    save_skill_selection({"git-commit", "fs-read"})
    loaded = load_skill_selection()
    assert loaded == {"git-commit", "fs-read"}


def test_load_skill_selection_empty_when_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #235 slice D: no file → empty set (first run)."""
    from cothis.skills import load_skill_selection

    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))
    assert load_skill_selection() == set()


def test_load_skill_selection_handles_corrupt_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #235 slice D: corrupt JSON → empty set + no crash."""
    from cothis.skills import _skill_selection_path, load_skill_selection

    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))
    path = _skill_selection_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert load_skill_selection() == set()


# ---------------------------------------------------------------------
# Ctrl-M skill-config menu end-to-end wiring (#415)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_menu_open_persists_selection_on_dismiss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#415: on_menu_open's dismiss path saves the selection (production wiring)."""
    from cothis.skills import load_skill_selection
    from cothis.tui import ConfigMenuModal, CothisApp

    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))
    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "list_configurable_skills", lambda: ["alpha", "beta"])
        app.on_menu_open()  # pushes the ConfigMenuModal
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ConfigMenuModal)
        modal.dismiss({"alpha"})  # simulate Done with "alpha" toggled on
        await pilot.pause()
    assert load_skill_selection() == {"alpha"}


@pytest.mark.asyncio
async def test_config_menu_seeds_from_saved_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#415: reopening the menu shows the previously-saved selection."""
    from cothis.skills import save_skill_selection
    from cothis.tui import ConfigMenuModal, CothisApp

    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))
    save_skill_selection({"alpha", "ghost"})  # 'ghost' is not available
    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "list_configurable_skills", lambda: ["alpha", "beta"])
        app.on_menu_open()
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ConfigMenuModal)
        # Seeded with saved ∩ available — the unavailable 'ghost' is dropped.
        assert modal._selected == {"alpha"}


# ---------------------------------------------------------------------
# Thinking-block rendering (#I11) — ``append_delta("thinking", ...)``
# accumulates into ``_thinking_buf`` (separate from ``_text_buf``) and
# finalises into a collapsed, dimmed ``Collapsible(Markdown(source),
# title="reasoning", classes="thinking-block")``. Thinking stays OUT of
# ``renderable_str`` (which reads ``_text_buf`` only).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_delta_renders_as_collapsible_after_finalise() -> None:
    """#I11: thinking deltas mount a collapsed Collapsible on finalise.

    ``append_delta("thinking", ...)`` accumulates into ``_thinking_buf``;
    ``_finalize_active()`` flushes it as ``Collapsible(Markdown(source),
    title="reasoning")``. Exactly one Collapsible mounts and the reasoning
    source is preserved inside its Markdown widget (collapsed by default
    so it stays out of the way until the user expands it).
    """
    from textual.widgets import Collapsible, Markdown

    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        view.append_delta("thinking", "I should consider the options.")
        await pilot.pause()
        view._finalize_active()
        await pilot.pause()

        collapsibles = list(view.query(Collapsible))
        assert len(collapsibles) == 1, (
            f"expected one thinking Collapsible; got {len(collapsibles)}"
        )
        col = collapsibles[0]
        assert str(col.title) == "reasoning", (
            f"expected Collapsible title 'reasoning'; got {col.title!r}"
        )
        # The reasoning source is preserved inside the Collapsible's
        # Markdown widget. ``_markdown`` holds the source after mount.
        md = col.query_one(Markdown)
        assert "consider the options" in md._markdown, (
            f"reasoning not in Collapsible Markdown; got {md._markdown!r}"
        )


@pytest.mark.asyncio
async def test_thinking_stays_out_of_renderable_str_while_collapsible_mounted() -> None:
    """#I11: a mounted thinking Collapsible coexists with a ``renderable_str``
    that EXCLUDES the reasoning.

    ``renderable_str`` reads ``_text_buf`` only; thinking lives in
    ``_thinking_buf`` and finalises into a sibling widget. So the
    reasoning is present in the DOM (as a Collapsible) yet absent from
    the text-segment source — guards against a regression that folds
    thinking back into the text buffer (which would clutter the answer).
    """
    from textual.widgets import Collapsible

    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        view.append_delta("thinking", "secret reasoning here")
        await pilot.pause()
        view._finalize_active()
        await pilot.pause()

        # The Collapsible IS mounted in the DOM...
        assert len(list(view.query(Collapsible))) == 1, (
            "expected the thinking Collapsible to be mounted"
        )
        # ...yet renderable_str (the text-segment source) excludes it.
        assert "secret reasoning" not in view.renderable_str, (
            f"thinking leaked into renderable_str: {view.renderable_str!r}"
        )
        assert view.renderable_str == ""


@pytest.mark.asyncio
async def test_thinking_then_text_mounts_collapsible_before_markdown() -> None:
    """#I11: a kind switch (thinking → text) finalises thinking first, so
    the Collapsible mounts BEFORE the text Markdown in DOM order.

    Event order must match DOM order: the model's reasoning block
    precedes its answer. The kind-switch in ``append_delta`` flushes the
    thinking buffer before accumulating text, so the Collapsible is
    already mounted when the text segment finalises below it.
    """
    from textual.widgets import Collapsible, Markdown

    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        view.append_delta("thinking", "let me think...")
        await pilot.pause()
        view.append_delta("text", "here is my answer.")
        await pilot.pause()
        view._finalize_active()
        await pilot.pause()

        # Direct children of ConversationView, in mount order. The
        # Markdown nested inside the Collapsible is NOT a direct child,
        # so positions_md captures only the text-segment Markdown.
        children = list(view.children)
        positions_col = [i for i, c in enumerate(children) if isinstance(c, Collapsible)]
        positions_md = [i for i, c in enumerate(children) if isinstance(c, Markdown)]
        assert len(positions_col) == 1, (
            f"expected one Collapsible direct child; got {positions_col}"
        )
        assert len(positions_md) == 1, (
            f"expected one text Markdown direct child; got {positions_md}"
        )
        assert positions_col[0] < positions_md[0], (
            f"thinking Collapsible (idx {positions_col[0]}) must precede the "
            f"text Markdown (idx {positions_md[0]})"
        )


@pytest.mark.asyncio
async def test_tool_call_after_thinking_flushes_collapsible_above_card() -> None:
    """#I11: a tool call after streaming thinking flushes the thinking
    block so the Collapsible mounts ABOVE the ToolCallCard.

    ``append_tool_call`` calls ``_finalize_active()`` (the boundary
    flush), which mounts the pending thinking Collapsible before the
    card itself is mounted — DOM order matches event order: reasoning →
    tool dispatch.
    """
    from textual.widgets import Collapsible

    from cothis.tui import ConversationView, CothisApp, ToolCallCard

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        view.append_delta("thinking", "planning the tool call.")
        await pilot.pause()
        view.append_tool_call("fs.read")
        await pilot.pause()
        # ``append_tool_call`` already flushes at the boundary; the explicit
        # call is idempotent and mirrors the idle-timer finalise path.
        view._finalize_active()
        await pilot.pause()

        children = list(view.children)
        positions_col = [i for i, c in enumerate(children) if isinstance(c, Collapsible)]
        positions_card = [i for i, c in enumerate(children) if isinstance(c, ToolCallCard)]
        assert len(positions_col) == 1, (
            f"expected one thinking Collapsible above the card; got {positions_col}"
        )
        assert len(positions_card) == 1, (
            f"expected one ToolCallCard; got {positions_card}"
        )
        assert positions_col[0] < positions_card[0], (
            f"thinking Collapsible (idx {positions_col[0]}) must precede the "
            f"ToolCallCard (idx {positions_card[0]})"
        )


# ---------------------------------------------------------------------
# Footer + Esc-to-interrupt (#I24)
#
# Headless coverage for the status bar + run-state lifecycle. Drives
# run-state via ``_dispatch_ws_message`` / ``action_interrupt_turn`` directly
# (NOT via scroll positioning) so the known-flaky scroll race (#450) is
# not perturbed.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_footer_is_mounted_and_renders_idle_state() -> None:
    """Footer mounts on launch + its initial render shows the 5 cells (#I24)."""
    from cothis.tui import CothisApp, CothisFooter

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        footer = app.query_one(CothisFooter)
        rendered = str(footer.content)
        assert "state:idle" in rendered
        assert "session:" in rendered
        assert "ctx:" in rendered
        assert "skills:" in rendered


@pytest.mark.asyncio
async def test_turn_started_sets_run_state_running() -> None:
    """``turn_started`` WS frame flips ``run_state`` to ``running`` (#I24)."""
    from cothis.tui import CothisApp, CothisFooter

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._dispatch_ws_message({"type": "turn_started"})
        await pilot.pause()
        assert app.run_state == "running"
        footer_render = str(app.query_one(CothisFooter).content)
        assert "state:running" in footer_render


@pytest.mark.asyncio
async def test_turn_finished_updates_footer_fields_and_run_state() -> None:
    """``turn_finished`` payload updates footer cells + reconciles to idle (#I24)."""
    from cothis.tui import CothisApp, CothisFooter

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._dispatch_ws_message({"type": "turn_started"})
        await pilot.pause()
        assert app.run_state == "running"
        app._dispatch_ws_message(
            {
                "type": "turn_finished",
                "model": "m1",
                "session_id": "abcdef0123",
                "pressure": "medium",
                "active_skills": ["git-commit"],
            }
        )
        await pilot.pause()
        assert app.run_state == "idle"
        assert app.footer_model == "m1"
        assert app.footer_session == "abcdef0123"
        assert app.footer_pressure == "medium"
        assert app.footer_skills == ["git-commit"]
        footer_render = str(app.query_one(CothisFooter).content)
        assert "m1" in footer_render
        assert "session:abcdef01" in footer_render
        assert "ctx:medium" in footer_render
        assert "skills:git-commit" in footer_render
        assert "state:idle" in footer_render


@pytest.mark.asyncio
async def test_action_interrupt_turn_sends_interrupt_and_sets_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While ``running``, ``action_interrupt_turn`` sends one ``interrupt_turn`` frame (#I24)."""
    import json as _json

    from cothis.tui import CothisApp

    fake = _FakeWS([])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        await pilot.pause()
        app.run_state = "running"
        await pilot.pause()
        await app.action_interrupt_turn()
        await pilot.pause()
        assert app.run_state == "interrupted"
        assert len(fake.sent) == 1
        assert _json.loads(fake.sent[0]) == {"type": "interrupt_turn"}
        await app.detach_ws()
        await pilot.pause()


@pytest.mark.asyncio
async def test_action_interrupt_turn_is_noop_when_idle() -> None:
    """When idle (default), ``action_interrupt_turn`` sends nothing + state unchanged (#I24)."""
    from cothis.tui import CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.run_state == "idle"
        await app.action_interrupt_turn()
        await pilot.pause()
        assert app.run_state == "idle"


@pytest.mark.asyncio
async def test_action_interrupt_turn_noop_when_running_but_no_ws() -> None:
    """Running + no WS attached → interrupt is a safe no-op (no crash, no state corruption) (#I24)."""
    from cothis.tui import CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.run_state = "running"
        await pilot.pause()
        # No WS attached — must not raise and must not flip to interrupted.
        await app.action_interrupt_turn()
        await pilot.pause()
        assert app.run_state == "running"


@pytest.mark.asyncio
async def test_turn_finished_after_interrupt_reconciles_to_idle() -> None:
    """Optimistic ``interrupted`` reconciles to ``idle`` on the terminal frame (#I24)."""
    from cothis.tui import CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.run_state = "running"
        await pilot.pause()
        app.run_state = "interrupted"
        await pilot.pause()
        app._dispatch_ws_message(
            {
                "type": "turn_finished",
                "model": "m1",
                "session_id": "abcdef0123",
                "pressure": "low",
                "active_skills": [],
            }
        )
        await pilot.pause()
        assert app.run_state == "idle"


@pytest.mark.asyncio
async def test_escape_keypress_routes_to_interrupt_when_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pilot.press('escape')`` while running triggers the interrupt action (#I24).

    Also covers the priority-binding + TextArea-focus interaction: with the
    input focused, Esc must still route to the app-level binding (not be
    swallowed by the TextArea).
    """
    import json as _json

    from cothis.tui import CothisApp

    fake = _FakeWS([])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        await pilot.pause()
        # Focus the input so the priority binding is the only path Esc can
        # take — without priority a focused TextArea would consume it.
        app.query_one("#input").focus()
        await pilot.pause()
        app.run_state = "running"
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.run_state == "interrupted"
        assert any(
            _json.loads(s) == {"type": "interrupt_turn"} for s in fake.sent
        ), f"expected an interrupt_turn frame; sent={fake.sent}"
        await app.detach_ws()
        await pilot.pause()


@pytest.mark.asyncio
async def test_escape_dismisses_modal_not_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a modal is pushed, Esc dismisses the modal — app interrupt does NOT fire (#I24).

    Regression guard for the Esc-binding-collision risk: the app-level
    non-priority ``Binding('escape')`` routes through the focused-widget
    layer, so a pushed modal's own Esc binding wins while it's open and the
    app interrupt action does not fire.
    """
    import json as _json

    from cothis.tui import AskUserModal, CothisApp

    fake = _FakeWS([])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        await pilot.pause()
        app.run_state = "running"
        await pilot.pause()
        # Capture the screen-stack depth so we can assert the modal was
        # actually pushed + then dismissed by Esc.
        base_depth = len(app.screen_stack)
        # Push a modal that has its own Esc binding (dismiss_modal).
        app.push_screen(AskUserModal("pick one", ["a", "b"]))
        await pilot.pause()
        assert len(app.screen_stack) == base_depth + 1
        await pilot.press("escape")
        await pilot.pause()
        # Modal dismissed — stack depth back to baseline…
        assert len(app.screen_stack) == base_depth
        # …and the app never sent an interrupt_turn frame.
        assert not any(
            _json.loads(s).get("type") == "interrupt_turn" for s in fake.sent
        ), (
            f"Esc should have dismissed the modal, not interrupted; sent={fake.sent}"
        )
        # run_state is unchanged — the interrupt action did not fire.
        assert app.run_state == "running"
        await app.detach_ws()
        await pilot.pause()


# ---------------------------------------------------------------------
# Replay-on-attach (#I29, slice A): on attach the TUI reads the session
# store, rebuilds the Anthropic-shape messages, and renders them into
# ConversationView reusing the existing primitives. The live-streaming
# attach path stays behaviour-identical (db_path defaults None). Tests
# seed a real temp DB via the Session API (the existing idiom), drive
# replay, and assert widget presence / count + DOM order — never scroll
# position (the #450 scroll-race neighbour).
# ---------------------------------------------------------------------


def _seed_history_session(
    db: Path, cwd: Path,
) -> tuple[str, Path]:
    """Seed a session with user text + assistant [text + tool_use] + tool_result.

    Mirrors the test_session.py seed idiom (flush_sync for determinism).
    Returns ``(session_id, db_path)``.
    """
    from cothis.session import Session

    s = Session.new(db, cwd=cwd, model="m", flush_sync=True)
    sid = s.session_id
    s.append_message("user", [{"type": "text", "text": "hello there"}])
    s.append_message(
        "assistant",
        [
            {"type": "text", "text": "I will read the file."},
            {
                "type": "tool_use",
                "id": "tu1",
                "name": "fs.read",
                "input": {"path": "/x"},
            },
        ],
    )
    s.append_block(
        "user",
        {
            "type": "tool_result",
            "tool_use_id": "tu1",
            "content": "file contents",
        },
    )
    s.close()
    return sid, db


@pytest.mark.asyncio
async def test_replay_renders_seeded_history(tmp_path: Path) -> None:
    """Replay renders stored text + a done tool card into ConversationView (#I29)."""
    from textual.widgets import Markdown

    from cothis.tui import ConversationView, CothisApp, ToolCallCard

    db = tmp_path / "sessions" / "session.db"
    sid, db = _seed_history_session(db, tmp_path)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.replay_session_history(sid, db)
        await pilot.pause()
        view = app.query_one(ConversationView)

        # Two Markdown segments: the user echo + the assistant text.
        markdowns = list(view.query(Markdown))
        assert len(markdowns) == 2
        sources = [getattr(md, "_markdown", "") or "" for md in markdowns]
        assert any("hello there" in src for src in sources)
        assert any("I will read the file" in src for src in sources)

        # One tool_use card, status='done' (historical).
        cards = list(view.query(ToolCallCard))
        assert len(cards) == 1
        assert cards[0]._status == "done"
        assert cards[0]._call_id == "tu1"


@pytest.mark.asyncio
async def test_replay_empty_session_leaves_view_blank(tmp_path: Path) -> None:
    """An empty session (peek_messages returns []) renders nothing (#I29)."""
    from textual.widgets import Markdown

    from cothis.session import Session
    from cothis.tui import ConversationView, CothisApp, ToolCallCard

    db = tmp_path / "sessions" / "session.db"
    s = Session.new(db, cwd=tmp_path, model="m", flush_sync=True)
    sid = s.session_id
    s.close()

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.replay_session_history(sid, db)
        await pilot.pause()
        view = app.query_one(ConversationView)
        assert list(view.query(Markdown)) == []
        assert list(view.query(ToolCallCard)) == []


@pytest.mark.asyncio
async def test_replay_missing_db_is_best_effort_no_crash(tmp_path: Path) -> None:
    """A missing/corrupt DB logs a warning and leaves the view blank (#I29).

    Mirrors ``refresh_session_list``'s missing-DB contract: the TUI
    stays usable when the storage layer is unavailable.
    """
    from textual.widgets import Markdown

    from cothis.tui import ConversationView, CothisApp, ToolCallCard

    bogus_db = tmp_path / "does-not-exist.db"

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Must not raise.
        app.replay_session_history("0" * 32, bogus_db)
        await pilot.pause()
        view = app.query_one(ConversationView)
        assert list(view.query(Markdown)) == []
        assert list(view.query(ToolCallCard)) == []


@pytest.mark.asyncio
async def test_attach_session_ws_replay_is_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """``db_path=`` triggers replay on attach; omitting it leaves the view blank (#I29).

    Pins the opt-in contract: every existing 3-positional-arg caller
    (production crash-restart + tests) stays behaviour-identical because
    ``db_path`` defaults ``None``.
    """
    from textual.widgets import Markdown

    from cothis.tui import ConversationView, CothisApp, ToolCallCard

    db = tmp_path / "sessions" / "session.db"
    sid, db = _seed_history_session(db, tmp_path)

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return _FakeWS([])

    monkeypatch.setattr("websockets.connect", fake_connect)

    # WITH db_path → replay populates the view.
    app_with = CothisApp()
    async with app_with.run_test() as pilot:
        await pilot.pause()
        await app_with.attach_session_ws(
            sid, "ws://fake/agent", "tok", db_path=db,
        )
        await pilot.pause()
        view = app_with.query_one(ConversationView)
        assert len(list(view.query(Markdown))) >= 1
        assert len(list(view.query(ToolCallCard))) == 1
        await app_with.detach_session_ws(sid)
        await pilot.pause()

    # WITHOUT db_path → view stays blank (no replay).
    app_without = CothisApp()
    async with app_without.run_test() as pilot:
        await pilot.pause()
        await app_without.attach_session_ws(sid, "ws://fake/agent", "tok")
        await pilot.pause()
        view = app_without.query_one(ConversationView)
        assert list(view.query(Markdown)) == []
        assert list(view.query(ToolCallCard)) == []
        await app_without.detach_session_ws(sid)
        await pilot.pause()


@pytest.mark.asyncio
async def test_replay_leaves_view_state_clean_for_live_stream(tmp_path: Path) -> None:
    """After replay, a live ``append_delta`` still renders a fresh segment (#I29).

    Regression guard for the streaming path: replay must not leave
    ``_finalized`` / ``_stream_static`` stuck, or the next live delta
    would append to a retained buffer instead of starting a fresh
    Markdown segment.
    """
    from textual.widgets import Markdown

    from cothis.tui import ConversationView, CothisApp

    db = tmp_path / "sessions" / "session.db"
    sid, db = _seed_history_session(db, tmp_path)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.replay_session_history(sid, db)
        await pilot.pause()
        view = app.query_one(ConversationView)
        replayed_markdowns = len(list(view.query(Markdown)))

        # A live assistant delta after replay starts a fresh segment.
        app.append_assistant_delta("text", "live token")
        view._finalize_segment()  # idle-timer proxy
        await pilot.pause()

        markdowns = list(view.query(Markdown))
        # One new Markdown mounted for the live delta.
        assert len(markdowns) == replayed_markdowns + 1
        live_sources = [
            getattr(md, "_markdown", "") or "" for md in markdowns
        ]
        assert any("live token" in src for src in live_sources)


@pytest.mark.asyncio
async def test_replay_multi_turn_history_preserves_dom_order(tmp_path: Path) -> None:
    """A multi-turn history (user/assistant/user/assistant) renders in message order (#I29)."""
    from textual.widgets import Markdown

    from cothis.session import Session
    from cothis.tui import ConversationView, CothisApp

    db = tmp_path / "sessions" / "session.db"
    s = Session.new(db, cwd=tmp_path, model="m", flush_sync=True)
    sid = s.session_id
    s.append_message("user", [{"type": "text", "text": "first question"}])
    s.append_message(
        "assistant", [{"type": "text", "text": "first answer"}],
    )
    s.append_message("user", [{"type": "text", "text": "second question"}])
    s.append_message(
        "assistant", [{"type": "text", "text": "second answer"}],
    )
    s.close()

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.replay_session_history(sid, db)
        await pilot.pause()
        view = app.query_one(ConversationView)

        # Four Markdown segments in DOM order: u1, a1, u2, a2.
        markdowns = list(view.query(Markdown))
        assert len(markdowns) == 4
        sources = [
            getattr(md, "_markdown", "") or "" for md in markdowns
        ]
        assert "first question" in sources[0]
        assert "first answer" in sources[1]
        assert "second question" in sources[2]
        assert "second answer" in sources[3]

        # DOM order of immediate children matches message order.
        children = list(view.children)
        positions = [
            i for i, c in enumerate(children) if isinstance(c, Markdown)
        ]
        assert positions == sorted(positions)
