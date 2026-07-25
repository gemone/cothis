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

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


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
        await app.action_send_prompt()
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

    Threshold 3.5× leaves headroom for CI runner variance + Textual's
    per-widget layout overhead (which grows linearly with conversation
    length). A pre-fix regression to true O(N²) gives ~4× on this
    workload — the threshold catches that while not flaking on overhead.
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

    # Two workloads: 200 deltas vs 400 deltas, flushing every 50.
    # Larger flush interval → per-segment buffer cost dominates over
    # per-widget mount overhead, isolating the regression we're guarding.
    t_small = await _run(200, flush_every=50)
    t_large = await _run(400, flush_every=50)
    ratio = t_large / t_small if t_small > 0 else float("inf")
    assert ratio <= 3.5, (
        f"expected ≤3.5× slowdown on 2× workload (linear + overhead); "
        f"got {ratio:.2f}× (small={t_small*1000:.1f}ms, large={t_large*1000:.1f}ms) — "
        f"buffer is accumulating O(N²) work somewhere"
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

    from cothis.tui import ConversationView, CothisApp, InputBar

    fake = _FakeWS([])

    async def fake_connect(uri: str, **kw: object) -> _FakeWS:
        return fake

    monkeypatch.setattr("websockets.connect", fake_connect)

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.attach_ws("ws://fake/agent", "tok")
        await pilot.pause()
        bar = app.query_one(InputBar)
        bar.set_text("what is 2+2?")
        await app.action_send_prompt()
        await pilot.pause()

        # Local echo: user prompt lands in the view.
        view = app.query_one(ConversationView)
        assert "what is 2+2?" in view.renderable_str
        # Bar cleared.
        assert bar.get_text() == ""
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
    from cothis.tui import ConversationView, CothisApp, InputBar

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(InputBar)
        bar.set_text("hello")
        await app.action_send_prompt()
        await pilot.pause()
        view = app.query_one(ConversationView)
        assert "hello" in view.renderable_str
        assert bar.get_text() == ""
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
