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
        # One button per worktree + 1 Cancel.
        assert len(buttons) == len(worktrees) + 1

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
        assert len(buttons) == 1
        assert buttons[0].id == "worktree-cancel"

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
    from cothis.tui import load_skill_selection, save_skill_selection

    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))

    save_skill_selection({"git-commit", "fs-read"})
    loaded = load_skill_selection()
    assert loaded == {"git-commit", "fs-read"}


def test_load_skill_selection_empty_when_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #235 slice D: no file → empty set (first run)."""
    from cothis.tui import load_skill_selection

    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))
    assert load_skill_selection() == set()


def test_load_skill_selection_handles_corrupt_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #235 slice D: corrupt JSON → empty set + no crash."""
    from cothis.tui import _skill_selection_path, load_skill_selection

    monkeypatch.setenv("COTHIS_HOME", str(tmp_path))
    path = _skill_selection_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert load_skill_selection() == set()
