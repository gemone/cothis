"""``cothis.tui`` — Textual TUI core (#228).

3-pane layout for a single attached session:

- ``SessionList`` (left): sessions from the session table.
- ``ConversationView`` (center): scrollable Markdown + tool-call cards.
- ``InputBar`` (bottom): multiline input with Ctrl+Enter to send.

Stream routing per the design-review sign-off (#228, 2026-07-24):
``ContentDelta(kind="text")`` renders as normal assistant content;
``ContentDelta(kind="thinking")`` renders dimmed. Tool calls render
as inline cards with a status badge.

WS attach + real ``run_turn`` forwarding lands when the worker CLI
entrypoint (#250) is finalised; for now, ``send_prompt`` emits a
``PostMessage`` that the app's action handler processes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Header,
    Label,
    ListItem,
    ListView,
    Markdown,
    Static,
    TextArea,
)

if TYPE_CHECKING:
    from typing import Any

    import websockets

logger = logging.getLogger(__name__)

_TOOL_STATUS_ICONS = {"running": ">>", "done": "OK", "failed": "XX"}


# ---------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------


class SessionList(ListView):
    """Left pane — sessions from the session table.

    Placeholder for now; real session-table population + selection
    handling lands when multi-session (#230) work begins.
    """

    DEFAULT_CSS = """
    SessionList {
        width: 24;
        dock: left;
        border: round $primary;
    }
    """


class ToolCallCard(Static):
    """Inline card for one tool dispatch — name + status badge.

    Status reflects only the start event today; wiring to
    ``tool_call_completed``/``tool_call_failed`` (notify bus
    events from #224) lands when the WS client attaches (#250).
    """

    DEFAULT_CSS = """
    ToolCallCard {
        margin: 0 0 0 2;
        padding: 0 1;
        background: $surface;
        border-left: thick $accent;
    }
    """

    def __init__(self, name: str, status: str = "running") -> None:
        self._name = name
        self._status = status
        super().__init__(self._render_str())

    def set_status(self, status: str) -> None:
        self._status = status
        self.update(self._render_str())

    def _render_str(self) -> str:
        icon = _TOOL_STATUS_ICONS.get(self._status, "?")
        return f"[{icon}] {self._name}"


class ConversationView(VerticalScroll):
    """Center pane — scrollable Markdown + tool-call cards.

    ``append_delta`` is the primary API the WS client calls per
    ``assistant_delta`` message. ``append_tool_call`` mounts a card
    between text segments so DOM order matches event order (each
    card flushes the active text segment; the next delta starts a
    fresh segment below the card).
    """

    DEFAULT_CSS = """
    ConversationView {
        width: 2fr;
        border: round $accent;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        # ``list[str]`` accumulator (#267): ``+=`` on a Python ``str`` is
        # O(N) per call (immutable copy); ``list.append`` is amortised O(1).
        # ``renderable_str`` joins lazily — only tests + the Markdown render
        # path need the joined string, not the per-delta append path.
        self._text_buf: list[str] = []
        # Pattern 2 (#267 + #228 Rule 3): one Markdown widget per text
        # segment. ``append_tool_call`` flushes the active segment (clears
        # the buffer + resets this handle) so the next delta mounts a fresh
        # widget below the card, preserving DOM/event order.
        self._active_markdown: Markdown | None = None

    @property
    def renderable_str(self) -> str:
        """Accumulated text-delta source for the active segment — tests + debugging."""
        return "".join(self._text_buf)

    def append_delta(self, kind: str, text: str) -> None:
        """Route a ContentDelta to the right rendering path.

        ``kind="text"`` → accumulate + re-render the active Markdown segment.
        ``kind="thinking"`` → logged but not rendered (collapsible block
        lands when the toggle UX is designed).
        """
        if kind == "text":
            self._text_buf.append(text)
            self._refresh_markdown()
        elif kind == "thinking":
            logger.debug("dropping thinking delta (%d chars)", len(text))

    def append_user_message(self, text: str) -> None:
        """Render a user prompt with a distinct prefix.

        User text is Markdown-escaped (brackets) so injected links
        or markup can't activate inside the Markdown widget.
        """
        safe = text.replace("[", "\\[").replace("]", "\\]")
        self._text_buf.append(f"\n> **you**: {safe}\n\n")
        self._refresh_markdown()

    def append_tool_call(self, name: str, status: str = "running") -> ToolCallCard:
        """Mount an inline tool-call card; return it for status updates.

        Flushes the active text segment (current buffer + Markdown widget)
        so the next text delta starts a fresh segment below this card.
        Without the flush, all text would accumulate in one Markdown
        widget and the card would render below all of it, regardless
        of when it was mounted — violating the "tool calls render as
        inline cards" acceptance criterion on #228 (Rule 3).
        """
        self._refresh_markdown()
        self._text_buf = []
        self._active_markdown = None
        card = ToolCallCard(name=name, status=status)
        self.mount(card)
        return card

    def _refresh_markdown(self) -> None:
        """Update the active Markdown segment, or mount a new one if none.

        When ``_active_markdown`` is ``None`` (initial state, or after a
        tool-call card reset the segment), mount a fresh widget so the
        next text deltas accumulate into a new segment below any prior
        cards. Otherwise update the existing active widget in place.

        Per-call cost is bounded by the active segment's size, not by
        total conversation size — each ``append_tool_call`` starts a
        new segment so a long conversation with N tool calls has N
        small segments instead of one growing buffer (#267).
        """
        source = "".join(self._text_buf)
        if self._active_markdown is None:
            self._active_markdown = Markdown(source)
            self.mount(self._active_markdown)
        else:
            self._active_markdown.update(source)


class InputBar(Container):
    """Bottom pane — multiline input with Ctrl+Enter to send."""

    DEFAULT_CSS = """
    InputBar {
        height: 3;
        dock: bottom;
        border: round $secondary;
    }
    InputBar TextArea {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield TextArea()

    def get_text(self) -> str:
        """Current input text."""
        return self.query_one(TextArea).text

    def set_text(self, text: str) -> None:
        """Replace the input text."""
        self.query_one(TextArea).text = text

    def clear(self) -> None:
        """Clear the input after send."""
        self.query_one(TextArea).text = ""


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------


class CothisApp(App):
    """Textual app shell — 3-pane layout, single session.

    Keymap per design-review sign-off (#228, 2026-07-24):

    | Ctrl+Enter | send prompt |
    | Esc        | interrupt / clear / dismiss overlay |
    | Ctrl+C     | quit |
    """

    TITLE = "cothis"
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+enter", "send_prompt", "Send", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    # WS attach state (#252 item 1). ``None`` until ``attach_ws`` runs;
    # ``attach_ws`` re-uses these slots idempotently. Typed as ``Any``
    # because websockets' client connection class moved across versions
    # (``ClientConnection`` in v13+, ``WebSocketClientProtocol`` pre-v13)
    # and we don't need to call any methods on it outside this file.
    _ws: Any = None
    _ws_pump_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            yield SessionList(
                ListItem(Label("session-1")),
                ListItem(Label("session-2")),
                id="session-list",
            )
            yield ConversationView()
        yield InputBar()

    def action_send_prompt(self) -> None:
        """Read InputBar text → render in conversation → clear bar.

        The actual WS ``run_turn`` forward will be wired here once the
        worker CLI entrypoint (#250) lands. For now this is the local
        echo path that the pilot tests exercise.
        """
        bar = self.query_one(InputBar)
        text = bar.get_text().strip()
        if not text:
            return
        view = self.query_one(ConversationView)
        view.append_user_message(text)
        bar.clear()

    def append_assistant_delta(self, kind: str = "text", text: str = "") -> None:
        """Forward a WS ``assistant_delta`` to the conversation view.

        ``kind`` defaults to ``"text"`` for mixed-version compatibility
        (old servers without the ``kind`` field in the WS message).
        """
        self.query_one(ConversationView).append_delta(kind, text)

    def append_tool_call(self, name: str, status: str = "running") -> Any:
        """Forward a WS ``tool_call_started`` to the conversation view."""
        return self.query_one(ConversationView).append_tool_call(name, status)

    # -----------------------------------------------------------------
    # WS attach (#252 item 1) — caller supplies URI + bearer token
    # from a worker spawn (via Supervisor.spawn_worker or a direct
    # ``cothis worker`` subprocess). The app opens a client, pumps
    # inbound frames to ``ConversationView`` / ``ToolCallCard``, and
    # exposes ``send_run_turn`` for ``action_send_prompt`` to use.
    # -----------------------------------------------------------------

    async def attach_ws(self, uri: str, token: str) -> None:
        """Open a WS client to a worker; pump inbound frames to the view.

        Caller decides how the worker got spawned (Supervisor, direct
        subprocess, etc.) — this method only needs the bind-handshake
        output (URI + bearer token). Inbound frames dispatch by
        ``type`` to ``append_assistant_delta`` / ``append_tool_call``.

        Idempotent: calling again replaces the previous attachment.
        """
        import websockets

        await self.detach_ws()
        self._ws = await websockets.connect(
            uri, additional_headers={"Authorization": f"Bearer {token}"},
        )
        self._ws_pump_task = asyncio.create_task(self._pump_ws())

    async def detach_ws(self) -> None:
        """Close the WS client + cancel the pump task (if attached)."""
        task = self._ws_pump_task
        self._ws_pump_task = None
        ws = self._ws
        self._ws = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if ws is not None:
            await ws.close()

    async def send_run_turn(self, prompt: str) -> None:
        """Forward a prompt as a ``run_turn`` control message over WS.

        No-op when not attached (caller falls back to local echo). The
        ``tool_call_result_pointer`` (#254) frame on the return path
        will update the matching ``ToolCallCard`` once status wiring
        lands; for now ``run_turn`` confirms with ``assistant_delta``
        + ``tool_call_started`` frames per #255/#254.
        """
        if self._ws is None:
            return
        await self._ws.send(json.dumps({"type": "run_turn", "prompt": prompt}))

    async def _pump_ws(self) -> None:
        """Read inbound WS frames + dispatch to ConversationView.

        The websockets library surfaces a closed connection as
        ``ConnectionClosed`` raised from ``recv``; we log + let the
        task end. The connection will be re-established by the caller
        (the supervisor will restart the worker per #227).
        """
        if self._ws is None:
            return
        import websockets

        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("tui: dropping non-JSON WS frame: %r", raw)
                    continue
                self._dispatch_ws_message(msg)
        except websockets.exceptions.ConnectionClosed as exc:
            logger.info("tui: WS connection closed (code=%s)", exc.code)

    def _dispatch_ws_message(self, msg: dict) -> None:
        """Route one decoded WS message to the appropriate view method."""
        typ = msg.get("type")
        if typ == "assistant_delta":
            self.append_assistant_delta(
                msg.get("kind", "text"), msg.get("text", ""),
            )
        elif typ == "tool_call_started":
            self.append_tool_call(msg.get("tool", "?"))
        elif typ == "tool_call_result_pointer":
            # #254 result frame — update the matching ToolCallCard's
            # status. Card identity isn't stable yet (cards are mounted
            # by name without a call_id); landing the card-id wiring
            # is a follow-up under #252 item 4.
            logger.debug(
                "tui: tool_call_result_pointer for %s (is_error=%s) — "
                "card status update not yet wired",
                msg.get("tool"), msg.get("is_error"),
            )
        elif typ == "error":
            logger.warning("tui: worker error: %s", msg.get("message", ""))
        else:
            logger.debug("tui: ignoring unknown WS message type: %r", typ)


def run() -> None:
    """Entry point: ``python -m cothis.tui``."""
    app = CothisApp()
    app.run()


if __name__ == "__main__":
    run()
