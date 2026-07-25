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
from pathlib import Path
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

    ``call_id`` is the stable per-call identifier (Anthropic
    ``tool_use.id``) that pairs this card with the matching
    ``tool_call_result_pointer`` frame (#252 item 4). ``None`` for
    legacy callers that haven't been updated yet.
    """

    DEFAULT_CSS = """
    ToolCallCard {
        margin: 0 0 0 2;
        padding: 0 1;
        background: $surface;
        border-left: thick $accent;
    }
    """

    def __init__(
        self, name: str, status: str = "running", call_id: str | None = None,
    ) -> None:
        self._name = name
        self._status = status
        self._call_id = call_id
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
        # Cards indexed by ``call_id`` (#252 item 4) so result frames
        # can update the matching card's status badge without ambiguity
        # when the same tool runs twice in one turn.
        self._cards_by_call_id: dict[str, ToolCallCard] = {}

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

    def append_tool_call(
        self, name: str, status: str = "running", call_id: str | None = None,
    ) -> ToolCallCard:
        """Mount an inline tool-call card; return it for status updates.

        Flushes the active text segment (current buffer + Markdown widget)
        so the next text delta starts a fresh segment below this card.
        Without the flush, all text would accumulate in one Markdown
        widget and the card would render below all of it, regardless
        of when it was mounted — violating the "tool calls render as
        inline cards" acceptance criterion on #228 (Rule 3).

        ``call_id`` indexes the card in ``_cards_by_call_id`` so a
        subsequent ``tool_call_result_pointer`` frame can find it (#252
        item 4). ``None`` keeps the legacy un-indexed behaviour (no
        status update will land for this card).
        """
        self._refresh_markdown()
        self._text_buf = []
        self._active_markdown = None
        card = ToolCallCard(name=name, status=status, call_id=call_id)
        if call_id is not None:
            self._cards_by_call_id[call_id] = card
        self.mount(card)
        return card

    def update_tool_call_status(
        self, call_id: str, *, is_error: bool,
    ) -> ToolCallCard | None:
        """Flip a card's status badge to ``done`` / ``failed`` by call_id.

        Returns the card if found, ``None`` if no card is indexed under
        ``call_id`` (e.g. the start frame predates this wiring, or the
        card was mounted by a caller that didn't pass call_id).
        """
        card = self._cards_by_call_id.get(call_id)
        if card is None:
            return None
        card.set_status("failed" if is_error else "done")
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
    SessionList > ListItem.active-session {
        background: $boost;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("ctrl+enter", "send_prompt", "Send", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("n", "new_session", "New session", show=True),
        Binding("ctrl+m", "menu", "Menu", show=True),
    ]

    def action_new_session(self) -> None:
        """Trigger the new-session flow (#234).

        Lists git worktrees visible from ``Path.cwd()`` and forwards them
        to ``on_new_session`` — an overridable hook the subclass / caller
        wires to a picker UI. Default hook logs + returns; subclasses
        override to mount a modal that lets the user choose where to
        create the session (then call ``Session.new`` + ``attach_ws``).

        Subprocess bound: ``list_worktrees`` runs ``git worktree list``
        synchronously with a 5s timeout (the helper's safety net).
        Acceptable here because the action is user-triggered (Ctrl-N)
        and the bound timeout prevents indefinite blocking.
        """
        from cothis.git import list_worktrees

        worktrees = list_worktrees(Path.cwd())
        self.on_new_session(worktrees)

    def on_new_session(self, worktrees: list) -> None:
        """Hook called by ``action_new_session`` with the visible worktrees.

        Default: log + return. Subclasses override to mount a picker
        modal (``ModalScreen``) that lets the user choose a worktree
        for the new session. The picker UI lands in a follow-up.
        """
        logger.info(
            "tui: new-session action fired; %d worktree(s) visible",
            len(worktrees),
        )

    # -----------------------------------------------------------------
    # Menu binding (#235 slice A) — Ctrl-M opens the config menu.
    # The modal listing skills / MCP / LSP servers lands in Slice B;
    # this is the binding + dispatch contract only.
    # -----------------------------------------------------------------

    def action_menu(self) -> None:
        """Trigger the config menu (#235).

        Calls ``on_menu_open`` — an overridable hook the subclass wires
        to a ``ModalScreen`` that lists discoverable skills, MCP servers,
        and LSP servers. Default: log + return.
        """
        self.on_menu_open()

    def on_menu_open(self) -> None:
        """Hook fired by ``action_menu`` (Ctrl-M).

        Default: log + return. Subclasses override to mount a modal
        (Slice B) that lists skills via ``discover_tools``, MCP servers
        via ``MCPServer``, and any LSP servers. Selecting entries
        re-runs ``discover_tools`` with the chosen layers (Slice C/D).
        """
        logger.info("tui: menu action fired (Ctrl-M)")

    def list_configurable_skills(self) -> list[str]:
        """Return the names of skills discoverable from the current cwd.

        Wraps ``cothis.skills.discover_skills`` so the menu modal
        (Slice B, not yet implemented) can display the list without
        importing the skills module directly. Returns an empty list
        when no skills are installed.
        """
        from cothis.skills import discover_skills

        return [s.name for s in discover_skills(Path.cwd())]

    # WS attach state (#252 item 1). ``None`` until ``attach_ws`` runs;
    # ``attach_ws`` re-uses these slots idempotently. Typed as ``Any``
    # because websockets' client connection class moved across versions
    # (``ClientConnection`` in v13+, ``WebSocketClientProtocol`` pre-v13)
    # and we don't need to call any methods on it outside this file.
    _ws: Any = None
    _ws_pump_task: asyncio.Task[None] | None = None
    # Multi-session WS connections (#230 slice B). Keyed by session_id;
    # each entry has its own pump task. The single-session ``_ws`` /
    # ``_ws_pump_task`` above stay for backward compat (``attach_ws``).
    _ws_by_session: dict[str, Any] = {}
    _ws_pump_tasks_by_session: dict[str, asyncio.Task[None]] = {}
    # Active session id (#230 slice A) — the session the user is currently
    # interacting with. Future slices route ``send_run_turn`` to the active
    # session's WS + highlight the entry in ``SessionList``.
    _active_session_id: str | None = None

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

    async def action_send_prompt(self) -> None:
        """Read InputBar text → render locally → forward to worker if attached.

        Local echo always runs (the user expects to see their prompt
        immediately). When a WS is attached (#252 item 1), the prompt
        is also forwarded as a ``run_turn`` control message — the
        worker drives the assistant-side rendering via subsequent
        ``assistant_delta`` frames pumped by ``_pump_ws``.

        Textual actions can be async; the framework awaits coroutine
        results, so ``await self.send_run_turn(text)`` blocks the
        action until the frame is on the wire (typically <1 ms).
        """
        bar = self.query_one(InputBar)
        text = bar.get_text().strip()
        if not text:
            return
        view = self.query_one(ConversationView)
        view.append_user_message(text)
        bar.clear()
        if self._ws is not None:
            await self.send_run_turn(text)

    def append_assistant_delta(self, kind: str = "text", text: str = "") -> None:
        """Forward a WS ``assistant_delta`` to the conversation view.

        ``kind`` defaults to ``"text"`` for mixed-version compatibility
        (old servers without the ``kind`` field in the WS message).
        """
        self.query_one(ConversationView).append_delta(kind, text)

    def append_tool_call(
        self, name: str, status: str = "running", call_id: str | None = None,
    ) -> Any:
        """Forward a WS ``tool_call_started`` to the conversation view."""
        return self.query_one(ConversationView).append_tool_call(
            name, status, call_id=call_id,
        )

    def refresh_session_list(self, db_path: Path) -> None:
        """Repopulate ``SessionList`` from the session storage DB.

        Opens ``Storage`` transiently for the read; no fcntl lock is
        acquired on read-only access (the worker's lock is on its own
        write connection). Closes the connection immediately so the
        TUI doesn't hold a long-running reader on the worker's DB.

        Sessions visible from ``Path.cwd()`` (the user's current
        directory tree) are listed; others are filtered out by
        ``list_sessions_in_cwd_tree``.

        Failures (missing DB, corrupt schema) log a warning + leave
        the existing list intact — the TUI stays usable without a
        session picker if the storage layer is unavailable.
        """
        from cothis.session.storage import Storage

        try:
            storage = Storage(db_path)
        except Exception as exc:  # noqa: BLE001 — best-effort UI populate
            logger.warning("tui: cannot open session DB %s: %s", db_path, exc)
            return
        try:
            rows = storage.list_sessions_in_cwd_tree(Path.cwd())
        except Exception as exc:  # noqa: BLE001
            logger.warning("tui: cannot list sessions in %s: %s", db_path, exc)
            return
        finally:
            storage.close()

        session_list = self.query_one(SessionList)
        session_list.clear()
        # Look up worktrees once; each session's label is enriched with
        # its worktree's branch when the session cwd belongs to a known
        # worktree (#234 AC #3). Failure to list worktrees (not a git
        # repo, git binary missing) degrades to plain cwd labels — the
        # list stays usable.
        from cothis.git import find_worktree_for_path, list_worktrees

        worktrees = list_worktrees(Path.cwd())
        for row in rows:
            label = row.title or f"session {row.id[:8]}"
            cwd_hint = str(row.cwd) if row.cwd else "(no cwd)"
            wt = (
                find_worktree_for_path(Path(row.cwd), worktrees)
                if row.cwd else None
            )
            if wt is not None and wt.branch is not None:
                cwd_hint = f"{cwd_hint} · branch:{wt.branch}"
            # Parens (not square brackets) — Textual parses ``[...]`` as
            # markup tags, so a bracketed cwd path raises MarkupError.
            # ``id`` prefix ``s_`` because Textual IDs can't begin with a
            # number — session ids are hex and may start with a digit.
            # ``on_list_view_selected`` strips the prefix.
            session_list.append(
                ListItem(Label(f"{label}  ({cwd_hint})"), id=f"s_{row.id}")
            )

    # -----------------------------------------------------------------
    # Session selection (#252 item 5 — selection half; list-half landed
    # in #280). The user clicks a ListItem in SessionList; ListView
    # posts a ``Selected`` event; the handler reads the session id off
    # the item (set as Textual ``id`` by ``refresh_session_list``) and
    # calls ``on_session_selected`` — a hook subclasses / tests can
    # override to wire up spawn-and-attach.
    # -----------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Read the selected ListItem's session id; call the hook."""
        # The event also fires for ListView subclasses; we only care
        # about SessionList. Both have the same API, so this handler
        # is fine as-is — but narrow explicitly to SessionList to avoid
        # triggering on any future ListView in the app.
        if not isinstance(event.list_view, SessionList):
            return
        session_id = event.item.id
        if not session_id or not session_id.startswith("s_"):
            return
        self.on_session_selected(session_id[2:])

    def on_session_selected(self, session_id: str) -> None:
        """Hook called when the user picks a session in SessionList.

        Default behaviour: ``set_active_session(session_id)`` + log.
        Callers that want to spawn a worker + attach WS on selection
        subclass ``CothisApp`` and override this method, OR monkeypatch
        the bound method on an existing instance.
        """
        self.set_active_session(session_id)

    # -----------------------------------------------------------------
    # Active-session tracking (#230 slice A)
    # -----------------------------------------------------------------

    def set_active_session(self, session_id: str) -> None:
        """Mark ``session_id`` as the active session + fire the change hook.

        Called by ``on_session_selected`` and by callers that spawn a
        new session (``on_new_session`` override → spawn → ``set_active``).
        Future slices use this to route ``send_run_turn`` to the right
        WS connection (#230 slice C) + highlight the focused entry.
        """
        previous = self._active_session_id
        self._active_session_id = session_id
        if previous != session_id:
            self.on_active_session_changed(session_id)

    def on_active_session_changed(self, session_id: str) -> None:
        """Hook fired when the active session changes (#230 slice A/D).

        Default: update SessionList visual highlight — the matching
        ListItem gains ``active-session`` CSS class; all others lose
        it. Subclasses can override for additional effects (input
        focus routing etc.) but should call ``super().on_active_session_changed()``
        to preserve the highlight.
        """
        logger.info("tui: active session changed → %s", session_id)
        try:
            session_list = self.query_one(SessionList)
        except Exception:  # noqa: BLE001 — compose may not have run yet
            return
        target_id = f"s_{session_id}"
        for item in session_list.query(ListItem):
            if item.id == target_id:
                item.add_class("active-session")
            else:
                item.remove_class("active-session")

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

    # -----------------------------------------------------------------
    # Multi-session WS attach (#230 slice B)
    # -----------------------------------------------------------------

    async def attach_session_ws(
        self, session_id: str, uri: str, token: str,
    ) -> None:
        """Open a WS client for a specific session (multi-session #230).

        Stores the connection in ``_ws_by_session`` keyed by
        ``session_id`` + starts a dedicated pump task. Marks the
        session as active via ``set_active_session``. Idempotent:
        re-attaching replaces the previous connection for that session.
        """
        import websockets

        await self.detach_session_ws(session_id)
        ws = await websockets.connect(
            uri, additional_headers={"Authorization": f"Bearer {token}"},
        )
        self._ws_by_session[session_id] = ws
        self._ws_pump_tasks_by_session[session_id] = asyncio.create_task(
            self._pump_ws_connection(ws)
        )
        self.set_active_session(session_id)

    async def detach_session_ws(self, session_id: str) -> None:
        """Close + remove one session's WS connection (multi-session #230)."""
        task = self._ws_pump_tasks_by_session.pop(session_id, None)
        ws = self._ws_by_session.pop(session_id, None)
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

        Routes to the active session's WS when multi-session is in use
        (``_ws_by_session``); falls back to the single-session ``_ws``
        for backward compat. No-op when neither is attached.
        """
        ws = self._ws_by_session.get(self._active_session_id or "") or self._ws
        if ws is None:
            return
        await ws.send(json.dumps({"type": "run_turn", "prompt": prompt}))

    async def _pump_ws(self) -> None:
        """Read inbound WS frames from ``self._ws`` (single-session path)."""
        if self._ws is None:
            return
        await self._pump_ws_connection(self._ws)

    async def _pump_ws_connection(self, ws: Any) -> None:
        """Read inbound WS frames from a specific connection + dispatch.

        Shared between single-session (``_pump_ws``) and multi-session
        (``attach_session_ws``) paths. Parameterized on ``ws`` so
        concurrent pump tasks don't race on ``self._ws`` (#230).
        """
        import websockets

        try:
            async for raw in ws:
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
            self.append_tool_call(
                msg.get("tool", "?"), call_id=msg.get("call_id"),
            )
        elif typ == "tool_call_result_pointer":
            # #252 item 4: flip the matching card's status badge by
            # call_id. Falls back to a debug log when the card isn't
            # found (start frame predated call_id wiring, or a stale
            # result arrives after the user cleared the view).
            call_id = msg.get("call_id")
            if call_id is None:
                logger.debug(
                    "tui: tool_call_result_pointer without call_id for "
                    "tool %s — cannot pair with a card",
                    msg.get("tool"),
                )
                return
            view = self.query_one(ConversationView)
            card = view.update_tool_call_status(
                call_id, is_error=bool(msg.get("is_error")),
            )
            if card is None:
                logger.debug(
                    "tui: tool_call_result_pointer for %s (call_id=%s) — "
                    "no matching card; dropping",
                    msg.get("tool"), call_id,
                )
        elif typ == "ask_user_request":
            # #229 slice C: forward to the overridable hook. Default
            # auto-rejects (sends resolve_ask with value=None) so the
            # worker doesn't block in tests; subclasses mount a modal
            # (Slice E).
            self.on_ask_user_request(
                ask_id=msg.get("ask_id", ""),
                prompt=msg.get("prompt", ""),
                choices=msg.get("choices", []),
            )
        elif typ == "error":
            logger.warning("tui: worker error: %s", msg.get("message", ""))
        else:
            logger.debug("tui: ignoring unknown WS message type: %r", typ)

    def on_ask_user_request(
        self, *, ask_id: str, prompt: str, choices: list,
    ) -> None:
        """Hook fired when the worker asks the user for input (#229 slice C).

        Default: auto-reject — send ``resolve_ask`` with ``value=None``
        so the worker's Future (Slice D) resolves + the tool returns
        promptly. Subclasses override to mount a modal (Slice E) that
        shows ``prompt`` + ``choices`` + sends the user's pick back.
        """
        logger.info("tui: ask_user_request %s: %s", ask_id, prompt)
        if self._ws is not None:
            asyncio.create_task(self._ws.send(json.dumps({
                "type": "resolve_ask", "ask_id": ask_id, "value": None,
            })))


def run() -> None:
    """Entry point: ``python -m cothis.tui``."""
    app = CothisApp()
    app.run()


if __name__ == "__main__":
    run()
