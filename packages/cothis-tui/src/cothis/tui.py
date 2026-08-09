"""``cothis.tui`` — focused transcript TUI.

The shell follows pi's alternate-screen model (``tui-plan.md``): one
full-height scrollable transcript with a fixed working dock beneath it,
and focus that always returns to the composer.

- ``ConversationView`` — the transcript. Full viewport, the ONLY
  scrolling context region.
- ``#composer`` — fixed dock: one ``TextArea`` input + a shortcut hint.
  Auto-grows with content, scrolls internally past its cap.
- ``CothisFooter`` — fixed one-line status dock (model / session short-id
  when multi-session / ctx pressure / skills / run state).
- Session navigation is TRANSIENT (``/sessions`` opens a picker overlay),
  never a permanent sidebar.
- The input owns focus at launch and after every session switch / modal
  dismissal — pi's editor-always-focused contract.

Stream routing per the design-review sign-off (#228, 2026-07-24):
``ContentDelta(kind="text")`` renders as normal assistant content;
``ContentDelta(kind="thinking")`` renders as a collapsed, dimmed
``Collapsible`` (expand to read the model's reasoning). Tool calls render
as inline cards with a status badge.

WS attach (``attach_ws`` / ``attach_session_ws``) + ``run_turn``
forwarding (``send_run_turn``) land via the worker's WS bridge;
``on_worktree_pick`` is the spawn contract for production CLI wiring.
Esc-to-interrupt: ``Binding('escape','interrupt_turn')`` cancels the
in-flight turn via the worker's ``interrupt_turn`` control message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
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
    from textual.timer import Timer

    from cothis.git import Worktree

logger = logging.getLogger(__name__)

_TOOL_STATUS_ICONS = {"running": ">>", "done": "OK", "failed": "XX"}

# Streaming-render throttle + finalisation (#407). Re-parsing Markdown on
# every text delta is O(S²) in the segment size. While streaming, deltas
# accumulate into a plain ``Static`` (no Markdown parse) refreshed at most
# every ``_STREAM_REFRESH_S``; the segment is parsed into a ``Markdown``
# widget ONCE, ``_STREAM_FINALIZE_S`` after the last delta (an idle-debounce
# proxy for turn-end — the worker emits no turn-end frame) or at a
# tool-call boundary. Net per-segment cost: O(S) appends + one O(S) parse.
_STREAM_REFRESH_S = 0.05
_STREAM_FINALIZE_S = 0.3


# ---------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------


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
        self,
        name: str,
        status: str = "running",
        call_id: str | None = None,
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
    """Full-viewport transcript — scrollable Markdown + tool-call cards.

    ``append_delta`` is the primary API the WS client calls per
    ``assistant_delta`` message. ``append_tool_call`` mounts a card
    between text segments so DOM order matches event order (each
    card flushes the active text segment; the next delta starts a
    fresh segment below the card).
    """

    DEFAULT_CSS = """
    ConversationView {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
        scrollbar-gutter: stable;
    }
    ConversationView > Markdown.user-message {
        background: $panel;
        color: $text;
        padding: 1 2;
        margin: 0 0 1 0;
        border-left: thick $accent;
    }
    ConversationView > Collapsible.thinking-block {
        margin: 0 0 0 2;
        padding: 0 1;
        border-left: thick $primary-darken-2;
        color: $text-disabled;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        # ``list[str]`` accumulator (#267): ``list.append`` is amortised O(1),
        # so the per-delta append path stays linear. ``renderable_str`` joins
        # lazily — only the final Markdown parse needs the joined string.
        self._text_buf: list[str] = []
        # Thinking-segment accumulator. Kept separate from ``_text_buf`` so
        # ``renderable_str`` (the text-segment source, read by tests +
        # inspection) stays free of reasoning content. Finalised into a
        # collapsed, dimmed ``Collapsible``.
        self._thinking_buf: list[str] = []
        # Plain-text widget shown WHILE a segment streams (#407). Mounting it
        # avoids re-parsing Markdown on every delta; swapped for a ``Markdown``
        # widget (one parse) at finalisation.
        self._stream_static: Static | None = None
        # Monotonic timestamp of the last plain-text refresh (throttle).
        self._last_stream_refresh: float = 0.0
        # Idle-finalise debounce timer (#407): rearmed per delta; fires
        # ``_STREAM_FINALIZE_S`` after the LAST delta to parse Markdown once.
        self._finalize_timer: Timer | None = None
        # True once the current buffer has been parsed into a mounted Markdown
        # widget. Gates idempotent re-finalise and signals ``append_delta`` to
        # start a fresh segment when text resumes. The buffer is RETAINED
        # across finalise so ``renderable_str`` still reflects the segment.
        self._finalized: bool = False
        # Cards indexed by ``call_id`` (#252 item 4) so result frames can
        # update the matching card's status badge without ambiguity when the
        # same tool runs twice in one turn.
        self._cards_by_call_id: dict[str, ToolCallCard] = {}

    @property
    def renderable_str(self) -> str:
        """Accumulated text-delta source for the active segment — tests + debugging."""
        return "".join(self._text_buf)

    def append_delta(self, kind: str, text: str) -> None:
        """Route a ContentDelta to the right rendering path.

        ``kind="text"`` → accumulate (O(1)) + cheap plain-text refresh; the
        segment is parsed into Markdown ONCE at finalisation, not per delta
        (#407). ``kind="thinking"`` → accumulate into ``_thinking_buf`` and
        finalise into a collapsed, dimmed ``Collapsible``. A kind switch
        finalises the active segment first, so each kind renders as its own
        block in event order.
        """
        if kind == "text":
            # Close any streaming thinking segment so text is its own block.
            self._finalize_thinking()
            if self._finalized:
                self._text_buf = []
                self._finalized = False
            self._text_buf.append(text)
            self._refresh_stream()
            self._arm_finalize()
        elif kind == "thinking":
            self._finalize_segment()
            self._thinking_buf.append(text)
            self._arm_finalize()

    def append_user_message(self, text: str) -> None:
        """Render a user prompt as a background-tinted block (pi's ``userMessageBg`` box).

        Finalises any streaming segment first so the prompt is its own
        block, then mounts the prompt as a Markdown widget with the
        ``.user-message`` class — a full-width tinted box, not a ``you:``
        prefix line. Text is Markdown-escaped (brackets) so injected
        links or markup can't activate inside the widget.
        """
        safe = text.replace("[", "\\[").replace("]", "\\]")
        self._finalize_active()
        self._text_buf = []
        self._finalized = False
        at_bottom = self._at_bottom()
        self.mount(Markdown(safe, classes="user-message"))
        self._follow(at_bottom)

    def append_tool_call(
        self,
        name: str,
        status: str = "running",
        call_id: str | None = None,
    ) -> ToolCallCard:
        """Mount an inline tool-call card; return it for status updates.

        Finalises the active text segment (parses its Markdown once) and
        resets the buffer so the next text delta starts a fresh segment
        below this card.
        """
        self._finalize_active()
        self._text_buf = []
        self._finalized = False
        card = ToolCallCard(name=name, status=status, call_id=call_id)
        if call_id is not None:
            self._cards_by_call_id[call_id] = card
        at_bottom = self._at_bottom()
        self.mount(card)
        self._follow(at_bottom)
        return card

    def update_tool_call_status(
        self,
        call_id: str,
        *,
        is_error: bool,
    ) -> ToolCallCard | None:
        """Flip a card's status badge to ``done`` / ``failed`` by call_id."""
        card = self._cards_by_call_id.get(call_id)
        if card is None:
            return None
        card.set_status("failed" if is_error else "done")
        return card

    def render_replayed_message(self, msg: dict) -> None:
        """Render one rebuilt ``{role, content: [blocks]}`` message.

        Replay-on-attach reuses the existing primitives — there is no
        parallel renderer. A user text block routes through
        ``append_user_message``; an assistant text block mounts one Markdown
        segment; a tool_use block mounts a card (tool_result blocks are
        skipped — the matching card already renders on the assistant side).
        """
        role = msg.get("role")
        content = msg.get("content") or []
        if role == "user":
            texts = [b.get("text", "") for b in content if b.get("type") == "text"]
            if texts:
                self.append_user_message(" ".join(texts))
            return
        if role != "assistant":
            return
        for block in content:
            btype = block.get("type")
            if btype == "text":
                self.append_delta("text", block.get("text", ""))
            elif btype == "thinking":
                self.append_delta(
                    "thinking",
                    block.get("thinking") or block.get("text", ""),
                )
            elif btype == "tool_use":
                name = block.get("name") or block.get("tool") or "?"
                self.append_tool_call(
                    name,
                    status="done",
                    call_id=block.get("id") or block.get("call_id"),
                )
            # image / tool_result: deferred.
        # Replay is a batch operation: force-finalise any pending segment so
        # the last assistant text mounts deterministically instead of waiting
        # on the idle-finalise timer (which a test pause may not reach).
        self._finalize_active()

    def clear(self) -> None:
        """Unmount every rendered message + reset the streaming state."""
        if self._finalize_timer is not None:
            self._finalize_timer.stop()
            self._finalize_timer = None
        for child in list(self.children):
            child.remove()
        self._stream_static = None
        self._text_buf = []
        self._thinking_buf = []
        self._cards_by_call_id = {}
        self._finalized = False

    def _refresh_stream(self) -> None:
        """Mount/refresh the plain-text streaming widget, throttled."""
        if self._stream_static is None:
            at_bottom = self._at_bottom()
            self._stream_static = Static("".join(self._text_buf))
            self.mount(self._stream_static)
            self._last_stream_refresh = time.monotonic()
            self._follow(at_bottom)
            return
        now = time.monotonic()
        if now - self._last_stream_refresh >= _STREAM_REFRESH_S:
            at_bottom = self._at_bottom()
            self._stream_static.update("".join(self._text_buf))
            self._last_stream_refresh = now
            self._follow(at_bottom)

    def _arm_finalize(self) -> None:
        """(Re)arm the idle-finalise debounce — flush both segments once streaming settles."""
        if self._finalize_timer is not None:
            self._finalize_timer.stop()
        self._finalize_timer = self.set_timer(
            _STREAM_FINALIZE_S,
            self._finalize_active,
        )

    def _finalize_active(self) -> None:
        """Flush whichever segment(s) are streaming — text and/or thinking."""
        self._finalize_segment()
        self._finalize_thinking()

    def _mount_thinking_block(self, text: str) -> None:
        """Mount one thinking block as a collapsed, dimmed ``Collapsible``."""
        at_bottom = self._at_bottom()
        self.mount(
            Collapsible(
                Markdown(text),
                title="reasoning",
                classes="thinking-block",
            )
        )
        self._follow(at_bottom)

    def _finalize_thinking(self) -> None:
        """Mount the accumulated thinking as a collapsed ``Collapsible``."""
        if not self._thinking_buf:
            return
        source = "".join(self._thinking_buf)
        self._thinking_buf = []
        self._mount_thinking_block(source)

    def _finalize_segment(self) -> None:
        """Swap the streaming ``Static`` for a ``Markdown`` widget (one parse)."""
        if self._finalize_timer is not None:
            self._finalize_timer.stop()
            self._finalize_timer = None
        if self._finalized or not self._text_buf:
            return
        at_bottom = self._at_bottom()
        source = "".join(self._text_buf)
        md = Markdown(source)
        if self._stream_static is not None:
            self._stream_static.remove()
            self._stream_static = None
        self._finalized = True
        self.mount(md)
        self._follow(at_bottom)

    def _at_bottom(self) -> bool:
        """True when the view is within a line of the bottom (#409)."""
        return self.scroll_y >= self.max_scroll_y - 1

    def _follow(self, was_at_bottom: bool) -> None:
        """Re-pin to the bottom iff the user was already there."""
        if was_at_bottom:
            self.scroll_end(animate=False, immediate=True)


# ---------------------------------------------------------------------
# Modals
# ---------------------------------------------------------------------


class ConfigMenuModal(ModalScreen[set[str] | None]):
    """Skill selection menu (Ctrl-M)."""

    DEFAULT_CSS = """
    ConfigMenuModal {
        align: center middle;
    }
    ConfigMenuModal > Label {
        padding: 0 2;
        width: 100%;
    }
    """

    BINDINGS = [("escape", "dismiss_modal", "Cancel")]

    def __init__(self, skills: list[str], *, selected: set[str] | None = None) -> None:
        self._skills = skills
        self._selected: set[str] = set(selected) if selected else set()
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Label("Active skills (click to toggle)", id="config-prompt")
        for name in self._skills:
            classes = (
                "skill-toggle -active" if name in self._selected else "skill-toggle"
            )
            yield Button(
                f"{'[x]' if name in self._selected else '[ ]'} {name}",
                id=f"skill-{name}",
                classes=classes,
            )
        yield Button("Done", id="menu-done")

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "menu-done":
            self.dismiss(self._selected)
        elif bid.startswith("skill-"):
            name = bid[len("skill-") :]
            if name in self._selected:
                self._selected.discard(name)
            else:
                self._selected.add(name)
            event.button.label = f"{'[x]' if name in self._selected else '[ ]'} {name}"


class AskUserModal(ModalScreen[str | None]):
    """Mid-turn question from a tool (``ask_user``)."""

    DEFAULT_CSS = """
    AskUserModal {
        align: center middle;
    }
    AskUserModal > Label {
        padding: 0 2;
        width: 100%;
    }
    """

    BINDINGS = [("escape", "dismiss_modal", "Cancel")]

    def __init__(self, prompt: str, choices: list[str]) -> None:
        self._prompt = prompt
        self._choices = choices
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Label(self._prompt, id="ask-prompt")
        for choice in self._choices:
            yield Button(choice, id=f"choice-{choice}")
        yield Button("Cancel", id="ask-cancel")

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "ask-cancel":
            self.dismiss(None)
        elif bid.startswith("choice-"):
            self.dismiss(bid[len("choice-") :])


class WorktreePickerModal(ModalScreen[str | None]):
    """Choose a git worktree for a new session."""

    DEFAULT_CSS = """
    WorktreePickerModal {
        align: center middle;
    }
    WorktreePickerModal > Label {
        padding: 0 2;
        width: 100%;
    }
    """

    BINDINGS = [("escape", "dismiss_modal", "Cancel")]

    def __init__(self, worktrees: list[Worktree]) -> None:
        self._worktrees = list(worktrees)
        super().__init__()

    def compose(self) -> ComposeResult:
        if not self._worktrees:
            yield Label(
                "No worktrees found. Pick the current directory below, "
                "or run `git worktree add <path>` outside cothis, then retry.",
                id="worktree-prompt",
            )
        else:
            yield Label("Pick a worktree for the new session", id="worktree-prompt")
            for i, wt in enumerate(self._worktrees):
                label = wt.branch or wt.path.name
                yield Button(label, id=f"wt-{i}")
        yield Button("Current directory", id="worktree-cwd")
        yield Button("Cancel", id="worktree-cancel")

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "worktree-cancel":
            self.dismiss(None)
        elif bid == "worktree-cwd":
            self.dismiss(str(Path.cwd()))
        elif bid.startswith("wt-"):
            idx = int(bid[len("wt-") :])
            self.dismiss(str(self._worktrees[idx].path))


class SessionPickerModal(ModalScreen[str | None]):
    """Transient session switcher — never a permanent sidebar.

    One row per known session (id, label); the active session is
    marked ``•``. Enter on a row (or a click) dismisses with the
    session id; Esc / Cancel dismisses with ``None``. Focus lands on
    the list so the keyboard can drive the switch immediately.
    """

    DEFAULT_CSS = """
    SessionPickerModal {
        align: center middle;
        background: $background 80%;
    }
    #session-picker {
        width: 72;
        max-width: 90%;
        height: 70%;
        max-height: 24;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }
    #session-picker-title {
        height: 1;
        color: $text;
        text-style: bold;
    }
    #session-picker-list {
        height: 1fr;
        margin: 1 0;
    }
    #session-picker-cancel {
        width: 100%;
    }
    """

    BINDINGS = [("escape", "dismiss_modal", "Cancel")]

    def __init__(self, sessions: list[tuple[str, str]], active: str | None) -> None:
        self._sessions = sessions
        self._active = active
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(id="session-picker"):
            yield Label("Sessions", id="session-picker-title")
            yield ListView(
                *[
                    ListItem(
                        Label(
                            ("• " if session_id == self._active else "  ") + label,
                        ),
                        id=f"p_{session_id}",
                    )
                    for session_id, label in self._sessions
                ],
                id="session-picker-list",
            )
            yield Button("Cancel", id="session-picker-cancel")

    def on_mount(self) -> None:
        self.query_one("#session-picker-list", ListView).focus()

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        if item_id.startswith("p_"):
            self.dismiss(item_id[2:])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "session-picker-cancel":
            self.dismiss(None)


# ---------------------------------------------------------------------
# Status dock
# ---------------------------------------------------------------------


class CothisFooter(Static):
    """One-line status dock — fixed beneath the composer.

    Renders up to five cells left-to-right:

    ``model | [session:<short-id> |] ctx:<pressure> | skills:[a,b] | state:<run_state>``

    * ``<short-id>`` — first 8 chars of the active session id. Shown ONLY
      when more than one session is attached; in the common single-session
      case the id is redundant noise and is hidden.
    * ``<pressure>`` — the ``PressureLevel`` value string (``none`` /
      ``low`` / ``medium`` / ``high`` / ``critical``) or ``?`` when unknown.
    * ``skills`` — comma-joined sorted active-skills set, or ``-`` when empty.
    * ``run_state`` — ``idle`` / ``running`` / ``interrupted``.

    The widget holds no state; it is repainted by ``CothisApp``'s combined
    ``_refresh_footer`` watcher whenever one of the footer reactives flips.
    """

    DEFAULT_CSS = """
    CothisFooter#footer {
        height: 1;
        background: $boost;
        color: $text-disabled;
        padding: 0 1;
    }
    """


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------


class CothisApp(App):
    """Focused transcript shell with a transient session index.

    Composition (top → bottom):

    - ``ConversationView`` — full-viewport scrollable transcript.
    - ``#composer`` — fixed dock: input + shortcut hint.
    - ``CothisFooter`` — fixed status dock.

    The input is focused at launch and restored after every transient UI
    (pi's editor-always-focused model). All app commands use modified
    keys so none are shadowed by the focused input.

    Keymap:

    | Ctrl+Enter | send prompt |
    | Ctrl+N     | new session |
    | Ctrl+M     | config menu |
    | Esc        | interrupt / clear / dismiss overlay |
    | Ctrl+C     | quit |
    """

    TITLE = "cothis"
    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }
    ConversationView {
        height: 1fr;
        width: 1fr;
    }
    #composer {
        height: auto;
        min-height: 6;
        max-height: 12;
        padding: 0 1;
        border-top: solid $panel;
        background: $surface;
    }
    #status-line {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    TextArea#input {
        height: auto;
        min-height: 3;
        max-height: 8;
        border: round $panel;
    }
    TextArea#input:focus {
        border: round $accent;
    }
    #composer-hint {
        height: 1;
        padding: 0 1;
        color: $text-disabled;
    }
    CothisFooter#footer {
        height: 1;
        background: $boost;
        color: $text-disabled;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+enter", "send_prompt", "Send", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
        # Ctrl+N (not bare ``n``): the input holds focus by default and bare
        # keys are text, so a bare ``n`` would type into the prompt instead of
        # opening the picker. Modified keys route through the app binding even
        # while the input is focused.
        Binding("ctrl+n", "new_session", "New session", show=True),
        Binding("ctrl+m", "menu", "Menu", show=True),
        # Esc → interrupt a running turn. Non-priority: modals install their
        # own Esc binding to dismiss, and Textual resolves bindings
        # top-screen-first — the modal's Esc wins while it's open.
        Binding("escape", "interrupt_turn", "Interrupt", show=False),
    ]

    # -----------------------------------------------------------------
    # Footer reactives — the status dock repaints through one combined
    # watcher whenever any cell flips.
    # -----------------------------------------------------------------
    run_state: reactive[str] = reactive("idle")
    footer_model: reactive[str] = reactive("")
    footer_session: reactive[str] = reactive("")
    footer_pressure: reactive[str] = reactive("")
    footer_skills: reactive[list[str]] = reactive[list[str]](list)

    # WS attach state (#252 item 1). ``None`` until ``attach_ws`` runs.
    # Mutable collections are instance attrs (``__init__``), not class attrs,
    # so concurrent app instances never share state.

    def __init__(self) -> None:
        super().__init__()
        self._ws: Any = None
        self._ws_pump_task: asyncio.Task[None] | None = None
        # Multi-session WS connections (#230). Keyed by session_id; each
        # entry has its own pump task.
        self._ws_by_session: dict[str, Any] = {}
        self._ws_pump_tasks_by_session: dict[str, asyncio.Task[None]] = {}
        # Active session id (#230) — the session the user is interacting with.
        self._active_session_id: str | None = None
        # Session DB last used to populate the session index / replay history.
        self._db_path: Path | None = None
        # Session index for the transient picker: (id, label) rows populated
        # by ``refresh_session_list``. The transcript never hosts a sidebar;
        # this is the single source the ``/sessions`` picker renders from.
        self._session_rows: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        # The transcript owns all flexible height. Everything below it is
        # fixed to the bottom — pi's transcript + dock model:
        #   [status line] [input] [hint]  above the status bar.
        yield ConversationView()
        with Vertical(id="composer"):
            yield Label("", id="status-line")
            yield TextArea(id="input")
            yield Static(
                "Ctrl+Enter send  ·  /sessions switch  ·  Ctrl+N new session  ·  Ctrl+M menu",
                id="composer-hint",
            )
        yield CothisFooter("", id="footer")

    async def on_mount(self) -> None:
        """Focus the composer input on launch — the persistent-focus contract."""
        self._refresh_footer()
        self._refocus_input()

    def _refocus_input(self) -> None:
        """Return focus to the composer input — the default + persistent focus.

        Textual does NOT restore focus when a modal pops (verified
        empirically: focus stays on the modal's button), so every dismiss
        callback re-focuses explicitly — pi's editor-always-focused model.
        Guarded for the not-yet-mounted case.
        """
        with suppress(Exception):  # compose may not have run yet
            self.query_one("#input", TextArea).focus()

    # -----------------------------------------------------------------
    # New session (#234) — Ctrl+N → worktree picker → on_worktree_pick.
    # -----------------------------------------------------------------

    def action_new_session(self) -> None:
        """List git worktrees visible from ``Path.cwd()``; open the picker."""
        from cothis.git import list_worktrees

        worktrees = list_worktrees(Path.cwd())
        self.on_new_session(worktrees)

    def on_new_session(self, worktrees: list) -> None:
        """Mount ``WorktreePickerModal``; route the chosen path to ``on_worktree_pick``."""

        def _on_dismiss(value: str | None) -> None:
            self._refocus_input()
            if value is None:
                logger.info("tui: new-session cancelled (no worktree picked)")
                return
            self.on_worktree_pick(value)

        self.push_screen(WorktreePickerModal(worktrees), _on_dismiss)

    def on_worktree_pick(self, path: str) -> None:
        """Hook fired when the user picks a worktree for a new session.

        The CLI / caller overrides this to call ``Supervisor.spawn_worker``
        + ``SessionStorage.new`` + ``attach_session_ws`` with the picked
        path as the session cwd.
        """
        logger.info(
            "tui: worktree picked for new session: %s "
            "(spawn + attach wiring lands in the CLI integration)",
            path,
        )

    # -----------------------------------------------------------------
    # Config menu (#235) — Ctrl-M.
    # -----------------------------------------------------------------

    def action_menu(self) -> None:
        self.on_menu_open()

    def on_menu_open(self) -> None:
        """Mount ``ConfigMenuModal`` listing discoverable skills; persist on Done."""

        def _on_config_done(selected: set[str] | None) -> None:
            self._refocus_input()
            if selected is None:  # Esc / Cancel
                return
            from cothis.skills import save_skill_selection

            save_skill_selection(selected)

        skills = self.list_configurable_skills()
        from cothis.skills import load_skill_selection

        saved = load_skill_selection()
        self.push_screen(
            ConfigMenuModal(skills, selected=saved & set(skills)),
            _on_config_done,
        )

    def list_configurable_skills(self) -> list[str]:
        """Return the names of skills discoverable from the current cwd."""
        from cothis.skills import discover_skills

        return [s.name for s in discover_skills(Path.cwd())]

    # -----------------------------------------------------------------
    # Session index + transient picker (``/sessions``).
    # -----------------------------------------------------------------

    def _picker_rows(self) -> list[tuple[str, str]]:
        """(id, label) rows for the picker: attached WS ids + the index.

        Attached sessions (live WS) are always shown, labeled by short id;
        the persisted index adds known sessions. De-duplicated by id.
        """
        rows: dict[str, str] = {}
        for sid in self._ws_by_session:
            rows[sid] = f"{sid[:8]}  (live)"
        for sid, label in self._session_rows:
            rows.setdefault(sid, label)
        return list(rows.items())

    def action_sessions(self) -> None:
        """Open the transient session picker (``/sessions`` command)."""
        rows = self._picker_rows()
        if not rows:
            logger.info("tui: no sessions to switch to")
            self._refocus_input()
            return

        def _on_dismiss(session_id: str | None) -> None:
            self._refocus_input()
            if session_id is None:
                return
            self.on_session_selected(session_id)

        self.push_screen(
            SessionPickerModal(rows, self._active_session_id),
            _on_dismiss,
        )

    # -----------------------------------------------------------------
    # Prompt input + slash commands.
    # -----------------------------------------------------------------

    async def action_send_prompt(self) -> None:
        """Read input text → render locally → forward to worker if attached.

        Slash-prefixed commands (``/``) are intercepted BEFORE local echo:
        ``/sessions`` opens the transient picker; ``/session <id>`` (alias
        ``/switch``) changes the active session without echoing. An unknown
        ``/...`` returns False from ``_handle_slash_command`` and falls
        through to the normal prompt path so agent-side slash commands
        still work.
        """
        input_widget = self.query_one("#input", TextArea)
        text = input_widget.text.strip()
        if not text:
            return
        if text.startswith("/") and self._handle_slash_command(text):
            input_widget.text = ""
            return
        view = self.query_one(ConversationView)
        view.append_user_message(text)
        input_widget.text = ""
        if self._ws is not None:
            await self.send_run_turn(text)

    def _handle_slash_command(self, text: str) -> bool:
        """Route a ``/``-prefixed command. Return True if consumed."""
        parts = text[1:].split(None, 1)
        if not parts or not parts[0]:
            return False
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd in ("sessions", "switch"):
            if cmd == "sessions":
                self.action_sessions()
                return True
            self._switch_session_command(arg)
            return True
        if cmd in ("session",):
            self._switch_session_command(arg)
            return True
        return False

    def _switch_session_command(self, arg: str) -> None:
        """``/session <id>`` — switch the active session by id or unique prefix.

        Matches ``arg`` against attached session ids: exact match first,
        then a unique-prefix match (so an 8-char short id works). The
        resolved id routes to ``on_session_selected``. The raw arg routes
        there ONLY when nothing is attached (the driven-app subclass
        spawn+attaches a NEW session for an unknown id).
        """
        if not arg:
            logger.info("tui: /session requires a session id")
            return
        session_ids = list(self._ws_by_session)
        target: str | None = arg if arg in session_ids else None
        if target is None:
            matches = [s for s in session_ids if s.startswith(arg)]
            if len(matches) == 1:
                target = matches[0]
            elif len(matches) > 1:
                logger.info(
                    "tui: /session %r matches %d attached sessions; "
                    "use more characters",
                    arg,
                    len(matches),
                )
                return
        if target is not None:
            self.on_session_selected(target)
            return
        if not session_ids:
            self.on_session_selected(arg)
            return
        logger.info("tui: /session %r matches no attached session", arg)

    # -----------------------------------------------------------------
    # Stream + tool routing from WS frames.
    # -----------------------------------------------------------------

    def append_assistant_delta(self, kind: str = "text", text: str = "") -> None:
        """Forward a WS ``assistant_delta`` to the conversation view."""
        self.query_one(ConversationView).append_delta(kind, text)

    def append_tool_call(
        self,
        name: str,
        status: str = "running",
        call_id: str | None = None,
    ) -> Any:
        """Forward a WS ``tool_call_started`` to the conversation view."""
        return self.query_one(ConversationView).append_tool_call(
            name,
            status,
            call_id=call_id,
        )

    # -----------------------------------------------------------------
    # Session index (``refresh_session_list``) — populates the transient
    # picker, never a permanent sidebar.
    # -----------------------------------------------------------------

    def refresh_session_list(self, db_path: Path) -> None:
        """Repopulate the session index from the session storage DB.

        Opens ``Storage`` transiently for the read (no fcntl lock on
        read-only access); closes the connection immediately. Sessions
        visible from ``Path.cwd()`` are listed; labels carry the session
        title + cwd + worktree branch when applicable. Failures log a
        warning and leave the existing index intact.
        """
        from cothis.session.storage import Storage

        self._db_path = db_path
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

        from cothis.git import find_worktree_for_path, list_worktrees

        worktrees = list_worktrees(Path.cwd())
        rows_sorted = sorted(
            rows,
            key=lambda r: (str(r.cwd) if r.cwd else "", r.updated_at),
            reverse=False,
        )
        self._session_rows = []
        for row in rows_sorted:
            label = row.title or f"session {row.id[:8]}"
            cwd_hint = str(row.cwd) if row.cwd else "(no cwd)"
            wt = find_worktree_for_path(Path(row.cwd), worktrees) if row.cwd else None
            if wt is not None and wt.branch is not None:
                cwd_hint = f"{cwd_hint} · branch:{wt.branch}"
            self._session_rows.append((row.id, f"{label}  ({cwd_hint})"))

    # -----------------------------------------------------------------
    # Session selection (#252 item 5).
    # -----------------------------------------------------------------

    def on_session_selected(self, session_id: str) -> None:
        """Hook called when the user picks a session (picker / slash command).

        Default behaviour: ``set_active_session(session_id)``. Callers that
        want to spawn a worker + attach WS on selection subclass
        ``CothisApp`` and override this method.
        """
        self.set_active_session(session_id)

    def set_active_session(self, session_id: str) -> None:
        """Mark ``session_id`` as the active session + fire the change hook."""
        previous = self._active_session_id
        self._active_session_id = session_id
        if previous != session_id:
            self.on_active_session_changed(session_id)

    def on_active_session_changed(self, session_id: str) -> None:
        """Hook fired when the active session changes (#230).

        Default: mirror ``session_id`` into ``footer_session`` so the
        (conditional) status ``session:`` cell reflects the active session
        immediately, AND — when a session DB is known — clear
        ``ConversationView`` + replay the target session's stored history.
        Focus returns to the composer input.
        """
        logger.info("tui: active session changed → %s", session_id)
        self.footer_session = session_id
        self._refresh_footer()
        if self._db_path is None:
            self._refocus_input()
            return
        try:
            view = self.query_one(ConversationView)
        except Exception:  # noqa: BLE001 — compose may not have run yet
            self._refocus_input()
            return
        view.clear()
        self.replay_session_history(session_id, self._db_path)
        self._refocus_input()

    # -----------------------------------------------------------------
    # WS attach (#252 item 1) — caller supplies URI + bearer token.
    # -----------------------------------------------------------------

    async def attach_ws(self, uri: str, token: str) -> None:
        """Open a WS client to a worker; pump inbound frames to the view.

        Caller decides how the worker got spawned — this method only needs
        the bind-handshake output (URI + bearer token). Idempotent.
        """
        import websockets

        await self.detach_ws()
        self._ws = await websockets.connect(
            uri,
            additional_headers={"Authorization": f"Bearer {token}"},
        )
        self._ws_pump_task = asyncio.create_task(self._pump_ws())

    async def detach_ws(self) -> None:
        """Close the WS client + cancel the pump task (if attached)."""
        task = self._ws_pump_task
        self._ws_pump_task = None
        ws = self._ws
        self._ws = None
        if self._ws_by_session.get(self._active_session_id or "") is None:
            self.run_state = "idle"
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        if ws is not None:
            await ws.close()

    # -----------------------------------------------------------------
    # Multi-session WS attach (#230).
    # -----------------------------------------------------------------

    def replay_session_history(self, session_id: str, db_path: Path) -> None:
        """Replay a session's stored history into ``ConversationView``.

        Reads the rebuilt messages via ``Session.peek_messages`` (the
        lock-free, read-only storage surface). Best-effort: a missing DB /
        corrupt schema / unknown id logs a warning + returns.
        """
        from cothis.session import Session

        try:
            messages = Session.peek_messages(db_path, session_id)
        except Exception as exc:  # noqa: BLE001 — best-effort replay
            logger.warning(
                "tui: cannot replay history for %s from %s: %s",
                session_id[:8],
                db_path,
                exc,
            )
            return
        view = self.query_one(ConversationView)
        for msg in messages:
            view.render_replayed_message(msg)

    async def attach_session_ws(
        self,
        session_id: str,
        uri: str,
        token: str,
        *,
        db_path: Path | None = None,
    ) -> None:
        """Open a WS client for a specific session (multi-session #230).

        Stores the connection in ``_ws_by_session`` keyed by session_id +
        starts a dedicated pump task. Marks the session as active via
        ``set_active_session``. Idempotent. When ``db_path`` is supplied,
        the session's stored history is replayed AFTER the WS connects but
        BEFORE the pump task starts.
        """
        import websockets

        await self.detach_session_ws(session_id)
        ws = await websockets.connect(
            uri,
            additional_headers={"Authorization": f"Bearer {token}"},
        )
        self._ws_by_session[session_id] = ws
        if db_path is not None:
            self._db_path = db_path
            self.replay_session_history(session_id, db_path)
        self._ws_pump_tasks_by_session[session_id] = asyncio.create_task(
            self._pump_ws_connection(ws)
        )
        self.set_active_session(session_id)
        self._refresh_footer()

    async def detach_session_ws(self, session_id: str) -> None:
        """Close + remove one session's WS connection (multi-session #230)."""
        task = self._ws_pump_tasks_by_session.pop(session_id, None)
        ws = self._ws_by_session.pop(session_id, None)
        if session_id == self._active_session_id:
            self.run_state = "idle"
        self._refresh_footer()
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        if ws is not None:
            await ws.close()

    async def send_run_turn(self, prompt: str) -> None:
        """Forward a prompt as a ``run_turn`` control message over WS."""
        ws = self._ws_by_session.get(self._active_session_id or "") or self._ws
        if ws is None:
            return
        await ws.send(json.dumps({"type": "run_turn", "prompt": prompt}))

    async def send_interrupt_turn(self) -> None:
        """Forward an ``interrupt_turn`` control message to the active worker."""
        ws = self._ws_by_session.get(self._active_session_id or "") or self._ws
        if ws is None:
            return
        with suppress(asyncio.CancelledError, Exception):
            await ws.send(json.dumps({"type": "interrupt_turn"}))

    async def action_interrupt_turn(self) -> None:
        """Esc-key action — interrupt the in-flight turn.

        Guarded on run-state: only interrupts when ``run_state == "running"``.
        Sets ``run_state="interrupted"`` optimistically; reconciled to
        ``"idle"`` when the worker's terminal ``turn_finished`` frame lands.
        """
        if self.run_state != "running":
            return
        ws = self._ws_by_session.get(self._active_session_id or "") or self._ws
        if ws is None:
            return
        self.run_state = "interrupted"
        await self.send_interrupt_turn()

    # -----------------------------------------------------------------
    # WS pump + dispatch.
    # -----------------------------------------------------------------

    async def _pump_ws(self) -> None:
        """Read inbound WS frames from ``self._ws`` (single-session path)."""
        if self._ws is None:
            return
        await self._pump_ws_connection(self._ws)

    async def _pump_ws_connection(self, ws: Any) -> None:
        """Read inbound WS frames from a specific connection + dispatch."""
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
                msg.get("kind", "text"),
                msg.get("text", ""),
            )
        elif typ == "tool_call_started":
            self.append_tool_call(
                msg.get("tool", "?"),
                call_id=msg.get("call_id"),
            )
        elif typ == "tool_call_result_pointer":
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
                call_id,
                is_error=bool(msg.get("is_error")),
            )
            if card is None:
                logger.debug(
                    "tui: tool_call_result_pointer for %s (call_id=%s) — "
                    "no matching card; dropping",
                    msg.get("tool"),
                    call_id,
                )
        elif typ == "ask_user_request":
            self.on_ask_user_request(
                ask_id=msg.get("ask_id", ""),
                prompt=msg.get("prompt", ""),
                choices=msg.get("choices", []),
            )
        elif typ == "turn_started":
            self.run_state = "running"
        elif typ == "turn_finished":
            self.footer_model = msg.get("model") or ""
            sid = msg.get("session_id") or ""
            self.footer_session = sid
            self.footer_pressure = msg.get("pressure") or ""
            self.footer_skills = list(msg.get("active_skills") or [])
            self.run_state = "idle"
        elif typ == "error":
            logger.warning("tui: worker error: %s", msg.get("message", ""))
        else:
            logger.debug("tui: ignoring unknown WS message type: %r", typ)

    def on_ask_user_request(
        self,
        *,
        ask_id: str,
        prompt: str,
        choices: list,
    ) -> None:
        """Mount ``AskUserModal``; route the user's pick to ``resolve_ask``."""

        def _on_dismiss(value: str | None) -> None:
            self._refocus_input()
            ws = self._ws_by_session.get(self._active_session_id or "") or self._ws
            if ws is None:
                logger.warning(
                    "tui: no active WS when resolving ask_id=%s; reply dropped",
                    ask_id,
                )
                return
            asyncio.create_task(
                ws.send(
                    json.dumps(
                        {
                            "type": "resolve_ask",
                            "ask_id": ask_id,
                            "value": value,
                        }
                    )
                )
            )

        self.push_screen(AskUserModal(prompt, choices), _on_dismiss)

    # -----------------------------------------------------------------
    # Status dock — one combined watcher repaints the footer.
    # -----------------------------------------------------------------

    def watch_run_state(self, _value: str) -> None:
        self._refresh_footer()
        self._refresh_status()

    def watch_footer_model(self, _value: str) -> None:
        self._refresh_footer()

    def watch_footer_session(self, _value: str) -> None:
        self._refresh_footer()

    def watch_footer_pressure(self, _value: str) -> None:
        self._refresh_footer()

    def watch_footer_skills(self, _value: list[str]) -> None:
        self._refresh_footer()

    def _render_footer_str(self) -> str:
        """Compose the one-line status dock from the current reactives.

        Cells: ``model | [session:<short-id> |] ctx:<pressure> |
        skills:[..] | state:<run_state>``. The ``session:<short-id>`` cell
        renders ONLY when more than one session is attached — in the common
        single-session case the id is redundant noise, so it is hidden.
        """
        pressure = self.footer_pressure or "?"
        skills = ",".join(self.footer_skills) if self.footer_skills else "-"
        model = self.footer_model or "-"
        cells = [model]
        if self._attached_session_count() > 1:
            cells.append(f"session:{self.footer_session[:8]}")
        cells.append(f"ctx:{pressure}")
        cells.append(f"skills:{skills}")
        cells.append(f"state:{self.run_state}")
        return " | ".join(cells)

    def _attached_session_count(self) -> int:
        """Number of sessions with a live WS the user can switch among."""
        return len(self._ws_by_session)

    def _refresh_status(self) -> None:
        """Update the composer status line (pi's statusContainer slot).

        Shows the working state while a turn is in flight (``>> running —
        Esc to interrupt``) and a brief interrupted marker; empty when
        idle so the dock stays calm.
        """
        if self.run_state == "running":
            text = "[b]>> running[/b] — Esc to interrupt"
        elif self.run_state == "interrupted":
            text = "[b]>> interrupted[/b] — awaiting turn end"
        else:
            text = ""
        with suppress(Exception):  # status line not yet mounted
            self.query_one("#status-line", Label).update(text)

    def _refresh_footer(self) -> None:
        """Re-render the status dock from the current reactives."""
        with suppress(Exception):  # footer not yet mounted
            self.query_one(CothisFooter).update(self._render_footer_str())


def run(app: CothisApp | None = None) -> None:
    """Entry point: ``python -m cothis.tui``.

    ``app`` lets a caller (e.g. the CLI ``tui`` command) pass a subclass of
    ``CothisApp`` with hooks overridden for production wiring. Default is a
    bare ``CothisApp`` — useful for development, tests, and scenarios where
    the TUI runs without a Supervisor.
    """
    if app is None:
        app = CothisApp()
    app.run()


if __name__ == "__main__":
    run()
