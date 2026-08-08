"""``cothis.tui`` — Textual TUI core (#228).

3-pane layout for a single attached session:

- ``SessionList`` (left): sessions from the session table.
- ``ConversationView`` (center): scrollable Markdown + tool-call cards.
- ``TextArea`` input (bottom, ``id="input"``): multiline input with Ctrl+Enter to send.
- ``CothisFooter`` (very bottom, ``id="footer"``): one-line status bar —
  model / session short-id / context pressure / active skills / run-state.

Stream routing per the design-review sign-off (#228, 2026-07-24):
``ContentDelta(kind="text")`` renders as normal assistant content;
``ContentDelta(kind="thinking")`` renders as a collapsed, dimmed
``Collapsible`` (expand to read the model's reasoning). Tool calls render
as inline cards with a status badge.

WS attach (``attach_ws`` / ``attach_session_ws``) + ``run_turn``
forwarding (``send_run_turn``) landed with #252/#319. Multi-session
dispatch + the worktree picker (#234) are wired up; ``on_worktree_pick``
is the spawn contract for production CLI wiring.

Esc-to-interrupt: ``Binding('escape','interrupt_turn')``
cancels the in-flight turn via the worker's ``interrupt_turn`` control
message (the same task-cancel primitive used for run_turn-supersede and
disconnect). The worker emits ``turn_started`` / ``turn_finished`` frames
that drive the footer's run-state cell + post-turn refresh; ``action_interrupt_turn``
is a no-op unless a turn is running. The TUI does not speak ACP — the WS
bridge is the minimal, correct interrupt path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
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
    from textual.timer import Timer

    from cothis.git import Worktree

logger = logging.getLogger(__name__)

_TOOL_STATUS_ICONS = {"running": ">>", "done": "OK", "failed": "XX"}

# Streaming-render throttle + finalisation (#407). Re-parsing Markdown on
# every text delta is O(S²) in the segment size (the parser runs ~1.3 µs/char
# and a 20 KB answer is ~4000 deltas → tens of seconds of parse work). While
# streaming, deltas accumulate into a plain ``Static`` (no Markdown parse)
# refreshed at most every ``_STREAM_REFRESH_S``; the segment is parsed into a
# ``Markdown`` widget ONCE, ``_STREAM_FINALIZE_S`` after the last delta (an
# idle-debounce proxy for turn-end — the worker emits no turn-end frame) or at
# a tool-call boundary. Net per-segment cost: O(S) appends + one O(S) parse.
_STREAM_REFRESH_S = 0.05
_STREAM_FINALIZE_S = 0.3


# ---------------------------------------------------------------------
# Skill selection persistence (#235)
# ---------------------------------------------------------------------


# Skill-selection persistence (save/load_skill_selection) lives in
# ``cothis.skills`` so the worker subprocess can import it without the
# Textual cost (#415). Imported locally where used below.


# ---------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------


class SessionList(ListView):
    """Left pane — sessions from the session table.

    Populated by ``CothisApp.refresh_session_list`` (driven from
    ``Storage.list_sessions_in_cwd_tree``). Each row's label carries
    the session's cwd + worktree branch when applicable (#234 AC #3);
    rows are sorted by cwd for visual grouping by worktree.
    Selection fires ``on_list_view_selected`` → ``on_session_selected``.
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
        # Thinking-segment accumulator. Kept separate from
        # ``_text_buf`` so ``renderable_str`` (the text-segment source, read
        # by tests + inspection) stays free of reasoning content. Finalised
        # into a collapsed, dimmed ``Collapsible`` so the model's reasoning is
        # available but doesn't clutter the conversation.
        self._thinking_buf: list[str] = []
        # Plain-text widget shown WHILE a segment streams (#407). Mounting it
        # avoids re-parsing Markdown on every delta; it is swapped for a
        # ``Markdown`` widget (one parse) at finalisation.
        self._stream_static: Static | None = None
        # Monotonic timestamp of the last plain-text refresh (throttle).
        self._last_stream_refresh: float = 0.0
        # Idle-finalise debounce timer (#407): rearmed per delta; fires
        # ``_STREAM_FINALIZE_S`` after the LAST delta to parse Markdown once.
        # The worker emits no turn-end frame, so idle is the turn-end proxy.
        self._finalize_timer: Timer | None = None
        # True once the current buffer has been parsed into a mounted Markdown
        # widget. Gates idempotent re-finalise (timer then a boundary) and
        # signals ``append_delta`` to start a fresh segment when text resumes.
        # The buffer is RETAINED across finalise so ``renderable_str`` (tests
        # + inspection) still reflects the last segment's text.
        self._finalized: bool = False
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

        ``kind="text"`` → accumulate (O(1)) + cheap plain-text refresh; the
        segment is parsed into Markdown ONCE at finalisation, not per delta
        (#407 — per-delta Markdown re-parse was O(S²) in segment size).
        ``kind="thinking"`` → accumulate into ``_thinking_buf`` (separate
        from the text buffer) and finalise into a collapsed, dimmed
        ``Collapsible`` so the model's reasoning is available without
        cluttering the conversation. A kind switch (thinking → text or vice
        versa) finalises the active segment first, so each kind renders as
        its own block in event order.
        """
        if kind == "text":
            # Close any streaming thinking segment so text is its own block.
            self._finalize_thinking()
            # If the previous segment already finalised (idle timer fired, or
            # a user/tool boundary), start a fresh segment below it — the old
            # Markdown widget stays mounted; the buffer + handles reset.
            if self._finalized:
                self._text_buf = []
                self._finalized = False
            self._text_buf.append(text)
            self._refresh_stream()
            self._arm_finalize()
        elif kind == "thinking":
            # Close any streaming text segment so thinking is its own block.
            self._finalize_segment()
            self._thinking_buf.append(text)
            self._arm_finalize()

    def append_user_message(self, text: str) -> None:
        """Render a user prompt with a distinct prefix.

        Finalises any streaming segment first (so the prompt is its own
        block), then renders the escaped prompt as Markdown. This is one
        call per user message — not per token — so it is not on the hot
        streaming path. User text is Markdown-escaped (brackets) so
        injected links or markup can't activate inside the widget.
        """
        safe = text.replace("[", "\\[").replace("]", "\\]")
        self._finalize_active()
        self._text_buf = []
        self._finalized = False
        self._text_buf.append(f"\n> **you**: {safe}\n\n")
        self._finalize_segment()

    def append_tool_call(
        self, name: str, status: str = "running", call_id: str | None = None,
    ) -> ToolCallCard:
        """Mount an inline tool-call card; return it for status updates.

        Finalises the active text segment (parses its Markdown once) and
        resets the buffer so the next text delta starts a fresh segment
        below this card. Without the reset, all text would accumulate in
        one segment and the card would render below all of it — violating
        the "tool calls render as inline cards" rule (#228 Rule 3).

        ``call_id`` indexes the card in ``_cards_by_call_id`` so a
        subsequent ``tool_call_result_pointer`` frame can find it (#252
        item 4). ``None`` keeps the legacy un-indexed behaviour (no
        status update will land for this card).
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

    def render_replayed_message(self, msg: dict) -> None:
        """Render one rebuilt ``{role, content: [blocks]}`` message.

        Replay-on-attach reuses the existing primitives — there is no
        parallel renderer. A user text block routes through
        ``append_user_message`` (multiple text blocks in one message are
        concatenated into one call; ``tool_result`` blocks are skipped
        because the matching ``tool_use`` card already renders on the
        assistant side). An assistant text block mounts one Markdown
        widget via ``_finalize_segment`` (the buffer is seeded directly
        so the streaming-throttle path is bypassed). A ``tool_use`` block
        mounts a ``done``-status card (historical calls are already
        finished). ``thinking`` / ``image`` blocks are silently skipped
        (deferred).

        Leaves the view state clean (``_finalized=True``,
        ``_stream_static=None``) after each assistant text block, so the
        next live ``append_delta`` starts a fresh segment — the existing
        streaming path stays behaviour-identical.
        """
        role = msg.get("role")
        blocks = msg.get("content", []) or []
        if role == "user":
            # Concatenate text blocks into one user-echo call; skip
            # tool_result (the tool_use card already shows on the
            # assistant side).
            text_parts = [
                b.get("text", "")
                for b in blocks
                if b.get("type") == "text"
            ]
            if text_parts:
                self.append_user_message("\n".join(text_parts))
            return
        if role == "assistant":
            for b in blocks:
                btype = b.get("type")
                if btype == "text" and b.get("text"):
                    # Seed the buffer + mark unfinalised so ``_finalize_segment``
                    # mounts one Markdown widget for this block (bypassing the
                    # streaming Static + throttle path). ``_finalize_segment``
                    # removes any mounted Static itself (idempotent), so we do
                    # NOT pre-null ``_stream_static`` — that would skip its
                    # cleanup and orphan a mounted Static.
                    self._text_buf = [b["text"]]
                    self._finalized = False
                    self._finalize_segment()
                elif btype == "tool_use":
                    self.append_tool_call(
                        b.get("name", "?"),
                        status="done",
                        call_id=b.get("id"),
                    )
                # thinking / image / tool_result: deferred.

    def _refresh_stream(self) -> None:
        """Mount/refresh the plain-text streaming widget, throttled.

        No Markdown parse here — that is the #407 win. The first delta of a
        segment mounts a ``Static``; later deltas update it at most every
        ``_STREAM_REFRESH_S`` (cheap text layout, no parser), so the per-delta
        cost stays O(1) amortised rather than O(S) per call.
        """
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
            _STREAM_FINALIZE_S, self._finalize_active,
        )

    def _finalize_active(self) -> None:
        """Flush whichever segment(s) are streaming — text and/or thinking.

        The idle-finalise timer's callback, and the boundary flush called by
        ``append_tool_call`` / ``append_user_message``. Order matters only in
        that text mounts before thinking when both are pending (the model
        streams text then a trailing thinking block rarely); DOM order is
        otherwise driven by the kind-switch finalisation in ``append_delta``.
        """
        self._finalize_segment()
        self._finalize_thinking()

    def _finalize_thinking(self) -> None:
        """Mount the accumulated thinking as a collapsed, dimmed ``Collapsible``.

        Idempotent: a no-op when ``_thinking_buf`` is empty. The buffer is
        cleared on mount so a subsequent thinking segment starts fresh.
        Collapsed by default (``Collapsible`` ctor) so reasoning stays out of
        the way until the user expands it — the "dimmed/collapsed, toggle to
        expand" contract documented on ``ContentDelta``.
        """
        if not self._thinking_buf:
            return
        source = "".join(self._thinking_buf)
        self._thinking_buf = []
        at_bottom = self._at_bottom()
        self.mount(
            Collapsible(
                Markdown(source), title="reasoning", classes="thinking-block",
            )
        )
        self._follow(at_bottom)

    def _finalize_segment(self) -> None:
        """Swap the streaming ``Static`` for a ``Markdown`` widget (one parse).

        Called by the idle-finalise timer (turn-end proxy) and by the segment
        boundaries (``append_tool_call`` / ``append_user_message``). Idempotent:
        a no-op when the buffer is empty or ``_finalized`` is already set. The
        buffer is retained, so ``renderable_str`` still reflects the segment.
        """
        if self._finalize_timer is not None:
            self._finalize_timer.stop()
            self._finalize_timer = None
        # Idempotent: nothing to parse, or this segment already parsed.
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
        """True when the view is within a line of the bottom.

        The "user is watching the stream" state. Captured BEFORE a content
        change so a user who scrolled up to read earlier output isn't yanked
        back to the bottom on the next delta (#409).
        """
        return self.scroll_y >= self.max_scroll_y - 1

    def _follow(self, was_at_bottom: bool) -> None:
        """Re-pin to the bottom iff the user was already there.

        Synchronous (``immediate=True``): Textual updates the container's
        virtual size during ``mount`` / ``Static.update``, so the new
        ``max_scroll_y`` is current when ``scroll_end`` reads it — no need to
        defer past a refresh, which would race a user's manual scroll-up.
        """
        if was_at_bottom:
            self.scroll_end(animate=False, immediate=True)


class ConfigMenuModal(ModalScreen[set[str] | None]):
    """Config menu modal — toggleable skill entries (#235).

    Each skill is a ``Button`` that toggles selected/unselected on click.
    ``Done`` dismisses with the selected set; ``Esc`` dismisses with
    ``None`` (cancel). The selection persists across sessions via
    ``save/load_skill_selection`` (#415).
    """

    DEFAULT_CSS = """
    ConfigMenuModal {
        align: center middle;
    }
    ConfigMenuModal > Label {
        padding: 0 2;
    }
    ConfigMenuModal > Button.skill-toggle.-active {
        background: $accent;
    }
    """

    BINDINGS = [("escape", "dismiss_modal", "Cancel")]

    def __init__(self, skills: list[str], *, selected: set[str] | None = None) -> None:
        self._skills = skills
        # Seed from the persisted selection (caller passes it) so the menu
        # reflects what was saved last time (#415).
        self._selected: set[str] = set(selected) if selected else set()
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Label("Configurable Skills", id="menu-title")
        if not self._skills:
            yield Label("(no skills configured)", id="menu-empty")
        for name in self._skills:
            classes = "skill-toggle -active" if name in self._selected else "skill-toggle"
            yield Button(name, id=f"skill-{name}", classes=classes)
        yield Button("Done", id="menu-done")

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "menu-done":
            self.dismiss(self._selected)
        elif bid.startswith("skill-"):
            skill = bid[len("skill-"):]
            if skill in self._selected:
                self._selected.discard(skill)
                event.button.remove_class("-active")
            else:
                self._selected.add(skill)
                event.button.add_class("-active")


class AskUserModal(ModalScreen[str | None]):
    """Modal for interactive tool questions (#229).

    Shows ``prompt`` + one ``Button`` per choice. On click: dismiss
    with the chosen value. Esc or Cancel: dismiss with ``None``
    (the caller treats ``None`` as "user declined").
    """

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
        self._choices = list(choices)
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Label(self._prompt, id="ask-prompt")
        for choice in self._choices:
            yield Button(choice, id=f"choice-{choice}")
        yield Button("Cancel", id="ask-cancel")

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ask-cancel":
            self.dismiss(None)
        elif event.button.id and event.button.id.startswith("choice-"):
            self.dismiss(event.button.id[len("choice-"):])


class WorktreePickerModal(ModalScreen[str | None]):
    """Modal for choosing a git worktree for a new session (#234).

    Shows one ``Button`` per worktree (label = branch name when on a
    branch, else the path basename). On click: dismiss with the
    worktree's ``path`` as a string — that's what the caller stuffs
    into the new session's ``cwd``. Esc or Cancel: dismiss with
    ``None`` (the caller treats ``None`` as "user cancelled, no new
    session").
    """

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
            # Empty-list UX: the default "Pick a worktree" label would
            # mislead — there's nothing to pick. The current-directory
            # button is the only useful option here (besides Cancel).
            yield Label(
                "No worktrees found. Pick the current directory below, "
                "or run `git worktree add <path>` outside cothis, then retry.",
                id="worktree-prompt",
            )
        else:
            yield Label("Pick a worktree for the new session", id="worktree-prompt")
            # Index-based IDs: paths contain ``/`` which Textual IDs reject.
            # The button label is branch name (preferred) or path basename
            # for detached HEAD — branch is what the user thinks in terms of.
            for i, wt in enumerate(self._worktrees):
                label = wt.branch or wt.path.name
                yield Button(label, id=f"wt-{i}")
        # Always-present fallback: a session scoped to the current cwd
        # (the directory the TUI was launched from). Gives users a path
        # forward in a non-git cwd, and a quick "just use here" option
        # even when worktrees are available.
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


class CothisFooter(Static):
    """One-line status bar — surfaces run-state + key signals at a glance.

    Renders five cells left-to-right:

    ``model | session:<short-id> | ctx:<pressure> | skills:[a,b] | state:<run_state>``

    * ``<short-id>`` — first 8 chars of the active session id.
    * ``<pressure>`` — the ``PressureLevel`` value string (``none`` / ``low`` /
      ``medium`` / ``high`` / ``critical``) or ``?`` when unknown.
    * ``skills`` — comma-joined sorted active-skills set, or ``-`` when empty.
    * ``run_state`` — ``idle`` / ``running`` / ``interrupted``.

    The widget itself holds no state; it is repainted by ``CothisApp``'s
    combined ``_refresh_footer`` watcher whenever one of the footer
    reactives flips. No polling, no per-second timer.
    """

    DEFAULT_CSS = """
    CothisFooter#footer {
        height: 1;
        dock: bottom;
        background: $boost;
        color: $text-disabled;
        padding: 0 1;
    }
    """


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
    TextArea#input {
        height: 3;
        dock: bottom;
        border: round $secondary;
    }
    """

    BINDINGS = [
        Binding("ctrl+enter", "send_prompt", "Send", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("n", "new_session", "New session", show=True),
        Binding("ctrl+m", "menu", "Menu", show=True),
        # Esc → interrupt a running turn. No ``priority=True``: an
        # app-level priority binding would steal Esc from pushed modal
        # screens (modals install their own non-priority Esc binding to
        # dismiss). Textual resolves bindings top-screen-first for non-
        # priority bindings, so a modal's Esc wins while it's open and the
        # app binding fires only when no modal is pushed. Esc is not a text
        # character, so a focused TextArea does not consume it — the
        # binding routes to the app the same way ctrl+enter does.
        Binding("escape", "interrupt_turn", "Interrupt", show=False),
    ]

    # -----------------------------------------------------------------
    # Run-state + footer reactives — the app's first reactives.
    # Plain attrs would work, but reactives let a single combined watcher
    # (``_refresh_footer``) re-render the footer widget on any change with
    # no ad-hoc call sites. ``run_state`` is a constrained literal set
    # (idle|running|interrupted) so the Esc guard ``run_state != "running"``
    # stays narrow + ty-friendly.
    # -----------------------------------------------------------------
    run_state: reactive[str] = reactive("idle")
    footer_model: reactive[str] = reactive("")
    footer_session: reactive[str] = reactive("")
    footer_pressure: reactive[str] = reactive("")
    footer_skills: reactive[list[str]] = reactive[list[str]](list)

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
        """Mount ``WorktreePickerModal``; route the chosen path to ``on_worktree_pick`` (#234).

        Hook fired by ``action_new_session`` (the ``n`` keypress). The
        picker shows one ``Button`` per worktree; on dismiss the chosen
        path (or ``None`` for Esc / Cancel) is forwarded to
        ``on_worktree_pick`` — the single entry point for "create a
        session bound to this cwd".

        Subclasses can also override ``on_new_session`` itself to
        capture the worktree list without mounting the modal (existing
        tests do this).
        """
        logger.info(
            "tui: new-session action fired; %d worktree(s) visible",
            len(worktrees),
        )

        def _on_dismiss(value: str | None) -> None:
            if value is None:
                logger.info("tui: new-session cancelled (no worktree picked)")
                return
            self.on_worktree_pick(value)

        self.push_screen(WorktreePickerModal(worktrees), _on_dismiss)

    def on_worktree_pick(self, path: str) -> None:
        """Hook fired when the user picks a worktree for a new session (#234).

        Default: log the choice. The CLI / caller overrides this to
        call ``Supervisor.spawn_worker`` + ``SessionStorage.new`` +
        ``attach_session_ws`` with the picked path as the session cwd.

        Kept as a separate hook so the TUI doesn't need to know about
        the Supervisor/SessionStorage APIs — same inversion as
        ``attach_ws`` (caller decides how the worker was spawned).
        Tests / headless runs can also override to capture the path
        without spawning.
        """
        logger.info(
            "tui: worktree picked for new session: %s "
            "(spawn + attach wiring lands in the CLI integration)",
            path,
        )

    # -----------------------------------------------------------------
    # Menu binding (#235) — Ctrl-M opens the config menu.
    # The modal listing skills / MCP / LSP servers lands in the config menu;
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
        that lists skills via ``discover_tools``, MCP servers
        via ``MCPServer``, and any LSP servers. Selecting entries
        re-runs ``discover_tools`` with the chosen layers.
        """
        logger.info("tui: menu action fired (Ctrl-M)")
        from cothis.skills import load_skill_selection, save_skill_selection

        skills = self.list_configurable_skills()
        saved = load_skill_selection()

        def _on_config_done(selected: set[str] | None) -> None:
            if selected is None:  # Esc / Cancel
                return
            save_skill_selection(selected)

        # Seed the menu with the saved-and-still-available skills so it
        # reflects the last choice; persist the new selection on Done (#415).
        self.push_screen(
            ConfigMenuModal(skills, selected=saved & set(skills)),
            _on_config_done,
        )

    def list_configurable_skills(self) -> list[str]:
        """Return the names of skills discoverable from the current cwd.

        Wraps ``cothis.skills.discover_skills`` so the menu modal
        (not yet implemented) can display the list without
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
    # Multi-session WS connections (#230). Keyed by session_id;
    # each entry has its own pump task. The single-session ``_ws`` /
    # ``_ws_pump_task`` above stay for backward compat (``attach_ws``).
    _ws_by_session: dict[str, Any] = {}
    _ws_pump_tasks_by_session: dict[str, asyncio.Task[None]] = {}
    # Active session id (#230) — the session the user is currently
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
        yield TextArea(id="input")
        # Footer: docked at the very bottom, beneath the input. Both
        # widgets use ``dock: bottom``; Textual stacks docked siblings in DOM
        # order with the LAST mounted closest to the screen edge, so mounting
        # the footer after the input places it below the input.
        yield CothisFooter("", id="footer")

    async def on_mount(self) -> None:
        """Focus the session list on launch — preserve the pre-#375 target.

        Removing the ``InputBar(Container)`` wrapper (#375) lets the bare
        ``TextArea`` grab initial focus, which would shadow the bare ``n``
        "new session" shortcut — a focused ``TextArea`` consumes printable
        keys as text. Re-focusing ``SessionList`` keeps that shortcut (and
        every existing test) working. The input is still Tab-reachable,
        exactly the issue's scenario ("focuses the input bar, and types");
        the fix is that typing now inserts characters instead of being
        dropped by the old wrapper.
        """
        self.query_one(SessionList).focus()
        # Seed the footer with the initial idle render so the status bar
        # shows the documented cells (``state:idle`` etc.) before any WS
        # frame arrives. The watcher paths refresh it thereafter.
        self._refresh_footer()

    # -----------------------------------------------------------------
    # Footer reactives → re-render. One ``watch_*`` per reactive
    # delegates to a single combined callback so a turn_finished payload
    # (which updates all four data cells at once) re-paints the footer
    # once per changed field rather than four times.
    # -----------------------------------------------------------------

    def watch_run_state(self, _value: str) -> None:
        self._refresh_footer()

    def watch_footer_model(self, _value: str) -> None:
        self._refresh_footer()

    def watch_footer_session(self, _value: str) -> None:
        self._refresh_footer()

    def watch_footer_pressure(self, _value: str) -> None:
        self._refresh_footer()

    def watch_footer_skills(self, _value: list[str]) -> None:
        self._refresh_footer()

    def _render_footer_str(self) -> str:
        """Compose the one-line footer render from the current reactives.

        Cells: ``model | session:<short-id> | ctx:<pressure> | skills:[..] |
        state:<run_state>``. ``<short-id>`` is the first 8 chars of
        ``footer_session`` (the full id is stored; only the render is
        shortened). ``<pressure>`` falls back to ``?`` when unknown (the
        TUI never carries the raw ``None`` to the user-facing string).
        """
        short_sid = self.footer_session[:8]
        pressure = self.footer_pressure or "?"
        skills = ",".join(self.footer_skills) if self.footer_skills else "-"
        model = self.footer_model or "-"
        return (
            f"{model} | session:{short_sid} | "
            f"ctx:{pressure} | skills:{skills} | state:{self.run_state}"
        )

    def _refresh_footer(self) -> None:
        """Re-render the footer widget from the current reactives.

        Safe to call before ``compose`` finishes (the watcher fires during
        reactive init); the ``try``/``except NoMatches`` guards the
        not-yet-mounted case the same way ``on_active_session_changed``
        does. ``Static.update`` is the cheap text-relayout path — no
        Markdown parse, no DOM remount.
        """
        try:
            self.query_one(CothisFooter).update(self._render_footer_str())
        except Exception:  # noqa: BLE001 — footer not yet mounted
            pass

    async def action_send_prompt(self) -> None:
        """Read input text → render locally → forward to worker if attached.

        Local echo always runs (the user expects to see their prompt
        immediately). When a WS is attached (#252 item 1), the prompt
        is also forwarded as a ``run_turn`` control message — the
        worker drives the assistant-side rendering via subsequent
        ``assistant_delta`` frames pumped by ``_pump_ws``.

        Textual actions can be async; the framework awaits coroutine
        results, so ``await self.send_run_turn(text)`` blocks the
        action until the frame is on the wire (typically <1 ms).
        """
        input_widget = self.query_one("#input", TextArea)
        text = input_widget.text.strip()
        if not text:
            return
        view = self.query_one(ConversationView)
        view.append_user_message(text)
        input_widget.text = ""
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
        # Group sessions by worktree cwd (#234 AC #5). Stable sort by cwd
        # so sessions in the same worktree land adjacent — visual grouping
        # rather than chronological scatter. Updated_at desc stays as the
        # tiebreaker inside a group, preserving the "recent first" feel
        # within one worktree's sessions.
        rows_sorted = sorted(
            rows,
            key=lambda r: (str(r.cwd) if r.cwd else "", r.updated_at),
            reverse=False,
        )
        for row in rows_sorted:
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
    # Active-session tracking (#230)
    # -----------------------------------------------------------------

    def set_active_session(self, session_id: str) -> None:
        """Mark ``session_id`` as the active session + fire the change hook.

        Called by ``on_session_selected`` and by callers that spawn a
        new session (``on_new_session`` override → spawn → ``set_active``).
        Future slices use this to route ``send_run_turn`` to the right
        WS connection (#230) + highlight the focused entry.
        """
        previous = self._active_session_id
        self._active_session_id = session_id
        if previous != session_id:
            self.on_active_session_changed(session_id)

    def on_active_session_changed(self, session_id: str) -> None:
        """Hook fired when the active session changes (#230).

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
        # A dropped worker mid-turn leaves run_state stale ("running"); if
        # the active session has no other reachable WS, return to idle so
        # the footer + the Esc guard don't reference a dead connection.
        if self._ws_by_session.get(self._active_session_id or "") is None:
            self.run_state = "idle"
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if ws is not None:
            await ws.close()

    # -----------------------------------------------------------------
    # Multi-session WS attach (#230)
    # -----------------------------------------------------------------

    def replay_session_history(self, session_id: str, db_path: Path) -> None:
        """Replay a session's stored history into ``ConversationView``.

        Reads the rebuilt messages via ``Session.peek_messages`` — the
        lock-free, read-only storage surface already used by
        ``cothis history <id>``. Lock-free is essential here: the worker
        subprocess holds the cross-process file lock on the session, so
        a locking read from the TUI would contend with it. Each rebuilt
        ``{role, content: [blocks]}`` message routes through
        ``ConversationView.render_replayed_message``, which reuses the
        existing rendering primitives (no parallel renderer).

        Best-effort: a missing DB / corrupt schema / unknown id logs a
        warning + returns so the TUI stays usable — mirrors
        ``refresh_session_list``'s failure contract. A missing/empty
        session (``peek_messages`` returns ``[]``) renders nothing, so a
        fresh session's view stays correctly blank.
        """
        from cothis.session import Session

        try:
            messages = Session.peek_messages(db_path, session_id)
        except Exception as exc:  # noqa: BLE001 — best-effort replay
            logger.warning(
                "tui: cannot replay history for %s from %s: %s",
                session_id[:8], db_path, exc,
            )
            return
        view = self.query_one(ConversationView)
        for msg in messages:
            view.render_replayed_message(msg)

    async def attach_session_ws(
        self, session_id: str, uri: str, token: str,
        *,
        db_path: Path | None = None,
    ) -> None:
        """Open a WS client for a specific session (multi-session #230).

        Stores the connection in ``_ws_by_session`` keyed by
        ``session_id`` + starts a dedicated pump task. Marks the
        session as active via ``set_active_session``. Idempotent:
        re-attaching replaces the previous connection for that session.

        Replay-on-attach: when ``db_path`` is supplied, the
        session's stored history is replayed into ``ConversationView``
        AFTER the WS connects + is stored but BEFORE the pump task
        starts — so the rendered history is visible immediately on
        attach and a fast worker frame can't race the history render.
        Defaults ``None`` (no replay) so every existing 3-positional-arg
        caller (tests, crash-restart re-attach) stays behaviour-identical.
        """
        import websockets

        await self.detach_session_ws(session_id)
        ws = await websockets.connect(
            uri, additional_headers={"Authorization": f"Bearer {token}"},
        )
        self._ws_by_session[session_id] = ws
        if db_path is not None:
            self.replay_session_history(session_id, db_path)
        self._ws_pump_tasks_by_session[session_id] = asyncio.create_task(
            self._pump_ws_connection(ws)
        )
        self.set_active_session(session_id)

    async def detach_session_ws(self, session_id: str) -> None:
        """Close + remove one session's WS connection (multi-session #230)."""
        task = self._ws_pump_tasks_by_session.pop(session_id, None)
        ws = self._ws_by_session.pop(session_id, None)
        # If the detached session was active, its worker is going away —
        # clear a stale "running" so the footer + Esc guard don't reference
        # a dead connection.
        if session_id == self._active_session_id:
            self.run_state = "idle"
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

    async def send_interrupt_turn(self) -> None:
        """Forward an ``interrupt_turn`` control message to the active worker.

        Mirrors ``send_run_turn``'s WS routing (active-session WS with a
        single-session ``_ws`` fallback). No-op when no WS is attached —
        ``action_interrupt_turn`` already guards on run-state AND active-WS
        presence before calling this, but the no-op keeps the helper safe
        to call directly from tests / subclasses.
        """
        ws = self._ws_by_session.get(self._active_session_id or "") or self._ws
        if ws is None:
            return
        # Guard the send: the WS may be mid-close when Esc is pressed (a
        # worker crash mid-turn is exactly when the user reaches for the
        # escape hatch). Swallow the send error so run_state — already
        # optimistically "interrupted" — stays "interrupted" until the next
        # turn / re-attach, mirroring worker._emit_turn_finished's guard.
        try:
            await ws.send(json.dumps({"type": "interrupt_turn"}))
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    async def action_interrupt_turn(self) -> None:
        """Esc-key action — interrupt the in-flight turn.

        Guarded on run-state: only interrupts when ``run_state == "running"``.
        When idle (or already interrupted) Esc is a harmless no-op — it sends
        nothing and leaves state untouched. Also no-ops when no WS is
        attached for the active session (cannot reach the worker).

        Sets ``run_state="interrupted"`` optimistically so the footer
        reflects the in-flight cancel AND a second Esc press is a no-op
        (``interrupted != "running"``). Reconciled to ``"idle"`` when the
        worker's terminal ``turn_finished`` frame lands.
        """
        if self.run_state != "running":
            return
        ws = self._ws_by_session.get(self._active_session_id or "") or self._ws
        if ws is None:
            return
        self.run_state = "interrupted"
        await self.send_interrupt_turn()

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
            # #229: forward to the overridable hook. Default
            # auto-rejects (sends resolve_ask with value=None) so the
            # worker doesn't block in tests; subclasses mount a modal
            # (handled by the CLI integration).
            self.on_ask_user_request(
                ask_id=msg.get("ask_id", ""),
                prompt=msg.get("prompt", ""),
                choices=msg.get("choices", []),
            )
        elif typ == "turn_started":
            # Worker opened a turn — flip run-state so the footer's
            # state cell reads "running" and Esc becomes an armed interrupt.
            self.run_state = "running"
        elif typ == "turn_finished":
            # Terminal frame on every turn exit path (normal end,
            # timeout, error, interrupt). This is the authoritative refresh
            # — it carries the post-turn context pressure + any skill
            # load/deactivate that happened during the turn. Reconciles
            # run_state to "idle" (an optimistic "interrupted" set by
            # ``action_interrupt_turn`` lands here too).
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
        self, *, ask_id: str, prompt: str, choices: list,
    ) -> None:
        """Mount ``AskUserModal``; route the user's pick to ``resolve_ask``.

        Hook fired when the worker emits an ``ask_user_request`` (#229).
        The modal shows ``prompt`` + one button per choice; on dismiss
        the chosen value (or ``None`` for Esc / Cancel) is sent back over
        the active session's WS as a ``resolve_ask`` control message —
        which the worker forwards to ``Agent.resolve_ask``, unblocking
        the tool that called ``_ask_user``.

        Replies target the active session's WS (``_ws_by_session`` with
        a ``_ws`` fallback for the single-session case). If the WS has
        been detached by the time the user picks, the reply is dropped
        — the agent's Future will simply not resolve and the turn will
        hit the worker's ``_TURN_TIMEOUT_S``.
        """
        logger.info("tui: ask_user_request %s: %s", ask_id, prompt)

        def _on_dismiss(value: str | None) -> None:
            ws = (
                self._ws_by_session.get(self._active_session_id or "")
                or self._ws
            )
            if ws is None:
                logger.warning(
                    "tui: no active WS when resolving ask_id=%s; "
                    "reply dropped",
                    ask_id,
                )
                return
            asyncio.create_task(ws.send(json.dumps({
                "type": "resolve_ask", "ask_id": ask_id, "value": value,
            })))

        self.push_screen(AskUserModal(prompt, choices), _on_dismiss)


def run(app: CothisApp | None = None) -> None:
    """Entry point: ``python -m cothis.tui``.

    ``app`` lets a caller (e.g. the CLI ``tui`` command) pass a
    subclass of ``CothisApp`` with hooks overridden for production
    wiring (Supervisor-backed spawn, real session routing). Default
    is a bare ``CothisApp`` — useful for development, tests, and
    scenarios where the TUI runs without a Supervisor.
    """
    if app is None:
        app = CothisApp()
    app.run()


if __name__ == "__main__":
    run()
