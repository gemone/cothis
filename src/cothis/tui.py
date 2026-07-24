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
    """Inline card for one tool dispatch — name + status badge."""

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
    ``assistant_delta`` message. ``append_tool_call`` renders inline
    cards for ``tool_call_started`` events.
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
        self._text_buf: str = ""

    @property
    def renderable_str(self) -> str:
        """Accumulated text-delta source — for tests + debugging."""
        return self._text_buf

    def append_delta(self, kind: str, text: str) -> None:
        """Route a ContentDelta to the right rendering path.

        ``kind="text"`` → accumulate + re-render Markdown.
        ``kind="thinking"`` → logged but not rendered (collapsible block
        lands when the toggle UX is designed).
        """
        if kind == "text":
            self._text_buf += text
            self._refresh_markdown()
        elif kind == "thinking":
            logger.debug("dropping thinking delta (%d chars)", len(text))

    def append_user_message(self, text: str) -> None:
        """Render a user prompt with a distinct prefix."""
        self._text_buf += f"\n> **you**: {text}\n\n"
        self._refresh_markdown()

    def append_tool_call(self, name: str, status: str = "running") -> ToolCallCard:
        """Mount an inline tool-call card; return it for status updates."""
        self._refresh_markdown()
        card = ToolCallCard(name=name, status=status)
        self.mount(card)
        return card

    def _refresh_markdown(self) -> None:
        """Re-render the accumulated Markdown widget."""
        from textual.css.query import NoMatches

        try:
            self.query_one(Markdown).update(self._text_buf)
        except NoMatches:
            self.mount(Markdown(self._text_buf))


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

    def append_assistant_delta(self, kind: str, text: str) -> None:
        """Forward a WS ``assistant_delta`` to the conversation view."""
        self.query_one(ConversationView).append_delta(kind, text)

    def append_tool_call(self, name: str, status: str = "running") -> Any:
        """Forward a WS ``tool_call_started`` to the conversation view."""
        return self.query_one(ConversationView).append_tool_call(name, status)


def run() -> None:
    """Entry point: ``python -m cothis.tui``."""
    app = CothisApp()
    app.run()


if __name__ == "__main__":
    run()
