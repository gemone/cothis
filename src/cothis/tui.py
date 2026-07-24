"""``cothis.tui`` — Textual TUI core (#228).

3-pane layout for a single attached session (multi-session lands
with #230):

- ``SessionList`` (left): sessions from the session table.
- ``ConversationView`` (center): scrollable Markdown.
- ``InputBar`` (bottom): multiline input.

This is the **design-review slice** — the visual layout is the
deliverable. Real interactivity (WS attach, ``run_turn`` forward,
notify_events polling, historical session reload) lands in
follow-ups (see ADR-0019 §6).
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Header, Label, ListItem, ListView, Markdown, TextArea


class SessionList(ListView):
    """Left pane — sessions from the session table.

    Placeholder for the design-review slice; real session-table
    population + selection-handling lands in a follow-up.
    """

    DEFAULT_CSS = """
    SessionList {
        width: 24;
        dock: left;
        border: round $primary;
    }
    """


class ConversationView(VerticalScroll):
    """Center pane — scrollable Markdown.

    ``append_markdown`` is the API the notify-poller will call once
    real WS / notify_events wiring lands. For the design-review slice
    it just holds whatever was appended.
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
        self._renderable: str = ""

    @property
    def renderable_str(self) -> str:
        """Accumulated Markdown source — useful for tests + debugging."""
        return self._renderable

    def append_markdown(self, markdown: str) -> None:
        """Append Markdown to the conversation; render via ``Markdown`` widget."""
        self._renderable += markdown + "\n"
        widget = Markdown(self._renderable)
        # Replace any existing Markdown child so re-renders don't stack.
        existing = self.query("Markdown")
        for old in existing:
            old.remove()
        self.mount(widget)


class InputBar(Container):
    """Bottom pane — multiline input + send hotkey.

    The actual send → WS → ``run_turn`` path lands in a follow-up.
    """

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

    DEFAULT_CLASSES = "-hidden"

    def compose(self) -> ComposeResult:
        yield TextArea()

    def get_text(self) -> str:
        """Current input text."""
        return self.query_one(TextArea).text

    def set_text(self, text: str) -> None:
        """Replace the input text (used by tests + the send-and-clear flow)."""
        self.query_one(TextArea).text = text


class CothisApp(App):
    """Textual app shell — 3-pane layout, single session.

    Design-review slice (#228): visual layout is the deliverable; real
    WS attach + notify_events polling + session reload are deferred.
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


def run() -> None:
    """Entry point: ``python -m cothis.tui``."""
    app = CothisApp()
    app.run()


if __name__ == "__main__":
    run()
