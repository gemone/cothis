"""Streaming chat TUI — rich + prompt_toolkit with a virtualized transcript.

Full-screen layout (no Textual):

::

    ┌──────────────────────────────────────────────┐
    │ scrollable conversation (virtualized)        │  ← only visible lines rendered
    │   … past turns + the live answer             │
    ├──────────────────────────────────────────────┤
    │ ❯ prompt_toolkit input                       │  ← Enter sends, / pops the slash menu
    └──────────────────────────────────────────────┘

Virtual rendering
    The conversation control stores pre-rendered rich lines; the Window
    requests only the visible slice per frame (``UIContent.get_line`` is
    lazy), so a long transcript costs O(viewport) per render — no full
    re-render, no widget-per-message DOM growth.

Continuous scroll / follow-end
    ``window.vertical_scroll`` is the preferred scroll (overrides the
    cursor-based computation). New content pins the view to the bottom
    only while the user is already there (``_following``); scrolling away
    (PgUp / wheel up) freezes the viewport so the stream never yanks the
    reader; PgDn / wheel down re-follows.

Slash menu
    Typing ``/`` pops a completion menu (``SlashCompleter``) of commands:
    ``/help``, ``/exit``, ``/clear``.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    HSplit,
    Layout,
    ScrollbarMargin,
    ScrollOffsets,
    UIContent,
    Window,
)
from prompt_toolkit.layout.controls import Point, UIControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from rich.console import Console, ConsoleOptions
from rich.markdown import Markdown

if TYPE_CHECKING:
    from prompt_toolkit.document import Document

# Rich → prompt_toolkit fragment conversion -------------------------------

_RICH_CONSOLE = Console()


def _rich_to_pt(renderable: Any, width: int) -> list[list[tuple[str, str]]]:
    """Render a rich renderable to per-line ``(style, text)`` fragments.

    prompt_toolkit formatted-text fragments are ``(style, text)`` tuples —
    the reverse of rich's Segment order. Swapping here means the Window
    paints the text with its style instead of parsing the text as a style
    string (which crashes on the ``❯`` prompt char). The Window only ever
    requests the visible slice of these lines (virtual rendering); the
    conversion itself is the only O(content) cost, and it is throttled for
    the streaming block.
    """
    options = _RICH_CONSOLE.options.update(width=width)
    seg_lines = _RICH_CONSOLE.render_lines(renderable, options)
    return [
        [(str(seg.style) if seg.style else "", seg.text) for seg in line]
        for line in seg_lines
    ]


def _markdown_lines(text: str, width: int) -> list[list[tuple[str, str]]]:
    return _rich_to_pt(Markdown(text), width)


# Slash menu ---------------------------------------------------------------

_SLASH_COMMANDS = {
    "help": "show this command list",
    "exit": "leave the session",
    "quit": "leave the session",
    "clear": "clear the transcript (history stays in the session)",
}


class SlashCompleter(Completer):
    """Complete ``/<command>`` while the input starts with ``/``."""

    def get_completions(
        self, document: Document, complete_event: Any
    ) -> list[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return []
        prefix = text[1:].lower()
        out: list[Completion] = []
        for name, desc in _SLASH_COMMANDS.items():
            if name.startswith(prefix):
                out.append(
                    Completion(
                        f"/{name}",
                        start_position=-len(text),
                        display=f"/{name}",
                        display_meta=desc,
                    )
                )
        return out


# Virtualized conversation control -----------------------------------------


class ConversationControl(UIControl):
    """Transcript control backed by a line list; renders the visible slice.

    ``create_content`` hands the Window a ``UIContent`` whose ``get_line``
    serves pre-rendered lines lazily — prompt_toolkit's renderer only asks
    for the rows it paints, so a long session stays O(viewport) per frame.
    """

    def __init__(self) -> None:
        self._lines: list[list[tuple[str, str]]] = []
        # Index where the live streaming block starts (re-rendered per
        # delta); lines before it are finalized history.
        self._stream_start: int | None = None

    @property
    def line_count(self) -> int:
        return len(self._lines)

    def append_fragments(self, fragments: list[tuple[str, str]]) -> None:
        """Append one logical line of formatted text."""
        self._lines.append(fragments)

    def append_text(self, text: str, style: str = "") -> None:
        self.append_fragments([(style, text)])

    def begin_stream(self) -> None:
        """Open a streaming block at the current end of the transcript."""
        self._stream_start = len(self._lines)

    def update_stream(self, fragments: list[list[tuple[str, str]]]) -> None:
        """Replace the streaming block with re-rendered fragments."""
        if self._stream_start is None:
            self.begin_stream()
        del self._lines[self._stream_start :]
        self._lines.extend(fragments)

    def end_stream(self) -> None:
        """Close the streaming block (finalized into history)."""
        self._stream_start = None

    def clear(self) -> None:
        self._lines.clear()
        self._stream_start = None

    def create_content(self, width: int, height: int) -> UIContent:
        lines = self._lines
        n = len(lines)

        def get_line(i: int) -> list[tuple[str, str]]:
            if 0 <= i < n:
                return lines[i]
            return []

        return UIContent(
            # ``get_line`` returns the narrower ``list[tuple[str, str]]``;
            # prompt_toolkit's renderer accepts it (the broader type also
            # allows mouse-event tuples we never emit).
            get_line=cast("Any", get_line),
            line_count=n,
            cursor_position=Point(max(0, n - 1), 0),
            show_cursor=False,
        )


# Streaming application -----------------------------------------------------


class StreamingChatApp:
    """prompt_toolkit full-screen chat: virtualized transcript + prompt."""

    def __init__(self, run_turn: Any) -> None:
        self._run_turn = run_turn  # async (prompt: str) -> None
        self.control = ConversationControl()
        self._following = True
        self._turn_task: asyncio.Task[Any] | None = None
        self._in_turn = False

        self._input = TextArea(
            multiline=False,
            prompt="❯ ",
            completer=SlashCompleter(),
            complete_while_typing=Condition(
                lambda: (self._input.text or "").startswith("/")
            ),
            accept_handler=self._on_accept,
            style="class:input",
        )

        self._window = Window(
            content=self.control,
            wrap_lines=False,
            right_margins=[ScrollbarMargin()],
            scroll_offsets=ScrollOffsets(top=0, bottom=0),
        )
        self._window.vertical_scroll = 10**9  # pin to bottom at launch

        kb = KeyBindings()
        self._bind_keys(kb)

        self._app = Application(
            layout=Layout(HSplit([self._window, self._input], padding=0)),
            key_bindings=kb,
            mouse_support=True,
            full_screen=True,
            style=Style(
                [
                    ("input", "ansiteal"),
                    ("prompt", "ansiteal bold"),
                    ("muted", "ansibrightblack"),
                ]
            ),
        )

    def _bind_keys(self, kb: KeyBindings) -> None:
        kb.add("pageup")(self._scroll(-1))
        kb.add("pagedown")(self._scroll(+1))
        kb.add("<scroll-up>")(self._scroll(-1))
        kb.add("<scroll-down>")(self._scroll(+1))
        kb.add("c-c")(self._on_ctrl_c)

    def _scroll(self, direction: int) -> Any:
        def handler(event: Any) -> None:
            info = self._window.render_info
            page = max(1, (info.window_height - 2) if info else 10)
            current = self._window.vertical_scroll or 0
            target = current + direction * page
            max_scroll = max(0, info.content_height - info.window_height) if info else 0
            target = max(0, min(target, max_scroll + 1))
            self._window.vertical_scroll = target
            self._following = bool(
                direction > 0 and (info is None or target >= max_scroll)
            )
            self._app.invalidate()

        return handler

    def _on_ctrl_c(self, event: Any) -> None:
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
            self.append_text("[interrupted]", style="class:muted")
            self._following = True
            self._app.invalidate()

    def _on_accept(self, buffer: Any) -> bool:
        text = (buffer.text or "").strip()
        buffer.text = ""
        if not text:
            return True
        if text.startswith("/"):
            self._handle_slash(text)
            return True
        self._start_turn(text)
        return True

    def _handle_slash(self, text: str) -> None:
        cmd = text[1:].split(None, 1)[0].lower()
        if cmd in ("exit", "quit"):
            self._app.exit()
        elif cmd == "help":
            for name, desc in _SLASH_COMMANDS.items():
                self.append_text(f"  /{name:<6} {desc}", style="class:muted")
        elif cmd == "clear":
            self.control.clear()
        else:
            self.append_text(
                f"unknown command: /{cmd} (try /help)", style="class:muted"
            )
        self._app.invalidate()

    def _start_turn(self, prompt: str) -> None:
        if self._in_turn:
            self.append_text(
                "[busy — wait for the turn to finish]", style="class:muted"
            )
            self._app.invalidate()
            return
        self._in_turn = True
        self.append_text("❯ " + prompt, style="class:prompt")
        self.control.begin_stream()
        self._turn_task = self._app.create_background_task(self._run_turn(prompt))

    # Helpers used by the turn consumer --------------------------------

    def append_text(self, text: str, style: str = "") -> None:
        self.control.append_text(text, style)
        self._pin_or_freeze()

    def append_rich(self, renderable: Any) -> None:
        width = self._width()
        for line in _rich_to_pt(renderable, width):
            self.control.append_fragments(line)
        self._pin_or_freeze()

    def stream_update(self, markdown_text: str) -> None:
        width = self._width()
        self.control.update_stream(_markdown_lines(markdown_text, width))
        self._pin_or_freeze()

    def stream_end(self) -> None:
        self.control.end_stream()

    def mark_turn_done(self) -> None:
        self._in_turn = False
        self._pin_or_freeze()

    def _width(self) -> int:
        with suppress(Exception):
            return max(20, self._app.output.get_size().columns)
        return 80

    def _pin_or_freeze(self) -> None:
        """Pin to the bottom while following; freeze the viewport otherwise."""
        if self._following:
            self._window.vertical_scroll = 10**9
        self._app.invalidate()

    async def run(self) -> None:
        await self._app.run_async()
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()


# Convenience entry ---------------------------------------------------------


async def run_streaming_chat(agent: Any) -> None:
    """Run the streaming chat loop against an ``Agent`` (in-process).

    ``agent.run_stream(prompt)`` drives each turn; events render into the
    virtualized transcript. Slash commands are handled by the app itself.
    """
    from cothis.agent import (
        ContentDelta,
        MaxIterationsError,
        ToolCallEvent,
        ToolResultEvent,
    )

    state = {"busy": False}
    app_ref: dict[str, StreamingChatApp] = {}

    async def run_turn(prompt: str) -> None:
        app = app_ref["app"]
        if state["busy"]:
            return
        state["busy"] = True
        accumulated = ""
        stream = agent.run_stream(prompt)
        try:
            async for event in stream:
                if isinstance(event, ToolCallEvent):
                    if accumulated:
                        app.stream_end()
                        app.append_text("")
                    app.append_text(
                        "calling "
                        + event.name
                        + "("
                        + ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
                        + ")",
                        style="class:muted",
                    )
                    accumulated = ""
                    app.control.begin_stream()
                elif isinstance(event, ToolResultEvent):
                    continue
                elif isinstance(event, ContentDelta):
                    if event.kind and event.kind != "text":
                        continue
                    accumulated += event.text
                    app.stream_update(accumulated)
        except MaxIterationsError as exc:
            app.stream_end()
            app.append_text(f"[red]Error:[/red] {exc}")
        except asyncio.CancelledError:
            app.stream_end()
            app.append_text("[interrupted]", style="class:muted")
        except Exception as exc:  # noqa: BLE001 — provider/config errors
            # Credentials, network, provider-config: render the error into
            # the transcript and return to the prompt instead of letting it
            # escape into the event loop ("Unhandled exception in event
            # loop" freezes the app). The user sees the message, fixes the
            # env, and retries — or /exit.
            app.stream_end()
            app.append_text(f"[red]Error:[/red] {exc}")
        finally:
            if accumulated:
                app.stream_end()
            app.mark_turn_done()
            state["busy"] = False

    app = StreamingChatApp(run_turn=run_turn)
    app_ref["app"] = app
    try:
        await app.run()
    finally:
        await agent.aclose()
