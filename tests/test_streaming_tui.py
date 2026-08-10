"""Tests for ``cothis.streaming_tui`` — the rich+prompt_toolkit chat shell.

Covers the three contracts the module exists for:

- virtual rendering: the conversation control serves lines lazily, so a
  long transcript costs O(viewport) per frame, not O(content);
- streaming block semantics: per-delta re-renders replace only the live
  block, never finalized history;
- the ``/`` menu: the completer suggests slash commands only while the
  input starts with ``/``.
"""

from __future__ import annotations


def _line(text: str) -> list[tuple[str, str]]:
    # prompt_toolkit fragments are (style, text) tuples.
    return [("", text)]


def test_virtual_control_serves_lines_lazily() -> None:
    """``create_content`` hands the Window a lazy ``get_line``.

    The control stores all lines, but the ``UIContent`` only materializes
    the slice the Window actually paints — the O(viewport) per-frame
    contract. Out-of-range indices return an empty line (the Window asks
    past the end while the content grows).
    """
    from cothis.streaming_tui import ConversationControl

    ctl = ConversationControl()
    for i in range(100):
        ctl.append_text(f"line {i}")

    content = ctl.create_content(width=80, height=24)
    assert content.line_count == 100
    # Lazy: get_line(50) is served from storage, get_line(999) is a no-op.
    assert content.get_line(50) == _line("line 50")
    assert content.get_line(999) == []
    # The content grows without touching the already-created UIContent.
    ctl.append_text("line 100")
    assert ctl.line_count == 101


def test_stream_block_replaces_only_live_lines() -> None:
    """``update_stream`` re-renders the live block; history stays intact."""
    from cothis.streaming_tui import ConversationControl

    ctl = ConversationControl()
    ctl.append_text("user: hi")  # finalized history
    ctl.begin_stream()
    ctl.update_stream([_line("the "), _line("answer")])
    assert ctl.line_count == 3

    # A delta re-renders ONLY the stream block.
    ctl.update_stream([_line("the full answer")])
    assert ctl.line_count == 2
    assert ctl._lines[0] == _line("user: hi")
    assert ctl._lines[1] == _line("the full answer")

    # Tool-call boundary: finalize the block, start a fresh one.
    ctl.end_stream()
    ctl.begin_stream()
    ctl.update_stream([_line("next segment")])
    assert ctl.line_count == 3
    assert ctl._lines[1] == _line("the full answer")


def test_slash_completer_suggests_commands_only_for_slash_input() -> None:
    """``/``-prefixed input pops the menu; plain text yields nothing."""
    from prompt_toolkit.document import Document

    from cothis.streaming_tui import SlashCompleter

    comp = SlashCompleter()

    def names(document_text: str) -> list[str]:
        doc = Document(document_text)
        return [c.text for c in comp.get_completions(doc, None)]

    assert "/exit" in names("/ex")
    assert "/help" in names("/")
    assert names("hello") == []
    assert names("/nonexistent") == []


def test_fragments_are_style_text_tuples() -> None:
    """The control emits ``(style, text)`` tuples — prompt_toolkit's contract.

    A swapped ``(text, style)`` order makes the renderer parse the prompt
    text as a style string, crashing with ``Wrong color format '❯'``
    (the ``❯`` prompt char). Regression guard for that crash.
    """
    from cothis.streaming_tui import ConversationControl

    ctl = ConversationControl()
    ctl.append_text("\u276f hi", style="class:prompt")
    line = ctl.create_content(width=80, height=24).get_line(0)
    assert line == [("class:prompt", "\u276f hi")]

    # rich-rendered markdown also comes out as (style, text).
    from cothis.streaming_tui import _markdown_lines

    md_lines = _markdown_lines("**bold**", width=40)
    for frag in md_lines[0]:
        assert isinstance(frag, tuple) and len(frag) == 2
        assert not frag[1].startswith("class")  # text is the second element
