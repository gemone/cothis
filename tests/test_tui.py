"""Tests for ``cothis.tui`` (#228).

Verifies the 3-pane Textual layout renders via the Pilot harness:
- SessionList (left)
- ConversationView (center, scrollable)
- InputBar (bottom, multiline)

This is the **design-review slice** — the panes exist + have their
roles; real WS / notify_events polling / session reload land in
follow-ups (see ADR-0019 §6).
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_app_launches_with_three_panes() -> None:
    """Pilot launches CothisApp; all three panes are queryable."""
    from cothis.tui import CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("SessionList") is not None
        assert app.query_one("ConversationView") is not None
        assert app.query_one("InputBar") is not None


@pytest.mark.asyncio
async def test_conversation_view_renders_markdown() -> None:
    """ConversationView accepts Markdown + shows it."""
    from cothis.tui import ConversationView, CothisApp

    app = CothisApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(ConversationView)
        view.append_markdown("# Heading\n\nbody text")
        await pilot.pause()
        # The view's child Static/Markdown widget renders the text.
        assert "Heading" in view.renderable_str or "body text" in view.renderable_str


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
