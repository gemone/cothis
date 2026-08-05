"""Tests for ``cothis.acp_bridge.AgentSessionBackend``.

Hermetic: a fake agent factory returns an object whose ``run_stream`` yields
real ``ContentDelta`` / ``ToolCallEvent`` / ``ToolResultEvent`` dataclasses.
No LLM, no network. Asserts the run-stream → TranscriptProgress translation
and the in-memory session/snapshot bookkeeping.
"""

from __future__ import annotations

from typing import Any

import pytest

from cothis.acp_bridge import AgentSessionBackend
from cothis.agent import ContentDelta, ToolCallEvent, ToolResultEvent
from cothis.protocol.messages import BackendError


class _FakeAgent:
    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)

    async def run_stream(self, _text: str):  # type: ignore[no-untyped-def]
        for event in self._events:
            yield event


def _backend(events: list[Any]) -> AgentSessionBackend:
    """A backend whose every session gets a fake agent yielding *events*."""
    return AgentSessionBackend(
        provider="p",
        model="m",
        make_agent=lambda **_kw: _FakeAgent(events),
    )


def _collector(into: list):
    """An async emit callback that appends each progress update."""

    async def emit(progress: Any) -> None:
        into.append(progress)

    return emit


async def _noop_emit(_progress: Any) -> None:
    """An async emit callback that discards progress (for snapshot-only tests)."""
    return None


@pytest.mark.asyncio
async def test_create_and_list_session() -> None:
    backend = _backend([])
    snap = await backend.create_session("/tmp", "demo", None, None)
    assert snap.cwd == "/tmp" and snap.name == "demo"
    assert snap.transcript == [] and snap.revision == 0

    sessions = await backend.list_sessions()
    assert [s.id for s in sessions] == [snap.id]
    assert sessions[0].name == "demo"


@pytest.mark.asyncio
async def test_prompt_unknown_session_raises_not_found() -> None:
    backend = _backend([])
    with pytest.raises(BackendError) as exc:
        await backend.prompt("nope", "hi", _noop_emit)
    assert exc.value.error.code == "not_found"


@pytest.mark.asyncio
async def test_prompt_translates_text_delta_to_progress() -> None:
    events = [ContentDelta(kind="text", text="Hello "), ContentDelta(kind="text", text="world")]
    backend = _backend(events)
    snap = await backend.create_session("/", None, None, None)

    emitted: list = []
    await backend.prompt(snap.id, "hi", _collector(emitted))

    # item_started(streaming) + two assistant_delta + item_finished(stop) at turn end.
    assert [p.type for p in emitted] == [
        "item_started", "assistant_delta", "assistant_delta", "item_finished"
    ]
    assert emitted[1].delta == "Hello " and emitted[2].delta == "world"
    # The finished item carries the accumulated text.
    assert emitted[3].item.content[0].text == "Hello world"


@pytest.mark.asyncio
async def test_prompt_emits_finished_with_accumulated_text() -> None:
    events = [ContentDelta(kind="text", text="AB")]
    backend = _backend(events)
    snap = await backend.create_session("/", None, None, None)

    await backend.prompt(snap.id, "hi", _noop_emit)

    # The snapshot returned by prompt carries the grown transcript.
    snap2 = await backend.create_session("/", None, None, None)
    after = await backend.prompt(snap2.id, "hi", _noop_emit)
    transcript = after.transcript
    assert [i.role for i in transcript] == ["user", "assistant"]
    assistant = transcript[1]
    assert assistant.role == "assistant"  # narrows to AssistantTranscriptItem
    assert assistant.status == "complete"
    assert assistant.stopReason == "stop"
    # Accumulated text is authoritative on the finished item.
    first = assistant.content[0]
    assert first.type == "text"  # narrows to TextContent
    assert first.text == "AB"


@pytest.mark.asyncio
async def test_tool_call_splits_assistant_message() -> None:
    events = [
        ContentDelta(kind="text", text="thinking..."),
        ToolCallEvent(name="fs.read", arguments={"path": "/x"}, call_id="tu_1"),
        ToolResultEvent(tool="fs.read", is_error=False, duration_ms=5, result_pointer=None, call_id="tu_1"),
        ContentDelta(kind="text", text="done"),
    ]
    backend = _backend(events)
    snap = await backend.create_session("/", None, None, None)

    emitted: list = []
    await backend.prompt(snap.id, "hi", _collector(emitted))

    kinds = [p.type for p in emitted]
    # assistant starts, delta; tool splits -> assistant finished(toolUse),
    # tool started; tool finished; then a second assistant message starts,
    # delta, finished(stop).
    assert kinds == [
        "item_started",          # assistant #1 streaming
        "assistant_delta",       # "thinking..."
        "item_finished",         # assistant #1 complete (toolUse)
        "item_started",          # tool running
        "item_finished",         # tool complete
        "item_started",          # assistant #2 streaming
        "assistant_delta",       # "done"
        "item_finished",         # assistant #2 complete (stop)
    ]
    assert emitted[2].item.stopReason == "toolUse"
    tool_started = emitted[3].item
    assert tool_started.role == "tool" and tool_started.status == "running"
    assert tool_started.toolName == "fs.read"
    tool_finished = emitted[4].item
    assert tool_finished.status == "complete" and not tool_finished.isError
    assert emitted[-1].item.stopReason == "stop"


@pytest.mark.asyncio
async def test_tool_error_marks_item_error() -> None:
    events = [
        ToolCallEvent(name="bash", arguments={}, call_id="tu_e"),
        ToolResultEvent(tool="bash", is_error=True, duration_ms=1, result_pointer=None, call_id="tu_e"),
    ]
    backend = _backend(events)
    snap = await backend.create_session("/", None, None, None)

    emitted: list = []
    await backend.prompt(snap.id, "hi", _collector(emitted))

    tool_finished = next(p for p in emitted if p.type == "item_finished" and p.item.role == "tool")
    assert tool_finished.item.status == "error" and tool_finished.item.isError


@pytest.mark.asyncio
async def test_models_advertises_configured_model_with_limits() -> None:
    # A backend configured for the default cothis model advertises exactly
    # that (provider, id), enriched with the limits bundled metadata resolves.
    backend = AgentSessionBackend(
        provider="openrouter",
        model="openai/gpt-oss-120b",
        make_agent=lambda **_kw: _FakeAgent([]),
    )
    [advertised] = await backend.models()
    assert advertised.provider == "openrouter"
    assert advertised.id == "openai/gpt-oss-120b"
    # litellm knows this model; both limits are populated.
    assert advertised.maxOutputTokens == 32768
    assert advertised.contextWindow == 131072


@pytest.mark.asyncio
async def test_models_unknown_model_advertises_none_limits() -> None:
    # An unknown configured model is still advertised by id; the limits are
    # honestly None rather than invented.
    backend = AgentSessionBackend(
        provider="p",
        model="no-such-model",
        make_agent=lambda **_kw: _FakeAgent([]),
    )
    [advertised] = await backend.models()
    assert advertised.provider == "p"
    assert advertised.id == "no-such-model"
    assert advertised.maxOutputTokens is None
    assert advertised.contextWindow is None
