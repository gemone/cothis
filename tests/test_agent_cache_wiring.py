"""End-to-end wiring of the session id into ``amessages``.

The agent threads ``session_id`` from the attached :class:`~cothis.session.Session`
into every provider ``amessages`` call (both the non-stream ``run`` and the
streaming ``run_stream`` paths) so per-session prompt-cache hints fire. When
no session is attached (the ephemeral ``ask`` path) ``session_id=None`` must
reach the provider instead of a runtime error.

The provider is replaced with a recording fake — no network. The session is a
minimal stand-in exposing the surface the agent loop touches (``session_id``,
``messages``, ``active_skills``, ``append_message``).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from anthropic.types import (
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    Usage,
)
from anthropic.types.message import Message
from anthropic.types.raw_message_delta_event import Delta

from cothis.agent import Agent


class _FakeSession:
    """Minimal Session stand-in for the agent loop's session surface."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages: list[dict[str, Any]] = []
        self.active_skills: frozenset[str] = frozenset()

    def append_message(self, role: str, content: Any) -> None:
        self.messages.append({"role": role, "content": content})

    def append_block(self, role: str, block: Any) -> None:
        self.messages.append({"role": role, "content": [block]})


def _msg_response(text: str) -> Message:
    return Message(
        id="m1",
        model="test-model",
        role="assistant",
        type="message",
        content=[TextBlock(type="text", text=text)],
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _make_agent(monkeypatch: pytest.MonkeyPatch) -> Agent:
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **kw: MagicMock())
    return Agent(model="x", provider="openrouter", tools=[], max_iterations=3)


def _stream_from(events: list[Any]) -> Any:
    async def gen() -> Any:
        for e in events:
            yield e

    return gen()


def _stream_events(text: str = "hi") -> list[Any]:
    return [
        RawMessageStartEvent(
            type="message_start",
            message=Message(
                id="m1",
                model="test-model",
                role="assistant",
                type="message",
                content=[],
                stop_reason=None,
                usage=Usage(input_tokens=1, output_tokens=0),
            ),
        ),
        RawContentBlockStartEvent(
            type="content_block_start", index=0, content_block=TextBlock(type="text", text="")
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text=text)
        ),
        RawContentBlockStopEvent(type="content_block_stop", index=0),
        RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason="end_turn"),
            usage=MessageDeltaUsage(input_tokens=1, output_tokens=1),
        ),
        RawMessageStopEvent(type="message_stop"),
    ]


# --------------------------------------------------------------------------- run


def test_run_threads_session_id_when_session_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    # Bind the session directly rather than via ``attach_session`` so the fake
    # does not have to satisfy the real ``Session`` type (attach_session also
    # does message-seeding + notify-bus wiring this fake does not exercise).
    agent_any: Any = agent
    agent_any._session = _FakeSession("sess-123")
    seen: list[Any] = []

    async def fake_amessages(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _msg_response("done")

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    assert asyncio.run(agent.run("hi")) == "done"
    assert seen, "amessages was not called"
    assert seen[0]["session_id"] == "sess-123"


def test_run_passes_none_when_no_session(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(monkeypatch)  # no attach_session → ask / ephemeral
    seen: list[Any] = []

    async def fake_amessages(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _msg_response("done")

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    assert asyncio.run(agent.run("hi")) == "done"
    assert seen, "amessages was not called"
    assert seen[0]["session_id"] is None


# ---------------------------------------------------------------------- run_stream


@pytest.mark.asyncio
async def test_run_stream_threads_session_id_when_session_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    agent_any: Any = agent
    agent_any._session = _FakeSession("sess-456")
    seen: list[Any] = []

    async def fake_amessages(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _stream_from(_stream_events())

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    async for _ in agent.run_stream("hi"):
        pass
    assert seen, "amessages was not called"
    assert seen[0]["session_id"] == "sess-456"


@pytest.mark.asyncio
async def test_run_stream_passes_none_when_no_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)  # no attach_session
    seen: list[Any] = []

    async def fake_amessages(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _stream_from(_stream_events())

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    async for _ in agent.run_stream("hi"):
        pass
    assert seen, "amessages was not called"
    assert seen[0]["session_id"] is None
