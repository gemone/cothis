"""Tests for :class:`cothis.ai.anthropic.AnthropicProvider`.

The Anthropic provider is a near-zero-translation pass-through, so the
contract is: ``amessages`` forwards kwargs verbatim to
``messages.create`` / ``messages.stream`` and returns / yields the SDK's
native objects unchanged. The SDK client is mocked — no network.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from anthropic.types import (
    MessageDeltaUsage,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    Usage,
)
from anthropic.types.message import Message
from anthropic.types.raw_message_delta_event import Delta

from cothis.ai.anthropic import AnthropicProvider

if TYPE_CHECKING:
    import pytest


class _FakeMessages:
    """Records create kwargs + stream kwargs; returns canned results."""

    def __init__(self) -> None:
        self.create_kwargs: dict[str, Any] = {}
        self.stream_kwargs: dict[str, Any] = {}
        self.create_result: Any = None
        self.stream_events: list[Any] = []

    async def create(self, **kwargs: Any) -> Any:
        self.create_kwargs = kwargs
        return self.create_result

    def stream(self, **kwargs: Any) -> _FakeStreamCM:
        self.stream_kwargs = kwargs
        return _FakeStreamCM(self.stream_events)


class _FakeStreamCM:
    """Async context manager mimicking ``client.messages.stream(...)``."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeStreamCM:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> _FakeStreamCM:
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def _build_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[AnthropicProvider, _FakeClient]:
    client = _FakeClient()
    monkeypatch.setattr(
        "anthropic.AsyncAnthropic", lambda **kw: _ensure_no_creds(kw) or client
    )
    provider = AnthropicProvider(api_key=None, api_base=None)
    return provider, client


def _ensure_no_creds(kwargs: dict[str, Any]) -> None:
    """AnthropicProvider omits api_key/base_url when caller passed None."""
    assert "api_key" not in kwargs
    assert "base_url" not in kwargs


def test_non_stream_forwards_kwargs_and_returns_sdk_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, client = _build_provider(monkeypatch)
    sentinel = Message(
        id="msg_1",
        model="claude-test",
        role="assistant",
        type="message",
        content=[TextBlock(type="text", text="hi")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=3, output_tokens=4),
    )
    client.messages.create_result = sentinel

    msgs = [{"role": "user", "content": [{"type": "text", "text": "ping"}]}]
    result = asyncio.run(
        provider.amessages(
            model="claude-test",
            messages=msgs,
            max_tokens=128,
            system=[{"type": "text", "text": "sys"}],
            tools=[
                {"name": "t", "description": "d", "input_schema": {"type": "object"}}
            ],
        )
    )

    assert result is sentinel  # pass-through returns the SDK Message unchanged
    kwargs = client.messages.create_kwargs
    assert kwargs["model"] == "claude-test"
    assert kwargs["messages"] is msgs
    assert kwargs["max_tokens"] == 128
    assert kwargs["system"] == [{"type": "text", "text": "sys"}]
    assert kwargs["tools"] == [
        {"name": "t", "description": "d", "input_schema": {"type": "object"}}
    ]


def test_non_stream_omits_optional_kwargs_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, client = _build_provider(monkeypatch)
    client.messages.create_result = MagicMock()
    asyncio.run(provider.amessages(model="m", messages=[], max_tokens=1))
    kwargs = client.messages.create_kwargs
    assert "system" not in kwargs
    assert "tools" not in kwargs


def test_stream_yields_sdk_events_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """stream=True returns an async iterator yielding the SDK's events unchanged."""
    provider, client = _build_provider(monkeypatch)
    events = [
        RawMessageStartEvent(
            type="message_start",
            message=Message(
                id="m",
                model="claude-test",
                role="assistant",
                type="message",
                content=[],
                stop_reason=None,
                usage=Usage(input_tokens=0, output_tokens=0),
            ),
        ),
        RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason="end_turn"),
            usage=MessageDeltaUsage(input_tokens=0, output_tokens=0),
        ),
        RawMessageStopEvent(type="message_stop"),
    ]
    client.messages.stream_events = list(events)

    async def driver() -> list[Any]:
        # Create + iterate the async generator inside one event loop.
        stream = await provider.amessages(
            model="claude-test", messages=[], max_tokens=10, stream=True
        )
        return [ev async for ev in stream]

    yielded = asyncio.run(driver())

    assert yielded == events  # identity-preserving pass-through
    # stream kwargs forwarded too
    assert client.messages.stream_kwargs == {
        "model": "claude-test",
        "messages": [],
        "max_tokens": 10,
    }


def test_stream_uses_messages_stream_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider routes stream=True through ``messages.stream`` (not create)."""
    provider, client = _build_provider(monkeypatch)
    client.messages.stream_events = [RawMessageStopEvent(type="message_stop")]

    async def driver() -> None:
        stream = await provider.amessages(
            model="m", messages=[], max_tokens=1, stream=True
        )
        async for _ in stream:
            pass

    asyncio.run(driver())
    assert client.messages.stream_kwargs == {
        "model": "m",
        "messages": [],
        "max_tokens": 1,
    }
    assert client.messages.create_kwargs == {}  # create not invoked
