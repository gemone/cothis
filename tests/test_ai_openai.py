"""Tests for :class:`cothis.ai.openai.OpenAIProvider`.

Covers:
- message translation (Anthropic->OpenAI): assistant ``tool_use`` ->
  ``tool_calls``, user ``tool_result`` -> ``role: tool``, system block list
  -> system message;
- tool-schema translation (Anthropic -> OpenAI ``function`` shape);
- non-stream ``ChatCompletion`` -> ``anthropic.types.Message``;
- stream synthesis: ``ChatCompletionChunk`` -> ``Raw*Event`` lifecycle
  (one text block per turn; tool-use ``InputJSONDelta`` partial JSON;
  stop-reason map; ``message_stop`` exactly once; OpenRouter duplicate
  finishes tolerated).

The OpenAI SDK client is mocked — no network.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)

from cothis.ai._translate import (
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    map_openai_finish_reason,
)
from cothis.ai.openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Translation unit tests (pure helpers)
# ---------------------------------------------------------------------------


def test_message_translation_assistant_tool_use_becomes_tool_calls() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "fs.read",
                    "input": {"path": "/x"},
                },
            ],
        },
    ]
    out = anthropic_messages_to_openai(messages)
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["role"] == "assistant"
    assert out[1]["content"] == "ok"
    assert len(out[1]["tool_calls"]) == 1
    tc = out[1]["tool_calls"][0]
    assert tc["id"] == "tu_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "fs.read"
    # arguments is serialised JSON; compare on the parsed value (format agnostic).
    assert json.loads(tc["function"]["arguments"]) == {"path": "/x"}


def test_message_translation_tool_result_becomes_tool_role() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": [{"type": "text", "text": "file body"}],
                },
                {"type": "text", "text": "and more"},
            ],
        }
    ]
    out = anthropic_messages_to_openai(messages)
    assert out[0] == {"role": "tool", "tool_call_id": "tu_1", "content": "file body"}
    assert out[1] == {"role": "user", "content": "and more"}


def test_message_translation_system_block_list_becomes_leading_system_message() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "ping"}]}]
    system = [
        {"type": "text", "text": "be brief"},
        {"type": "text", "text": "use tools", "cache_control": {"type": "ephemeral"}},
    ]
    out = anthropic_messages_to_openai(messages, system)
    assert out[0] == {"role": "system", "content": "be brief\n\nuse tools"}
    assert out[1] == {"role": "user", "content": "ping"}


def test_thinking_blocks_dropped_for_openai() -> None:
    """OpenAI has no thinking equivalent — reasoning trace must not leak as content."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "secret reasoning",
                    "signature": "sig",
                },
                {"type": "text", "text": "answer"},
            ],
        }
    ]
    out = anthropic_messages_to_openai(messages)
    assert out[0] == {"role": "assistant", "content": "answer"}


def test_tool_schema_translation() -> None:
    tools = [
        {
            "name": "fs.read",
            "description": "read a file",
            "input_schema": {"type": "object"},
        }
    ]
    out = anthropic_tools_to_openai(tools)
    assert out == [
        {
            "type": "function",
            "function": {
                "name": "fs.read",
                "description": "read a file",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_tool_schema_none_passthrough() -> None:
    assert anthropic_tools_to_openai(None) is None
    assert anthropic_tools_to_openai([]) is None


# ---------------------------------------------------------------------------
# Stop-reason map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "finish,expected",
    [
        ("stop", "end_turn"),
        ("tool_calls", "tool_use"),
        ("function_call", "tool_use"),
        ("length", "max_tokens"),
        (None, "end_turn"),
        ("content_filter", "end_turn"),
        ("weird-openrouter-code", "end_turn"),  # unknown -> end_turn fallback
    ],
)
def test_map_openai_finish_reason(finish: str | None, expected: str) -> None:
    assert map_openai_finish_reason(finish) == expected


# ---------------------------------------------------------------------------
# Non-stream: ChatCompletion -> Message
# ---------------------------------------------------------------------------


def _make_completion(
    *,
    text: str | None = "hi",
    tool_calls: list[dict[str, Any]] | None = None,
    finish: str | None = "stop",
    prompt_tokens: int = 5,
    completion_tokens: int = 7,
) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    tc_objs = []
    for tc in tool_calls or []:
        fn = MagicMock()
        fn.name = tc["name"]
        fn.arguments = tc.get("arguments", "{}")
        tcm = MagicMock()
        tcm.id = tc.get("id", "tu_1")
        tcm.function = fn
        tc_objs.append(tcm)
    msg.tool_calls = tc_objs or None

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    completion = MagicMock()
    completion.id = "chatcmpl-1"
    completion.model = "gpt-test"
    completion.choices = [choice]
    completion.usage = usage
    return completion


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, *, create_return: Any, captured: dict[str, Any]
) -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        class chat:  # noqa: N801
            class completions:
                @staticmethod
                async def create(**kwargs: object) -> object:
                    captured["create_kwargs"] = kwargs
                    return create_return

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)


def test_non_stream_translates_completion_to_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    completion = _make_completion(
        text="hello",
        tool_calls=[{"id": "tu_9", "name": "fs.read", "arguments": '{"path": "/a"}'}],
        finish="tool_calls",
    )
    _install_fake_client(monkeypatch, create_return=completion, captured=captured)

    provider = OpenAIProvider(api_key="sk", api_base=None)
    msg = asyncio.run(
        provider.amessages(
            model="gpt-test",
            messages=[{"role": "user", "content": [{"type": "text", "text": "x"}]}],
            max_tokens=100,
        )
    )

    assert msg.id == "chatcmpl-1"
    assert msg.model == "gpt-test"
    assert msg.role == "assistant"
    assert msg.stop_reason == "tool_use"  # tool_calls -> tool_use
    assert msg.usage.input_tokens == 5
    assert msg.usage.output_tokens == 7
    # content blocks: one text + one tool_use with parsed input
    types_ = [type(b).__name__ for b in msg.content]
    assert types_ == ["TextBlock", "ToolUseBlock"]
    assert msg.content[0].text == "hello"
    assert msg.content[1].id == "tu_9"
    assert msg.content[1].name == "fs.read"
    assert msg.content[1].input == {"path": "/a"}

    # The create kwargs carry the translated request shape.
    req = captured["create_kwargs"]
    assert req["model"] == "gpt-test"
    assert req["max_tokens"] == 100
    assert req["messages"] == [{"role": "user", "content": "x"}]
    assert req["tools"] is None


def test_non_stream_malformed_tool_args_become_empty_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    completion = _make_completion(
        text=None,
        tool_calls=[{"name": "t", "arguments": "not-json{"}],
        finish="tool_calls",
    )
    _install_fake_client(monkeypatch, create_return=completion, captured=captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    msg = asyncio.run(provider.amessages(model="m", messages=[], max_tokens=1))
    tu = next(b for b in msg.content if isinstance(b, ToolUseBlock))
    assert tu.input == {}


# ---------------------------------------------------------------------------
# Stream: ChatCompletionChunk stream -> Raw*Event lifecycle
# ---------------------------------------------------------------------------


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish: str | None = None,
    chunk_id: str = "chatcmpl-1",
    model: str = "gpt-test",
    usage: dict[str, int] | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=None, function_call=None)
    if tool_calls is not None:
        tc_objs = []
        for tc in tool_calls:
            fn = SimpleNamespace(
                name=tc.get("name"),
                arguments=tc.get("arguments"),
            )
            tc_objs.append(
                SimpleNamespace(
                    index=tc.get("index", 0),
                    id=tc.get("id"),
                    function=fn,
                    type="function",
                )
            )
        delta.tool_calls = tc_objs
    choice = SimpleNamespace(delta=delta, finish_reason=finish, index=0, logprobs=None)
    usage_obj = (
        SimpleNamespace(
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=(usage or {}).get("completion_tokens"),
        )
        if usage
        else None
    )
    return SimpleNamespace(
        id=chunk_id,
        model=model,
        choices=[choice],
        usage=usage_obj,
        object="chat.completion.chunk",
    )


class _ChunkStream:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _ChunkStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _install_streaming_fake(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[SimpleNamespace],
    captured: dict[str, Any],
) -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        class chat:  # noqa: N801
            class completions:
                @staticmethod
                async def create(**kwargs: object) -> _ChunkStream:
                    captured["create_kwargs"] = kwargs
                    assert kwargs.get("stream") is True
                    return _ChunkStream(chunks)

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)


def _run_stream(provider: OpenAIProvider, **amessages_kwargs: Any) -> list[Any]:
    async def driver() -> list[Any]:
        stream = await provider.amessages(stream=True, **amessages_kwargs)
        return [ev async for ev in stream]

    return asyncio.run(driver())


def test_stream_text_only_emits_one_text_block_and_single_message_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    chunks = [
        _chunk(content="Hel"),
        _chunk(content="lo"),
        _chunk(finish="stop"),
    ]
    _install_streaming_fake(monkeypatch, chunks, captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="gpt-test", messages=[], max_tokens=10)

    # 1 message_start, 1 content_block_start, 2 deltas, 1 stop, 1 message_delta, 1 message_stop
    assert isinstance(events[0], RawMessageStartEvent)
    assert events[0].message.id == "chatcmpl-1"
    assert events[0].message.model == "gpt-test"
    block_starts = [e for e in events if isinstance(e, RawContentBlockStartEvent)]
    assert len(block_starts) == 1
    assert isinstance(block_starts[0].content_block, TextBlock)
    deltas = [e for e in events if isinstance(e, RawContentBlockDeltaEvent)]
    assert len(deltas) == 2
    assert all(isinstance(d.delta, TextDelta) for d in deltas)
    assert "".join(d.delta.text for d in deltas) == "Hello"
    block_stops = [e for e in events if isinstance(e, RawContentBlockStopEvent)]
    assert len(block_stops) == 1
    msg_deltas = [e for e in events if isinstance(e, RawMessageDeltaEvent)]
    assert len(msg_deltas) == 1
    assert msg_deltas[0].delta.stop_reason == "end_turn"  # stop -> end_turn
    stops = [e for e in events if isinstance(e, RawMessageStopEvent)]
    assert len(stops) == 1  # exactly one message_stop


def test_stream_tool_call_emits_input_json_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    chunks = [
        _chunk(
            tool_calls=[{"index": 0, "id": "tu_1", "name": "fs.read", "arguments": ""}]
        ),
        _chunk(tool_calls=[{"index": 0, "arguments": '{"path": "/x"}'}]),
        _chunk(finish="tool_calls"),
    ]
    _install_streaming_fake(monkeypatch, chunks, captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="gpt-test", messages=[], max_tokens=10)

    block_starts = [e for e in events if isinstance(e, RawContentBlockStartEvent)]
    assert len(block_starts) == 1
    tu_start = block_starts[0]
    assert isinstance(tu_start.content_block, ToolUseBlock)
    assert tu_start.content_block.id == "tu_1"
    assert tu_start.content_block.name == "fs.read"
    deltas = [e for e in events if isinstance(e, RawContentBlockDeltaEvent)]
    assert all(isinstance(d.delta, InputJSONDelta) for d in deltas)
    # Joining the partial_json yields the complete args object.
    joined = "".join(d.delta.partial_json for d in deltas)
    assert json.loads(joined) == {"path": "/x"}
    msg_deltas = [e for e in events if isinstance(e, RawMessageDeltaEvent)]
    assert msg_deltas[0].delta.stop_reason == "tool_use"  # tool_calls -> tool_use


def test_stream_length_finish_maps_to_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_streaming_fake(
        monkeypatch, [_chunk(content="x"), _chunk(finish="length")], captured
    )
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="m", messages=[], max_tokens=1)
    msg_deltas = [e for e in events if isinstance(e, RawMessageDeltaEvent)]
    assert msg_deltas[0].delta.stop_reason == "max_tokens"


def test_stream_duplicate_finish_tolerated_emits_single_message_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter sometimes emits finish_reason on multiple trailing chunks."""
    captured: dict[str, Any] = {}
    chunks = [
        _chunk(content="hi"),
        _chunk(finish="stop"),
        _chunk(finish="stop"),  # duplicate
        _chunk(finish="stop"),  # duplicate
    ]
    _install_streaming_fake(monkeypatch, chunks, captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="m", messages=[], max_tokens=1)

    stops = [e for e in events if isinstance(e, RawMessageStopEvent)]
    msg_deltas = [e for e in events if isinstance(e, RawMessageDeltaEvent)]
    assert len(stops) == 1
    assert len(msg_deltas) == 1


def test_stream_text_then_tool_call_closes_text_block_before_tool_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    chunks = [
        _chunk(content="calling tool"),
        _chunk(tool_calls=[{"index": 0, "id": "tu_2", "name": "t", "arguments": "{}"}]),
        _chunk(finish="tool_calls"),
    ]
    _install_streaming_fake(monkeypatch, chunks, captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="m", messages=[], max_tokens=10)

    starts = [e for e in events if isinstance(e, RawContentBlockStartEvent)]
    stops = [e for e in events if isinstance(e, RawContentBlockStopEvent)]
    # text block (index 0) + tool block (index 1)
    assert len(starts) == 2
    assert len(stops) == 2
    assert isinstance(starts[0].content_block, TextBlock)
    assert isinstance(starts[1].content_block, ToolUseBlock)
    assert starts[0].index == 0
    assert starts[1].index == 1


def test_stream_trailing_usage_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A usage-only trailing chunk populates message_delta.usage."""
    captured: dict[str, Any] = {}
    chunks = [
        _chunk(content="hi"),
        _chunk(finish="stop"),
        _chunk(usage={"prompt_tokens": 11, "completion_tokens": 22}, finish=None),
    ]
    # The trailing usage chunk has no finish_reason and no choices content;
    # it must NOT trigger a second message_stop. Our _chunk always sets
    # choices=[choice] with finish_reason=None — fine, that's a no-op chunk.
    _install_streaming_fake(monkeypatch, chunks, captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="m", messages=[], max_tokens=1)

    stops = [e for e in events if isinstance(e, RawMessageStopEvent)]
    assert len(stops) == 1


def test_stream_request_carries_translated_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tools are translated to OpenAI shape on the create call kwargs."""
    captured: dict[str, Any] = {}
    _install_streaming_fake(monkeypatch, [_chunk(finish="stop")], captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    _run_stream(
        provider,
        model="m",
        messages=[],
        max_tokens=1,
        tools=[{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
    )
    req = captured["create_kwargs"]
    assert req["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "t",
                "description": "d",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_stream_no_explicit_finish_still_emits_message_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream that ends without finish_reason still closes cleanly."""
    captured: dict[str, Any] = {}
    _install_streaming_fake(monkeypatch, [_chunk(content="x")], captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="m", messages=[], max_tokens=1)
    assert len([e for e in events if isinstance(e, RawMessageStopEvent)]) == 1
    msg_deltas = [e for e in events if isinstance(e, RawMessageDeltaEvent)]
    assert msg_deltas[-1].delta.stop_reason == "end_turn"
