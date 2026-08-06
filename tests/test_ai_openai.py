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
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
)

from cothis.ai._translate import (
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    map_openai_finish_reason,
)
from cothis.ai.openai import OpenAIProvider, _derive_prompt_cache_key

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
    first_block, second_block = msg.content[0], msg.content[1]
    assert isinstance(first_block, TextBlock)
    assert isinstance(second_block, ToolUseBlock)
    assert first_block.text == "hello"
    assert second_block.id == "tu_9"
    assert second_block.name == "fs.read"
    assert second_block.input == {"path": "/a"}

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
    reasoning: str | None = None,
    reasoning_field: str = "reasoning_content",
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=None, function_call=None)
    # Reasoning-model trace arrives as an undocumented extra on the wire
    # (the openai SDK does not type it). Attach it under the requested wire
    # name so the synthesiser's check-both-names path is exercised.
    if reasoning is not None:
        setattr(delta, reasoning_field, reasoning)
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


# ---------------------------------------------------------------------------
# Stream: reasoning-model reasoning trace -> ThinkingBlock lifecycle
# ---------------------------------------------------------------------------


def test_stream_reasoning_only_emits_one_thinking_block_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reasoning turn emits one ThinkingBlock start, ThinkingDelta deltas on
    that index (joined text == full trace), one content_block_stop, then the
    text-block lifecycle for the answer, then exactly one message_stop."""
    captured: dict[str, Any] = {}
    chunks = [
        _chunk(reasoning="think"),
        _chunk(reasoning="ing"),
        _chunk(content="hi"),
        _chunk(finish="stop"),
    ]
    _install_streaming_fake(monkeypatch, chunks, captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="gpt-test", messages=[], max_tokens=10)

    assert isinstance(events[0], RawMessageStartEvent)

    block_starts = [e for e in events if isinstance(e, RawContentBlockStartEvent)]
    thinking_starts = [e for e in block_starts if isinstance(e.content_block, ThinkingBlock)]
    text_starts = [e for e in block_starts if isinstance(e.content_block, TextBlock)]
    assert len(thinking_starts) == 1
    assert len(text_starts) == 1
    thinking_idx = thinking_starts[0].index
    text_idx = text_starts[0].index
    assert thinking_idx != text_idx

    deltas = [e for e in events if isinstance(e, RawContentBlockDeltaEvent)]
    thinking_deltas = [d for d in deltas if isinstance(d.delta, ThinkingDelta)]
    assert len(thinking_deltas) == 2
    assert all(d.index == thinking_idx for d in thinking_deltas)
    assert "".join(d.delta.thinking for d in thinking_deltas) == "thinking"

    # Exactly one content_block_stop on the thinking index.
    thinking_stops = [
        e
        for e in events
        if isinstance(e, RawContentBlockStopEvent) and e.index == thinking_idx
    ]
    assert len(thinking_stops) == 1

    # The answer's text-block lifecycle follows.
    text_deltas = [d for d in deltas if isinstance(d.delta, TextDelta)]
    assert "".join(d.delta.text for d in text_deltas) == "hi"

    stops = [e for e in events if isinstance(e, RawMessageStopEvent)]
    assert len(stops) == 1


def test_stream_reasoning_via_reasoning_field_name_matches_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthesiser reads both wire names — populating ``reasoning`` instead
    of ``reasoning_content`` still emits the ThinkingBlock lifecycle (the
    check-both-names defence is the unit under test)."""
    captured: dict[str, Any] = {}
    chunks = [
        _chunk(reasoning="plan", reasoning_field="reasoning"),
        _chunk(content="answer"),
        _chunk(finish="stop"),
    ]
    _install_streaming_fake(monkeypatch, chunks, captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="m", messages=[], max_tokens=10)

    block_starts = [e for e in events if isinstance(e, RawContentBlockStartEvent)]
    thinking_starts = [e for e in block_starts if isinstance(e.content_block, ThinkingBlock)]
    assert len(thinking_starts) == 1
    thinking_deltas = [
        d
        for d in events
        if isinstance(d, RawContentBlockDeltaEvent) and isinstance(d.delta, ThinkingDelta)
    ]
    assert "".join(d.delta.thinking for d in thinking_deltas) == "plan"


def test_stream_non_reasoning_model_emits_no_thinking_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: when the backend sends no reasoning, the new branch
    must not fire — no ThinkingBlock start, no ThinkingDelta."""
    captured: dict[str, Any] = {}
    chunks = [
        _chunk(content="Hel"),
        _chunk(content="lo"),
        _chunk(finish="stop"),
    ]
    _install_streaming_fake(monkeypatch, chunks, captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="gpt-test", messages=[], max_tokens=10)

    block_starts = [e for e in events if isinstance(e, RawContentBlockStartEvent)]
    assert not any(isinstance(e.content_block, ThinkingBlock) for e in block_starts)
    deltas = [e for e in events if isinstance(e, RawContentBlockDeltaEvent)]
    assert not any(isinstance(d.delta, ThinkingDelta) for d in deltas)
    # The existing contract still holds.
    assert len([e for e in events if isinstance(e, RawMessageStopEvent)]) == 1


def test_stream_reasoning_then_text_closes_thinking_before_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thinking content_block_stop is emitted BEFORE the text
    content_block_start; both blocks present on distinct indices."""
    captured: dict[str, Any] = {}
    chunks = [
        _chunk(reasoning="plan"),
        _chunk(content="answer"),
        _chunk(finish="stop"),
    ]
    _install_streaming_fake(monkeypatch, chunks, captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="m", messages=[], max_tokens=10)

    thinking_idx = next(
        e.index
        for e in events
        if isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ThinkingBlock)
    )
    text_idx = next(
        e.index
        for e in events
        if isinstance(e, RawContentBlockStartEvent) and isinstance(e.content_block, TextBlock)
    )
    thinking_stop_pos = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, RawContentBlockStopEvent) and e.index == thinking_idx
    )
    text_start_pos = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, RawContentBlockStartEvent) and e.index == text_idx
    )
    assert thinking_stop_pos < text_start_pos
    assert thinking_idx != text_idx
    assert len([e for e in events if isinstance(e, RawMessageStopEvent)]) == 1


def test_stream_reasoning_then_tool_call_closes_thinking_before_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thinking block is closed before the tool_use content_block_start."""
    captured: dict[str, Any] = {}
    chunks = [
        _chunk(reasoning="plan"),
        _chunk(tool_calls=[{"index": 0, "id": "tu_1", "name": "t", "arguments": "{}"}]),
        _chunk(finish="tool_calls"),
    ]
    _install_streaming_fake(monkeypatch, chunks, captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="m", messages=[], max_tokens=10)

    thinking_idx = next(
        e.index
        for e in events
        if isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ThinkingBlock)
    )
    tool_idx = next(
        e.index
        for e in events
        if isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ToolUseBlock)
    )
    thinking_stop_pos = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, RawContentBlockStopEvent) and e.index == thinking_idx
    )
    tool_start_pos = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, RawContentBlockStartEvent) and e.index == tool_idx
    )
    assert thinking_stop_pos < tool_start_pos
    assert len([e for e in events if isinstance(e, RawMessageStopEvent)]) == 1


def test_stream_reasoning_no_explicit_finish_closes_thinking_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reasoning stream that ends without finish_reason still closes the
    open thinking block and emits exactly one message_stop with
    stop_reason='end_turn' (extends the no-finish contract to thinking)."""
    captured: dict[str, Any] = {}
    _install_streaming_fake(monkeypatch, [_chunk(reasoning="plan")], captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    events = _run_stream(provider, model="m", messages=[], max_tokens=10)

    thinking_idx = next(
        e.index
        for e in events
        if isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ThinkingBlock)
    )
    # The fallback tail emits content_block_stop for the thinking block.
    assert any(
        isinstance(e, RawContentBlockStopEvent) and e.index == thinking_idx
        for e in events
    )
    assert len([e for e in events if isinstance(e, RawMessageStopEvent)]) == 1
    msg_deltas = [e for e in events if isinstance(e, RawMessageDeltaEvent)]
    assert msg_deltas[-1].delta.stop_reason == "end_turn"


def test_synthesised_thinking_block_round_trips_through_agent_accumulator() -> None:
    """The synthesised ThinkingBlock start + ThinkingDelta deltas + stop match
    what the agent accumulator's thinking branch expects: block
    type=='thinking' and the ``thinking`` field joins to the full string.
    Guards against a shape mismatch that would silently drop the block."""
    from cothis.agent import (
        _apply_stream_delta,
        _finalize_stream_block,
        _init_stream_block,
    )

    start_block = ThinkingBlock(type="thinking", thinking="", signature="")
    block = _init_stream_block(
        RawContentBlockStartEvent(
            type="content_block_start", index=0, content_block=start_block
        ).content_block
    )
    assert block["type"] == "thinking"

    for piece in ("think", "ing"):
        _apply_stream_delta(
            block,
            ThinkingDelta(type="thinking_delta", thinking=piece),
        )
    _finalize_stream_block(block)

    assert block["type"] == "thinking"
    assert block["thinking"] == "thinking"


# ---------------------------------------------------------------------------
# Prompt-cache hints: per-session ``prompt_cache_key`` (I22).
# ---------------------------------------------------------------------------


def test_derive_prompt_cache_key_is_deterministic_and_length_bounded() -> None:
    """``_derive_prompt_cache_key`` is stable for one input and ≤64 chars."""
    key = _derive_prompt_cache_key("abc")
    again = _derive_prompt_cache_key("abc")
    assert key == again  # deterministic
    assert len(key) <= 64  # within OpenAI's bound
    assert len(key) == 32  # sha256 prefix[:32]


def test_derive_prompt_cache_key_distinct_for_distinct_sessions() -> None:
    assert _derive_prompt_cache_key("session-one") != _derive_prompt_cache_key(
        "session-two"
    )


def test_non_stream_sets_prompt_cache_key_for_real_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_client(
        monkeypatch, create_return=_make_completion(text="hi"), captured=captured
    )
    provider = OpenAIProvider(api_key="sk", api_base=None)  # real OpenAI
    asyncio.run(
        provider.amessages(
            model="gpt-test",
            messages=[],
            max_tokens=1,
            session_id="abc",
        )
    )
    req = captured["create_kwargs"]
    assert req["prompt_cache_key"] == _derive_prompt_cache_key("abc")


def test_prompt_cache_key_stable_across_calls_within_one_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_client(
        monkeypatch, create_return=_make_completion(text="hi"), captured=captured
    )
    provider = OpenAIProvider(api_key="sk", api_base=None)
    asyncio.run(
        provider.amessages(model="m", messages=[], max_tokens=1, session_id="same")
    )
    first = captured["create_kwargs"]["prompt_cache_key"]
    captured["create_kwargs"] = {}
    asyncio.run(
        provider.amessages(model="m", messages=[], max_tokens=1, session_id="same")
    )
    second = captured["create_kwargs"]["prompt_cache_key"]
    assert first == second  # stable across turns within one session


def test_prompt_cache_key_distinct_across_two_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_client(
        monkeypatch, create_return=_make_completion(text="hi"), captured=captured
    )
    provider = OpenAIProvider(api_key="sk", api_base=None)
    asyncio.run(
        provider.amessages(model="m", messages=[], max_tokens=1, session_id="sess-a")
    )
    first = captured["create_kwargs"]["prompt_cache_key"]
    captured["create_kwargs"] = {}
    asyncio.run(
        provider.amessages(model="m", messages=[], max_tokens=1, session_id="sess-b")
    )
    second = captured["create_kwargs"]["prompt_cache_key"]
    assert first != second  # distinct across sessions


def test_prompt_cache_key_absent_when_no_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ephemeral ``ask`` path (no session attached) omits the key — OpenAI
    still auto-caches by prefix, so this degrades gracefully, not a crash."""
    captured: dict[str, Any] = {}
    _install_fake_client(
        monkeypatch, create_return=_make_completion(text="hi"), captured=captured
    )
    provider = OpenAIProvider(api_key="sk", api_base=None)
    asyncio.run(provider.amessages(model="m", messages=[], max_tokens=1))
    assert "prompt_cache_key" not in captured["create_kwargs"]


def test_prompt_cache_key_absent_for_openai_compat_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI-compatible backends (OpenRouter / DeepSeek / Groq / Mistral) 400
    on unknown top-level params, so the key is gated off when ``api_base`` is
    set — covers the ``api_base is None`` gate pin."""
    captured: dict[str, Any] = {}
    _install_fake_client(
        monkeypatch, create_return=_make_completion(text="hi"), captured=captured
    )
    provider = OpenAIProvider(
        api_key="sk", api_base="https://api.openrouter.ai/api/v1"
    )
    asyncio.run(
        provider.amessages(
            model="m", messages=[], max_tokens=1, session_id="abc"
        )
    )
    assert "prompt_cache_key" not in captured["create_kwargs"]


def test_stream_sets_prompt_cache_key_for_real_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_streaming_fake(monkeypatch, [_chunk(finish="stop")], captured)
    provider = OpenAIProvider(api_key="sk", api_base=None)
    _run_stream(provider, model="m", messages=[], max_tokens=1, session_id="abc")
    assert captured["create_kwargs"]["prompt_cache_key"] == _derive_prompt_cache_key(
        "abc"
    )
