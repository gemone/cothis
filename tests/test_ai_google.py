"""Tests for :class:`cothis.ai.google.GoogleProvider`.

Covers:
- ``contents`` translation (Anthropic messages -> Google contents: text,
  tool_use -> function_call, tool_result -> function_response);
- tool-schema translation (Anthropic -> Google ``function_declarations``);
- non-stream ``GenerateContentResponse`` -> ``anthropic.types.Message``;
- stream synthesis: text parts -> text-block lifecycle; function_call
  parts -> tool_use block with one complete-JSON ``InputJSONDelta``;
  ``message_stop`` exactly once.

The Google GenAI client is mocked — no network.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

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
    anthropic_messages_to_google_contents,
    anthropic_tools_to_google,
    map_google_finish_reason,
)
from cothis.ai.google import GoogleProvider

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# Translation unit tests
# ---------------------------------------------------------------------------


def test_contents_translation_text_and_tool_use() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling"},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "fs.read",
                    "input": {"path": "/x"},
                },
            ],
        },
    ]
    contents = anthropic_messages_to_google_contents(messages)
    assert contents[0] == {"role": "user", "parts": [{"text": "hi"}]}
    # assistant -> "model"
    assert contents[1]["role"] == "model"
    parts = contents[1]["parts"]
    assert parts[0] == {"text": "calling"}
    assert parts[1] == {
        "function_call": {"name": "fs.read", "args": {"path": "/x"}, "id": "tu_1"}
    }


def test_contents_translation_tool_result_becomes_function_response() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "name": "fs.read",
                    "content": [{"type": "text", "text": "body"}],
                }
            ],
        }
    ]
    contents = anthropic_messages_to_google_contents(messages)
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0] == {
        "function_response": {
            "name": "fs.read",
            "id": "tu_1",
            "response": {"output": "body"},
        }
    }


def test_tool_schema_translation_to_function_declarations() -> None:
    from google.genai import types

    tools = [
        {"name": "fs.read", "description": "read", "input_schema": {"type": "object"}}
    ]
    out = anthropic_tools_to_google(tools)
    assert out is not None
    assert len(out) == 1
    decl = out[0]
    assert isinstance(decl, types.FunctionDeclaration)
    assert decl.name == "fs.read"
    assert decl.description == "read"
    assert decl.parameters is not None  # wrapped into a google Schema


def test_map_google_finish_reason_accepts_enum_name_and_string() -> None:
    assert map_google_finish_reason(SimpleNamespace(name="STOP")) == "end_turn"
    assert map_google_finish_reason("MAX_TOKENS") == "max_tokens"
    assert map_google_finish_reason("MALFORMED_FUNCTION_CALL") == "tool_use"
    assert map_google_finish_reason("UNKNOWN_THING") == "end_turn"  # fallback


# ---------------------------------------------------------------------------
# Fake Google client
# ---------------------------------------------------------------------------


def _part(
    text: str | None = None, function_call: SimpleNamespace | None = None
) -> SimpleNamespace:
    return SimpleNamespace(text=text, function_call=function_call)


def _fc(name: str, args: dict[str, Any], *, id: str | None = None) -> SimpleNamespace:
    """Build a fake ``FunctionCall`` (attribute access, like the real SDK)."""
    return SimpleNamespace(name=name, args=args, id=id)


def _response(
    parts: list[SimpleNamespace],
    *,
    finish: str = "STOP",
    response_id: str = "resp_1",
    model_version: str = "gemini-test",
    prompt_tokens: int = 3,
    candidates_tokens: int = 4,
) -> SimpleNamespace:
    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(
        content=content, finish_reason=SimpleNamespace(name=finish)
    )
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens, candidates_token_count=candidates_tokens
    )
    return SimpleNamespace(
        candidates=[candidate],
        finish_reason=None,
        response_id=response_id,
        model_version=model_version,
        usage_metadata=usage,
    )


class _FakeModels:
    def __init__(self) -> None:
        self.generate_result: Any = None
        self.generate_stream_results: list[Any] = []
        self.generate_kwargs: dict[str, Any] = {}
        self.stream_kwargs: dict[str, Any] = {}

    async def generate_content(self, **kwargs: Any) -> Any:
        self.generate_kwargs = kwargs
        return self.generate_result

    async def generate_content_stream(self, **kwargs: Any) -> _ResponseStream:
        self.stream_kwargs = kwargs
        return _ResponseStream(self.generate_stream_results)


class _ResponseStream:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)

    def __aiter__(self) -> _ResponseStream:
        return self

    async def __anext__(self) -> Any:
        if not self._responses:
            raise StopAsyncIteration
        return self._responses.pop(0)


class _FakeAio:
    def __init__(self) -> None:
        self.models = _FakeModels()


class _FakeClient:
    def __init__(self, **_: object) -> None:
        self.aio = _FakeAio()


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    monkeypatch.setattr("google.genai.Client", lambda **kw: client)
    # The provider builds a Tool/Config from google.genai.types — leave the
    # real types in place; only the Client is faked.


# ---------------------------------------------------------------------------
# Non-stream
# ---------------------------------------------------------------------------


def test_non_stream_translates_response_to_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    _install_fake_client(monkeypatch, client)
    client.aio.models.generate_result = _response(
        [
            _part(text="hello"),
            _part(function_call=_fc("fs.read", {"path": "/a"}, id="fc_1")),
        ],
        finish="STOP",
    )
    provider = GoogleProvider(api_key="g-key", api_base=None)
    msg = asyncio.run(
        provider.amessages(
            model="gemini-test",
            messages=[{"role": "user", "content": [{"type": "text", "text": "x"}]}],
            max_tokens=200,
            system=[{"type": "text", "text": "be nice"}],
            tools=[
                {
                    "name": "fs.read",
                    "description": "d",
                    "input_schema": {"type": "object"},
                }
            ],
        )
    )
    assert msg.id == "resp_1"
    assert msg.model == "gemini-test"
    assert msg.stop_reason == "end_turn"
    assert msg.usage.input_tokens == 3
    assert msg.usage.output_tokens == 4
    assert [type(b).__name__ for b in msg.content] == ["TextBlock", "ToolUseBlock"]
    tu = next(b for b in msg.content if isinstance(b, ToolUseBlock))
    assert tu.id == "fc_1"
    assert tu.name == "fs.read"
    assert tu.input == {"path": "/a"}

    req = client.aio.models.generate_kwargs
    assert req["model"] == "gemini-test"
    # contents were translated
    assert req["contents"][0] == {"role": "user", "parts": [{"text": "x"}]}
    # config carries system_instruction, tools, max_output_tokens
    config = req["config"]
    assert config.system_instruction == "be nice"
    assert config.max_output_tokens == 200
    # tools is a google Tool with a function_declaration carrying the name
    fd = config.tools[0].function_declarations[0]
    assert fd.name == "fs.read"


# ---------------------------------------------------------------------------
# Stream synthesis
# ---------------------------------------------------------------------------


def _run_stream(provider: GoogleProvider, **amessages_kwargs: Any) -> list[Any]:
    async def driver() -> list[Any]:
        stream = await provider.amessages(stream=True, **amessages_kwargs)
        return [ev async for ev in stream]

    return asyncio.run(driver())


def test_stream_text_then_function_call(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    _install_fake_client(monkeypatch, client)
    # Response 1: a text part, no finish. Response 2: function call + finish.
    client.aio.models.generate_stream_results = [
        _response([_part(text="ans")], finish="FINISH_REASON_UNSPECIFIED"),
        _response([_part(function_call=_fc("t", {"x": 1}, id="fc_2"))], finish="STOP"),
    ]
    provider = GoogleProvider(api_key="g", api_base=None)
    events = _run_stream(provider, model="gemini-test", messages=[], max_tokens=10)

    # message_start once
    starts = [e for e in events if isinstance(e, RawMessageStartEvent)]
    assert len(starts) == 1
    assert starts[0].message.model == "gemini-test"

    block_starts = [e for e in events if isinstance(e, RawContentBlockStartEvent)]
    # text block (index 0) + tool block (index 1)
    assert len(block_starts) == 2
    assert isinstance(block_starts[0].content_block, TextBlock)
    assert isinstance(block_starts[1].content_block, ToolUseBlock)
    assert block_starts[1].content_block.id == "fc_2"
    assert block_starts[1].content_block.name == "t"

    # Text deltas carry the text.
    text_deltas = [
        e
        for e in events
        if isinstance(e, RawContentBlockDeltaEvent) and isinstance(e.delta, TextDelta)
    ]
    assert "".join(d.delta.text for d in text_deltas) == "ans"

    # tool_use block has exactly ONE InputJSONDelta carrying the complete args.
    json_deltas = [
        e
        for e in events
        if isinstance(e, RawContentBlockDeltaEvent)
        and isinstance(e.delta, InputJSONDelta)
    ]
    assert len(json_deltas) == 1
    assert json.loads(json_deltas[0].delta.partial_json) == {"x": 1}

    # message_stop exactly once + message_delta carries mapped stop_reason
    stops = [e for e in events if isinstance(e, RawMessageStopEvent)]
    assert len(stops) == 1
    msg_deltas = [e for e in events if isinstance(e, RawMessageDeltaEvent)]
    assert len(msg_deltas) == 1
    assert msg_deltas[0].delta.stop_reason == "end_turn"  # STOP -> end_turn


def test_stream_max_tokens_maps_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    _install_fake_client(monkeypatch, client)
    client.aio.models.generate_stream_results = [
        _response([_part(text="partial")], finish="MAX_TOKENS"),
    ]
    provider = GoogleProvider(api_key="g", api_base=None)
    events = _run_stream(provider, model="m", messages=[], max_tokens=1)
    msg_deltas = [e for e in events if isinstance(e, RawMessageDeltaEvent)]
    assert msg_deltas[-1].delta.stop_reason == "max_tokens"


def test_stream_no_explicit_finish_still_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response stream with no finish_reason still emits message_stop once."""
    client = _FakeClient()
    _install_fake_client(monkeypatch, client)
    # finish_reason unspecified -> our helper sets name="FINISH_REASON_UNSPECIFIED"
    # but the synthesiser only treats None as "no finish"; emulate a stream
    # where the candidate.finish_reason is genuinely absent.
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[_part(text="x")]), finish_reason=None
            )
        ],
        response_id="r",
        model_version="m",
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=2),
    )
    client.aio.models.generate_stream_results = [response]
    provider = GoogleProvider(api_key="g", api_base=None)
    events = _run_stream(provider, model="m", messages=[], max_tokens=1)
    assert len([e for e in events if isinstance(e, RawMessageStopEvent)]) == 1
