"""Tests for ``cothis.agent`` — Anthropic Messages API wire format.

Covers the pure message-assembly helpers and the streaming accumulator (the
silent-breakage surfaces of the agent loop):

- ``_system_param`` / ``_tool_result_block`` / ``_concat_text`` /
  ``_request_messages`` — Anthropic block-list construction + projection.
- ``_assistant_msg_from_response`` — MessageResponse → stored dict.
- ``_init_stream_block`` / ``_apply_stream_delta`` / ``_finalize_stream_block``
  — the streaming accumulator. If these drift, tool arguments corrupt or
  text fragments drop.
- ``run`` — stop_reason turn decision (tool_use vs final answer).
- ``_execute_tool`` — tool dispatch returning ``(is_error, output)``.
- ``run_stream`` — consumes a synthetic ``MessageStreamEvent`` flow.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    SignatureDelta,
    ThinkingDelta,
)
from anthropic.types.message import Message
from anthropic.types.raw_message_delta_event import Delta
from any_llm.types.messages import (
    InputJSONDelta,
    MessageDeltaUsage,
    MessageResponse,
    MessageUsage,
    StopReason,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)
from pydantic import ValidationError

from cothis.agent import (
    Agent,
    MaxIterationsError,
    ToolCallEvent,
    _apply_stream_delta,
    _assistant_msg_from_response,
    _concat_text,
    _finalize_stream_block,
    _init_stream_block,
    _request_messages,
    _system_param,
    _tool_result_block,
)

if TYPE_CHECKING:
    from cothis.tools import Tool


# --- pure helpers -----------------------------------------------------------


def test_system_param_str_becomes_persona_block_with_cache_control() -> None:
    assert _system_param("You are helpful.") == [
        {
            "type": "text",
            "text": "You are helpful.",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_system_param_none_stays_none() -> None:
    assert _system_param(None) is None


def test_system_param_list_passed_through_verbatim() -> None:
    blocks = [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]
    # caller owns cache_control placement; no copy, no mutation.
    assert _system_param(blocks) is blocks


def test_tool_result_block_normal_omits_is_error() -> None:
    b = _tool_result_block("tu_1", "ok", is_error=False)
    assert b == {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}
    assert "is_error" not in b


def test_tool_result_block_error_sets_is_error() -> None:
    b = _tool_result_block("tu_1", "Error: boom", is_error=True)
    assert b["is_error"] is True
    assert b["content"] == "Error: boom"


def test_concat_text_joins_text_blocks_skips_others() -> None:
    content = [
        {"type": "thinking", "thinking": "..."},
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "world"},
    ]
    assert _concat_text(content) == "Hello world"


def test_concat_text_empty_when_no_text_blocks() -> None:
    assert _concat_text([{"type": "thinking", "thinking": "x"}]) == ""


def test_request_messages_strips_assistant_metadata() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "yo"}],
            "id": "m1",
            "model": "x",
            "stop_reason": "end_turn",
            "usage": {"output_tokens": 1},
        },
    ]
    assert _request_messages(messages) == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
    ]


# --- _assistant_msg_from_response ------------------------------------------


def _msg_response(
    content: list[Any], stop_reason: StopReason = "end_turn"
) -> MessageResponse:
    return MessageResponse(
        id="msg_1",
        model="test-model",
        role="assistant",
        type="message",
        content=content,
        stop_reason=stop_reason,
        usage=MessageUsage(input_tokens=3, output_tokens=4),
    )


def test_assistant_msg_from_response_carries_metadata() -> None:
    msg = _assistant_msg_from_response(
        _msg_response([TextBlock(type="text", text="answer")])
    )
    assert msg["role"] == "assistant"
    assert msg["content"] == [{"type": "text", "text": "answer"}]
    assert msg["id"] == "msg_1"
    assert msg["model"] == "test-model"
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"]["output_tokens"] == 4


def test_assistant_msg_from_response_tool_use_dumps_input() -> None:
    resp = _msg_response(
        [ToolUseBlock(type="tool_use", id="tu1", name="fs.read", input={"path": "/x"})],
        stop_reason="tool_use",
    )
    msg = _assistant_msg_from_response(resp)
    assert msg["stop_reason"] == "tool_use"
    assert msg["content"][0] == {
        "type": "tool_use",
        "id": "tu1",
        "name": "fs.read",
        "input": {"path": "/x"},
    }


# --- streaming accumulator --------------------------------------------------


def test_init_stream_block_seeds_tool_use_with_input_accumulator() -> None:
    block = _init_stream_block(
        ToolUseBlock(type="tool_use", id="tu1", name="add", input={})
    )
    assert block["type"] == "tool_use"
    assert block["_input_json"] == ""


def test_apply_delta_text_appends() -> None:
    block = {"type": "text", "text": ""}
    _apply_stream_delta(block, TextDelta(type="text_delta", text="Hel"))
    _apply_stream_delta(block, TextDelta(type="text_delta", text="lo"))
    assert block["text"] == "Hello"


def test_apply_delta_input_json_accumulates_then_finalize_parses() -> None:
    block = {
        "type": "tool_use",
        "id": "tu1",
        "name": "add",
        "input": {},
        "_input_json": "",
    }
    _apply_stream_delta(
        block, InputJSONDelta(type="input_json_delta", partial_json='{"a"')
    )
    _apply_stream_delta(
        block, InputJSONDelta(type="input_json_delta", partial_json=":2}")
    )
    _finalize_stream_block(block)
    assert block["input"] == {"a": 2}
    assert "_input_json" not in block


def test_apply_delta_signature_overwrites_not_appends() -> None:
    block = {"type": "thinking", "thinking": "", "signature": ""}
    _apply_stream_delta(
        block, SignatureDelta(type="signature_delta", signature="sig-A")
    )
    _apply_stream_delta(
        block, SignatureDelta(type="signature_delta", signature="sig-B")
    )
    # overwrite: signature carries the block's final value, not a concat.
    assert block["signature"] == "sig-B"


def test_apply_delta_thinking_appends() -> None:
    block = {"type": "thinking", "thinking": "", "signature": ""}
    _apply_stream_delta(block, ThinkingDelta(type="thinking_delta", thinking="hm"))
    _apply_stream_delta(
        block, ThinkingDelta(type="thinking_delta", thinking=" ...")
    )
    assert block["thinking"] == "hm ..."


def test_finalize_stream_block_malformed_json_falls_back_to_empty() -> None:
    block = {
        "type": "tool_use",
        "id": "tu1",
        "name": "x",
        "input": {},
        "_input_json": "{bad",
    }
    _finalize_stream_block(block)
    assert block["input"] == {}


def test_finalize_stream_block_noop_for_text() -> None:
    block = {"type": "text", "text": "hi"}
    _finalize_stream_block(block)  # must not raise or mutate
    assert block == {"type": "text", "text": "hi"}


# --- run() turn decision ----------------------------------------------------


def _patched_agent(monkeypatch: pytest.MonkeyPatch) -> Agent:
    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    return Agent(model="x", provider="openrouter", tools=[], max_iterations=5)


def test_run_final_answer_returned_on_end_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)

    async def fake_amessages(**kwargs: Any) -> Any:
        return _msg_response([TextBlock(type="text", text="done")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    assert asyncio.run(agent.run("hi")) == "done"


def test_run_executes_tools_on_tool_use_then_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)
    agent._tool_map["echo"] = lambda **kw: kw["msg"]
    state = {"turn": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        state["turn"] += 1
        if state["turn"] == 1:
            return _msg_response(
                [
                    ToolUseBlock(
                        type="tool_use", id="tu1", name="echo", input={"msg": "hi"}
                    )
                ],
                stop_reason="tool_use",
            )
        return _msg_response([TextBlock(type="text", text="final")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    assert asyncio.run(agent.run("hi")) == "final"
    # the tool turn enqueued a user message carrying a tool_result block
    # (it sits before the final assistant answer in the history).
    tool_result_msgs = [
        m
        for m in agent._messages
        if m["role"] == "user"
        and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert tool_result_msgs
    assert tool_result_msgs[-1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "tu1",
        "content": "hi",
    }


def test_run_returns_empty_on_end_turn_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)

    async def fake_amessages(**kwargs: Any) -> Any:
        return _msg_response([], stop_reason="end_turn")

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    assert asyncio.run(agent.run("hi")) == ""


def test_run_raises_max_iterations_on_persistent_tool_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)
    agent._tool_map["echo"] = lambda **kw: "ok"

    async def fake_amessages(**kwargs: Any) -> Any:
        return _msg_response(
            [ToolUseBlock(type="tool_use", id="tu1", name="echo", input={})],
            stop_reason="tool_use",
        )

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    with pytest.raises(MaxIterationsError):
        asyncio.run(agent.run("hi"))


def test_run_sends_system_param_and_anthropic_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)
    agent.system = "be brief"
    seen: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _msg_response([TextBlock(type="text", text="ok")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    asyncio.run(agent.run("hi"))
    # system is a block list with cache_control, not a {role: system} message.
    assert seen["system"] == [
        {
            "type": "text",
            "text": "be brief",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    # messages carry only {role, content}; no system role message present.
    assert all("role" in m and "content" in m for m in seen["messages"])
    assert seen["messages"][0]["role"] == "user"
    assert seen["messages"][0]["content"] == [{"type": "text", "text": "hi"}]


# --- _execute_tool ----------------------------------------------------------


def test_execute_tool_str_result_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)
    agent._tool_map["str_tool"] = lambda **kw: "plain text"
    is_error, out = asyncio.run(
        agent._execute_tool({"name": "str_tool", "input": {}})
    )
    assert (is_error, out) == (False, "plain text")


def test_execute_tool_dict_result_serialised_as_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)
    payload = {"name": "src", "type": "dir"}
    agent._tool_map["dict_tool"] = lambda **kw: payload
    is_error, out = asyncio.run(
        agent._execute_tool({"name": "dict_tool", "input": {}})
    )
    assert is_error is False
    assert json.loads(out) == payload
    assert "'name'" not in out  # JSON double-quoted, not python repr


def test_execute_tool_unknown_tool_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)
    is_error, out = asyncio.run(agent._execute_tool({"name": "nope", "input": {}}))
    assert is_error is True
    assert "unknown tool" in out


def test_execute_tool_exception_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)

    def boom(**kw: Any) -> Any:
        raise ValueError("kaboom")

    agent._tool_map["boom"] = boom
    is_error, out = asyncio.run(agent._execute_tool({"name": "boom", "input": {}}))
    assert is_error is True
    assert "kaboom" in out


def test_execute_tool_async_tool_coroutine_awaited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)

    async def async_echo(**kw: Any) -> str:
        return "from coroutine"

    agent._tool_map["async_tool"] = async_echo
    is_error, out = asyncio.run(
        agent._execute_tool({"name": "async_tool", "input": {}})
    )
    assert (is_error, out) == (False, "from coroutine")


# --- rejects non-tool (retained) -------------------------------------------


def test_agent_rejects_non_tool() -> None:
    not_a_tool = cast("Tool", object())
    with pytest.raises(ValidationError):
        Agent(model="x", provider="mistral", tools=[not_a_tool])


def test_tool_schemas_builds_anthropic_schema_for_bare_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare callable (no ``__cothis_schema__``) gets an Anthropic-shape dict.

    ``amessages`` validates ``tools: list[dict]``; returning a raw callable
    (the old fall-through) would ``ValidationError`` at send time.
    """
    agent = _patched_agent(monkeypatch)

    def echo(msg: str) -> str:
        """Echo the message.

        Args:
            msg: The message to echo.
        """
        return msg

    agent._tool_map["echo"] = echo
    schemas = agent._tool_schemas()
    assert schemas is not None  # _tool_map populated above
    schema = schemas[0]
    assert schema["name"] == "echo"
    assert schema["description"] == "Echo the message."
    assert schema["input_schema"]["properties"]["msg"]["description"] == (
        "The message to echo."
    )
    assert schema["input_schema"]["required"] == ["msg"]


# --- run(): max_tokens cutoff mid-tool-call ---------------------------------


def test_run_max_tokens_mid_tool_call_executes_tool_not_poisons_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_tokens`` cut off mid-tool-call: ``tool_use`` present but
    ``stop_reason == "max_tokens"``.

    Must still execute the tool so the session stays valid (every
    ``tool_use`` paired with a ``tool_result``). Returning here would store
    an unpaired ``tool_use`` that 400s the next turn — permanently breaking
    a ``chat`` session.
    """
    agent = _patched_agent(monkeypatch)
    agent._tool_map["echo"] = lambda **kw: "ok"
    state = {"turn": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        state["turn"] += 1
        if state["turn"] == 1:
            return _msg_response(
                [ToolUseBlock(type="tool_use", id="tu1", name="echo", input={})],
                stop_reason="max_tokens",
            )
        return _msg_response([TextBlock(type="text", text="recovered")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    result = asyncio.run(agent.run("hi"))
    assert result == "recovered"
    # The partial tool_use was executed and paired with a tool_result.
    tool_result_msgs = [
        m
        for m in agent._messages
        if m["role"] == "user"
        and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert tool_result_msgs
    assert tool_result_msgs[-1]["content"][0]["tool_use_id"] == "tu1"




# --- run_stream -------------------------------------------------------------
#
# Stream fixtures use the real anthropic SDK ``Raw*Event`` pydantic models
# (any-llm re-exports them as ``MessageStreamEvent``): ``run_stream`` narrows
# the union by ``isinstance``, so SimpleNamespace envelopes would never match.


def _message_start(msg_id: str = "m1") -> RawMessageStartEvent:
    return RawMessageStartEvent(
        type="message_start",
        message=Message(
            id=msg_id,
            model="test-model",
            role="assistant",
            content=[],
            type="message",
            stop_reason=None,
            usage=MessageUsage(input_tokens=1, output_tokens=0),
        ),
    )


def _delta(stop_reason: StopReason) -> Delta:
    return Delta(stop_reason=stop_reason)


def _msg_delta(stop_reason: StopReason) -> RawMessageDeltaEvent:
    return RawMessageDeltaEvent(
        type="message_delta",
        delta=_delta(stop_reason),
        usage=MessageDeltaUsage(output_tokens=2, input_tokens=1),
    )


def _msg_stop() -> RawMessageStopEvent:
    return RawMessageStopEvent(type="message_stop")


def _block_start(index: int, block: Any) -> RawContentBlockStartEvent:
    return RawContentBlockStartEvent(
        type="content_block_start", index=index, content_block=block
    )


def _block_delta(index: int, delta: Any) -> RawContentBlockDeltaEvent:
    return RawContentBlockDeltaEvent(
        type="content_block_delta", index=index, delta=delta
    )


def _block_stop(index: int) -> RawContentBlockStopEvent:
    return RawContentBlockStopEvent(type="content_block_stop", index=index)


async def _drain(gen: Any) -> list[Any]:
    out: list[Any] = []
    async for ev in gen:
        out.append(ev)
    return out


def _stream_from(events: list[Any]) -> Any:
    async def gen() -> Any:
        for e in events:
            yield e

    return gen()


@pytest.mark.asyncio
async def test_run_stream_yields_text_deltas_and_stores_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)
    events = [
        _message_start(),
        _block_start(0, TextBlock(type="text", text="")),
        _block_delta(0, TextDelta(type="text_delta", text="He")),
        _block_delta(0, TextDelta(type="text_delta", text="llo")),
        _block_stop(0),
        _msg_delta("end_turn"),
        _msg_stop(),
    ]
    async def fake_amessages(**kwargs: Any) -> Any:
        return _stream_from(events)

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    out = await _drain(agent.run_stream("hi"))
    assert out == ["He", "llo"]
    stored = agent._messages[-1]
    assert stored["role"] == "assistant"
    assert stored["content"][0]["text"] == "Hello"
    assert stored["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_run_stream_tool_turn_yields_event_then_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)
    agent._tool_map["add"] = lambda **kw: str(kw["a"] + kw["b"])

    def turn1() -> list[Any]:
        return [
            _message_start("m1"),
            _block_start(
                0, ToolUseBlock(type="tool_use", id="tu1", name="add", input={})
            ),
            _block_delta(
                0, InputJSONDelta(type="input_json_delta", partial_json='{"a":2,"b":3}')
            ),
            _block_stop(0),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]

    def turn2() -> list[Any]:
        return [
            _message_start("m2"),
            _block_start(0, TextBlock(type="text", text="")),
            _block_delta(0, TextDelta(type="text_delta", text="5")),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]

    turn = {"i": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        turn["i"] += 1
        return _stream_from(turn1() if turn["i"] == 1 else turn2())

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    out = await _drain(agent.run_stream("hi"))
    assert isinstance(out[0], ToolCallEvent)
    assert out[0].name == "add"
    assert out[0].arguments == {"a": 2, "b": 3}
    assert out[1] == "5"


@pytest.mark.asyncio
async def test_run_stream_max_tokens_mid_tool_call_still_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_tokens`` mid-tool-call in the stream path: ``tool_use`` block
    present but ``stop_reason == "max_tokens"``. Must still execute the tool
    so the session stays valid (pairing invariant)."""
    agent = _patched_agent(monkeypatch)
    agent._tool_map["add"] = lambda **kw: "ok"

    def turn1() -> list[Any]:
        return [
            _message_start("m1"),
            _block_start(
                0, ToolUseBlock(type="tool_use", id="tu1", name="add", input={})
            ),
            _block_stop(0),
            _msg_delta("max_tokens"),  # cut off, but tool_use block is present
            _msg_stop(),
        ]

    def turn2() -> list[Any]:
        return [
            _message_start("m2"),
            _block_start(0, TextBlock(type="text", text="")),
            _block_delta(0, TextDelta(type="text_delta", text="recovered")),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]

    turn = {"i": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        turn["i"] += 1
        return _stream_from(turn1() if turn["i"] == 1 else turn2())

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    out = await _drain(agent.run_stream("hi"))
    assert isinstance(out[0], ToolCallEvent)
    assert out[1] == "recovered"
    # tool_use paired with tool_result — session valid.
    tool_result_msgs = [
        m
        for m in agent._messages
        if m["role"] == "user"
        and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert tool_result_msgs


@pytest.mark.asyncio
async def test_run_stream_latches_on_first_message_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _patched_agent(monkeypatch)
    events = [
        _message_start(),
        _block_start(0, TextBlock(type="text", text="")),
        _block_delta(0, TextDelta(type="text_delta", text="x")),
        _block_stop(0),
        _msg_delta("end_turn"),
        _msg_stop(),
        _msg_stop(),  # duplicate (openrouter emits these)
    ]
    async def fake_amessages(**kwargs: Any) -> Any:
        return _stream_from(events)

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    out = await _drain(agent.run_stream("hi"))
    assert out == ["x"]  # duplicate stop did not error or re-process


@pytest.mark.asyncio
async def test_run_stream_tolerates_empty_text_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """content_block_start → content_block_stop with no delta (empty block)."""
    agent = _patched_agent(monkeypatch)
    events = [
        _message_start(),
        _block_start(0, TextBlock(type="text", text="")),
        _block_stop(0),  # no delta in between
        _msg_delta("end_turn"),
        _msg_stop(),
    ]
    async def fake_amessages(**kwargs: Any) -> Any:
        return _stream_from(events)

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    out = await _drain(agent.run_stream("hi"))
    assert out == []  # no text deltas, but no crash
