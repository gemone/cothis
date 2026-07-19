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
import logging
import re
from pathlib import Path
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
    ThinkingBlock,
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
    _assemble_system,
    _assistant_msg_from_response,
    _coalesce_content,
    _concat_text,
    _finalize_stream_block,
    _init_stream_block,
    _load_agents_md,
    _read_first_matching,
    _read_text,
    _request_messages,
    _sanitize_tool_name,
    _system_param,
    _tool_result_block,
)

if TYPE_CHECKING:
    from cothis.tools import Tool


# --- pure helpers -----------------------------------------------------------


def test_system_param_str_becomes_persona_block_with_cache_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cothis.agent._load_agents_md", lambda: None
    )
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


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("fs.read", "fs_read"),
        ("mcp:context7", "mcp_context7"),
        ("date.calculate", "date_calculate"),
        ("code.lines", "code_lines"),
        ("echo", "echo"),
        ("resolve-library-id", "resolve-library-id"),
    ],
)
def test_sanitize_tool_name_matches_openai_pattern(raw, expected) -> None:
    """cothis dotted/colon names must map onto ``^[a-zA-Z0-9_-]+$``."""
    out = _sanitize_tool_name(raw)
    assert out == expected
    assert re.fullmatch(r"[a-zA-Z0-9_-]+", out)


def test_tool_schemas_emit_sanitized_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Schema name reuses the wire-sanitised ``_tool_map`` key verbatim."""

    class _FakeSchemaTool:
        __name__ = "fs.read"
        __cothis_schema__ = {
            "name": "fs.read",  # own (unsanitised) name — overridden by map key
            "description": "read",
            "input_schema": {"type": "object", "properties": {}},
        }

        def __call__(self, *args: Any, **kw: Any) -> Any: ...

    agent = _patched_agent(monkeypatch)
    agent._tool_map = {"fs_read": _FakeSchemaTool()}  # key sanitised at registration
    schemas = agent._tool_schemas()
    assert schemas is not None
    assert len(schemas) == 1
    assert schemas[0]["name"] == "fs_read"  # wire name matches dispatch key


def test_execute_tool_resolves_sanitised_wire_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch hits the map directly — key was sanitised at registration."""
    agent = _patched_agent(monkeypatch)
    agent._tool_map["fs_read"] = lambda **kw: "ok"  # sanitised key
    is_error, out = asyncio.run(
        agent._execute_tool({"name": "fs_read", "input": {}})
    )
    assert is_error is False
    assert out == "ok"


def test_init_warns_on_sanitised_key_collision(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two names sanitising to the same wire key (``fs.read`` vs ``fs_read``)
    must log a WARNING; last-write-wins matches dict semantics."""
    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    class _ToolA:
        __name__ = "fs.read"

        def __call__(self, *args: Any, **kw: Any) -> Any:
            return "a"

    class _ToolB:
        __name__ = "fs_read"

        def __call__(self, *args: Any, **kw: Any) -> Any:
            return "b"

    tool_a, tool_b = _ToolA(), _ToolB()
    with caplog.at_level(logging.WARNING, logger="cothis.agent"):
        agent = Agent(model="x", provider="openrouter", tools=[tool_a, tool_b])
    assert agent._tool_map == {"fs_read": tool_b}  # last-write-wins
    assert "shadowed" in caplog.text
    assert "fs.read" in caplog.text


def test_execute_tool_error_message_uses_original_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Human-facing error string uses the tool's original ``__name__``
    (``fs.read``), not the wire-sanitised key the model echoed."""

    class _Boom:
        __name__ = "fs.read"

        def __call__(self, **kw: Any) -> str:
            raise ValueError("nope")

    agent = _patched_agent(monkeypatch)
    agent._tool_map["fs_read"] = _Boom()  # sanitised key, original __name__
    is_error, msg = asyncio.run(
        agent._execute_tool({"name": "fs_read", "input": {}})
    )
    assert is_error is True
    assert "fs.read" in msg  # original name
    assert "fs_read" not in msg  # wire name must not leak into diagnostics


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


def test_run_empty_text_warning_carries_stop_reason_and_usage(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The empty-text warning surfaces ``stop_reason`` and ``usage`` so the
    user can diagnose (max-tokens cutoff vs end_turn) without guessing."""
    agent = _patched_agent(monkeypatch)

    async def fake_amessages(**kwargs: Any) -> Any:
        return _msg_response([], stop_reason="end_turn")

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    with caplog.at_level(logging.WARNING, logger="cothis.agent"):
        assert asyncio.run(agent.run("hi")) == ""
    assert "stop_reason" in caplog.text
    assert "end_turn" in caplog.text
    assert "usage" in caplog.text


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
    monkeypatch.setattr("cothis.agent._load_agents_md", lambda: None)
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
    _add = lambda **kw: str(kw["a"] + kw["b"])  # noqa: E731
    _add.__name__ = "add"  # ToolCallEvent.name restores the tool's __name__
    agent._tool_map["add"] = _add

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
async def test_run_stream_tool_call_event_uses_original_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ToolCallEvent.name`` is the tool's original ``__name__`` (``fs.read``),
    not the wire-sanitised name the model echoes back (``fs_read``)."""

    class _Dotted:
        __name__ = "fs.read"

        def __call__(self, **kw: Any) -> str:
            return "ok"

    agent = _patched_agent(monkeypatch)
    agent._tool_map["fs_read"] = _Dotted()  # sanitised key, dotted __name__

    def turn1() -> list[Any]:
        return [
            _message_start("m1"),
            _block_start(
                0, ToolUseBlock(type="tool_use", id="tu1", name="fs_read", input={})
            ),
            _block_delta(
                0, InputJSONDelta(type="input_json_delta", partial_json="{}")
            ),
            _block_stop(0),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]

    def turn2() -> list[Any]:
        return [
            _message_start("m2"),
            _block_start(0, TextBlock(type="text", text="ok")),
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
    assert out[0].name == "fs.read"  # original, not wire-sanitised "fs_read"


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


@pytest.mark.asyncio
async def test_run_stream_coalesces_fragmented_assistant_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION: gpt-oss-120b interleaves reasoning/text per chunk, and
    any-llm's OpenAI→Messages stream converter opens a NEW content block on
    every reasoning→text transition. The result is an assistant message with
    dozens of tiny ``thinking`` fragments interleaved with empty ``text``
    blocks. When that malformed message is replayed on the next turn, some
    providers (openrouter) silently return empty content — observed in
    ``cothis chat`` as the agent finishing a tool call and returning to
    ``>>>`` with no answer.

    ``_coalesce_content`` (applied in ``run_stream`` before storing) merges
    adjacent same-type blocks and drops empty text blocks, restoring the
    canonical block shape so the next-turn request is well-formed.

    This test feeds a fragmented turn-1 stream (three thinking/text
    alternations + a tool_use) and asserts the stored assistant message
    coalesces to one ``thinking`` + one ``text`` + one ``tool_use`` block.
    """
    agent = _patched_agent(monkeypatch)
    agent._tool_map["echo"] = lambda **kw: "ok"

    # Fragmented stream: the converter emits thinking[0] text[1] thinking[2]
    # text[3] thinking[4] text[5] tool_use[6], where text[1]/text[3] are
    # empty (delta.content == "" gets a block start) and text[5] carries
    # the actual content.
    def turn1() -> list[Any]:
        return [
            _message_start("m1"),
            _block_start(0, ThinkingBlock(type="thinking", thinking="", signature="")),
            _block_delta(0, ThinkingDelta(type="thinking_delta", thinking="th")),
            _block_stop(0),
            _block_start(1, TextBlock(type="text", text="")),
            _block_stop(1),  # empty text block (dropped on coalesce)
            _block_start(2, ThinkingBlock(type="thinking", thinking="", signature="")),
            _block_delta(2, ThinkingDelta(type="thinking_delta", thinking="ink")),
            _block_stop(2),
            _block_start(3, TextBlock(type="text", text="")),
            _block_stop(3),  # empty text block (dropped)
            _block_start(4, ThinkingBlock(type="thinking", thinking="", signature="")),
            _block_stop(4),  # empty thinking fragment (kept — non-empty after merge)
            _block_start(5, TextBlock(type="text", text="")),
            _block_delta(5, TextDelta(type="text_delta", text="hello")),
            _block_stop(5),
            _block_start(
                6, ToolUseBlock(type="tool_use", id="tu1", name="echo", input={})
            ),
            _block_stop(6),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]

    state = {"n": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        # Turn 1: fragmented stream. Turn 2+: empty-text final turn.
        state["n"] += 1
        if state["n"] == 1:
            return _stream_from(turn1())
        return _stream_from([
            _message_start("m2"),
            _block_start(0, TextBlock(type="text", text="")),
            _block_delta(0, TextDelta(type="text_delta", text="done")),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    out = await _drain(agent.run_stream("hi"))
    # Stream yielded the text delta from turn 1 + tool event + turn 2 text.
    assert "hello" in out
    assert "done" in out
    assert any(isinstance(e, ToolCallEvent) for e in out)

    # Stored turn-1 assistant message is coalesced: 3 blocks, not 7.
    stored_turn1 = [
        m for m in agent._messages if m["role"] == "assistant"
    ][0]
    types = [b["type"] for b in stored_turn1["content"]]
    assert types == ["thinking", "text", "tool_use"]
    # Thinking fragments merged.
    assert stored_turn1["content"][0]["thinking"] == "think"
    # Text delta carried through.
    assert stored_turn1["content"][1]["text"] == "hello"


# --- _coalesce_content unit tests -------------------------------------------


def test_coalesce_merges_adjacent_text_blocks() -> None:
    out = _coalesce_content([
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "world"},
    ])
    assert out == [{"type": "text", "text": "Hello world"}]


def test_coalesce_merges_adjacent_thinking_blocks() -> None:
    out = _coalesce_content([
        {"type": "thinking", "thinking": "th", "signature": "x"},
        {"type": "thinking", "thinking": "ink", "signature": "y"},
    ])
    # Thinking merged; signatures are not preserved across merges (this slice
    # doesn't pass the thinking param, so Anthropic doesn't validate them).
    assert out == [{"type": "thinking", "thinking": "think", "signature": "x"}]


def test_coalesce_drops_empty_text_blocks() -> None:
    out = _coalesce_content([
        {"type": "text", "text": ""},  # dropped
        {"type": "text", "text": "   "},  # whitespace-only, dropped
        {"type": "text", "text": "real"},
    ])
    assert out == [{"type": "text", "text": "real"}]


def test_coalesce_preserves_tool_use_blocks() -> None:
    blocks = [
        {"type": "text", "text": "ok"},
        {"type": "tool_use", "id": "tu1", "name": "echo", "input": {}},
        {"type": "tool_use", "id": "tu2", "name": "echo", "input": {}},
    ]
    out = _coalesce_content(blocks)
    # Two tool_use blocks are NOT merged into each other (each is a distinct call).
    assert [b["type"] for b in out] == ["text", "tool_use", "tool_use"]
    assert out[1]["id"] == "tu1"
    assert out[2]["id"] == "tu2"


def test_coalesce_does_not_mutate_input() -> None:
    original = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    snapshot = [dict(b) for b in original]
    _coalesce_content(original)
    assert original == snapshot


def test_coalesce_alternating_thinking_then_text_stays_separate() -> None:
    """Real shape from gpt-oss: thinking, text, thinking, text — coalesce
    keeps each as a separate block (they alternate, not adjacent)."""
    out = _coalesce_content([
        {"type": "thinking", "thinking": "hm"},
        {"type": "text", "text": "answer"},
        {"type": "thinking", "thinking": "more"},
        {"type": "text", "text": " continued"},
    ])
    assert [b["type"] for b in out] == ["thinking", "text", "thinking", "text"]
    assert out[1]["text"] == "answer"
    assert out[3]["text"] == " continued"


# --- system prompt assembly (#33) -------------------------------------------


def test_assemble_system_persona_block_with_cache_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cothis.agent._load_agents_md", lambda: None)
    blocks = _assemble_system("You are helpful.")
    assert len(blocks) == 1
    assert blocks[0] == {
        "type": "text",
        "text": "You are helpful.",
        "cache_control": {"type": "ephemeral"},
    }


def test_assemble_system_includes_agents_md_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cothis.agent._load_agents_md", lambda: '<agents_md type="project">\nrules\n</agents_md>'
    )
    blocks = _assemble_system("persona")
    assert len(blocks) == 2
    assert blocks[0]["text"] == "persona"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["text"] == '<agents_md type="project">\nrules\n</agents_md>'
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}


def test_assemble_system_omits_agents_md_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cothis.agent._load_agents_md", lambda: None)
    blocks = _assemble_system("persona")
    assert len(blocks) == 1


def test_load_agents_md_three_layer_concat_xml_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 layers → concatenated, XML-tagged in order."""
    home = tmp_path / "home"
    agents_dir = home / ".agents"
    cothis_home = tmp_path / "cothis_home"
    project = tmp_path / "project"
    for d in (agents_dir, cothis_home, project):
        d.mkdir(parents=True, exist_ok=True)

    (agents_dir / "AGENTS.md").write_text("rule from agents")
    (cothis_home / "AGENTS.md").write_text("rule from cothis")
    (project / "AGENTS.md").write_text("rule from project")

    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setenv("COTHIS_HOME", str(cothis_home))
    monkeypatch.setattr("pathlib.Path.cwd", lambda: project)

    result = _load_agents_md()
    assert result is not None
    assert '<agents_md type="user-agents">' in result
    assert '<agents_md type="user-cothis">' in result
    assert '<agents_md type="project">' in result
    assert "rule from agents" in result
    assert "rule from cothis" in result
    assert "rule from project" in result
    # Order: user-agents comes first
    idx_agents = result.index("user-agents")
    idx_cothis = result.index("user-cothis")
    idx_project = result.index("project")
    assert idx_agents < idx_cothis < idx_project


def test_load_agents_md_omits_when_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cothis_home = tmp_path / "cothis_home"
    cothis_home.mkdir()
    project = tmp_path / "project"
    project.mkdir()

    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setenv("COTHIS_HOME", str(cothis_home))
    monkeypatch.setattr("pathlib.Path.cwd", lambda: project)

    assert _load_agents_md() is None


def test_load_agents_md_custom_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "CODING.md").write_text("coding rules")

    monkeypatch.setattr("pathlib.Path.home", lambda: Path("/nonexistent"))
    monkeypatch.setenv("COTHIS_HOME", "/nonexistent")
    monkeypatch.setenv("COTHIS_AGENTS_PATTERN", "CODING.md,RULES.md")
    monkeypatch.setenv("COTHIS_AGENTS_USER_GLOBAL", "0")
    monkeypatch.setattr("pathlib.Path.cwd", lambda: project)

    result = _load_agents_md()
    assert result is not None
    assert "coding rules" in result
    assert "CODING.md" not in result  # filename not in output, only content


def test_load_agents_md_custom_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    cothis_home = tmp_path / "cothis_home"
    for d in (project, cothis_home):
        d.mkdir()

    (project / "AGENTS.md").write_text("project rules")
    (cothis_home / "AGENTS.md").write_text("cothis rules")

    monkeypatch.setattr("pathlib.Path.home", lambda: Path("/nonexistent"))
    monkeypatch.setenv("COTHIS_HOME", str(cothis_home))
    monkeypatch.setenv("COTHIS_AGENTS_ORDER", "project,user-cothis")
    monkeypatch.setattr("pathlib.Path.cwd", lambda: project)

    result = _load_agents_md()
    assert result is not None
    idx_project = result.index("project")
    idx_cothis = result.index("user-cothis")
    assert idx_project < idx_cothis  # project comes first


def test_load_agents_md_user_global_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents_dir = tmp_path / "home" / ".agents"
    cothis_home = tmp_path / "cothis_home"
    project = tmp_path / "project"
    for d in (agents_dir, cothis_home, project):
        d.mkdir(parents=True, exist_ok=True)

    (agents_dir / "AGENTS.md").write_text("agents rule")
    (cothis_home / "AGENTS.md").write_text("cothis rule")
    (project / "AGENTS.md").write_text("project rule")

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    monkeypatch.setenv("COTHIS_HOME", str(cothis_home))
    monkeypatch.setenv("COTHIS_AGENTS_USER_GLOBAL", "0")
    monkeypatch.setattr("pathlib.Path.cwd", lambda: project)

    result = _load_agents_md()
    assert result is not None
    # Only project layer should be present
    assert "user-agents" not in result
    assert "user-cothis" not in result
    assert "project" in result
    assert "project rule" in result


def test_load_agents_md_skips_empty_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text("   \n  ")  # whitespace only

    monkeypatch.setattr("pathlib.Path.home", lambda: Path("/nonexistent"))
    monkeypatch.setenv("COTHIS_HOME", "/nonexistent")
    monkeypatch.setenv("COTHIS_AGENTS_USER_GLOBAL", "0")
    monkeypatch.setattr("pathlib.Path.cwd", lambda: project)

    assert _load_agents_md() is None


def test_read_first_matching_first_match_wins(
    tmp_path: Path,
) -> None:
    (tmp_path / "A.md").write_text("first")
    (tmp_path / "B.md").write_text("second")
    result = _read_first_matching(tmp_path, ["A.md", "B.md"])
    assert result == "first"


def test_read_first_matching_falls_back_to_second(
    tmp_path: Path,
) -> None:
    (tmp_path / "B.md").write_text("second")
    result = _read_first_matching(tmp_path, ["A.md", "B.md"])
    assert result == "second"


def test_read_first_matching_returns_none_when_no_match(
    tmp_path: Path,
) -> None:
    result = _read_first_matching(tmp_path, ["NOPE.md"])
    assert result is None


def test_read_first_matching_skips_empty_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "A.md").write_text("   \n  ")
    result = _read_first_matching(tmp_path, ["A.md"])
    assert result is None


def test_read_text_decodes_utf8(tmp_path: Path) -> None:
    f = tmp_path / "A.md"
    f.write_text("héllo 中文", encoding="utf-8")
    assert _read_text(f, "gbk") == "héllo 中文"


def test_read_text_falls_back_to_locale_encoding(tmp_path: Path) -> None:
    # GBK bytes that are NOT valid UTF-8 — exercises the fallback tier.
    f = tmp_path / "A.md"
    f.write_bytes("你好".encode("gbk"))
    assert _read_text(f, "gbk") == "你好"


def test_read_text_returns_none_when_neither_decodes(tmp_path: Path) -> None:
    # 0x80 is invalid in both UTF-8 and ASCII — confirms we skip (return
    # None) rather than raise or lossy-replace with U+FFFD.
    f = tmp_path / "A.md"
    f.write_bytes(b"\x80\xff")
    assert _read_text(f, "ascii") is None


def test_read_first_matching_reads_gbk_via_locale_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End-to-end: a non-UTF-8 file is readable when the locale fallback
    # matches its encoding. Locks in that bad bytes never crash the run.
    import locale as _locale

    (tmp_path / "AGENTS.md").write_bytes("规则".encode("gbk"))
    monkeypatch.setattr(_locale, "getpreferredencoding", lambda _x: "gbk")
    assert _read_first_matching(tmp_path, ["AGENTS.md"]) == "规则"


def test_system_param_str_calls_assembler_and_includes_agents_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: _system_param(str) → assembled blocks with AGENTS.md."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text("project rules")

    monkeypatch.setattr("pathlib.Path.home", lambda: Path("/nonexistent"))
    monkeypatch.setenv("COTHIS_HOME", "/nonexistent")
    monkeypatch.setenv("COTHIS_AGENTS_USER_GLOBAL", "0")
    monkeypatch.setattr("pathlib.Path.cwd", lambda: project)

    blocks = _system_param("be brief")
    assert blocks is not None
    assert len(blocks) == 2
    assert blocks[0] == {
        "type": "text",
        "text": "be brief",
        "cache_control": {"type": "ephemeral"},
    }
    assert "project rules" in blocks[1]["text"]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}


# --- regression: system prompt assembled once per run (#33 review) ----------


def test_run_assembles_system_prompt_once_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A str persona must snapshot the system prompt, not re-read per turn.

    Without the hoist, ``_load_agents_md`` runs every iteration → disk I/O
    each turn and the persona/rules can drift if ``./AGENTS.md`` changes
    mid-run (e.g. via a file tool). This test fails if the snapshot is
    moved back inside the loop.
    """
    agent = _patched_agent(monkeypatch)
    agent.system = "be brief"
    agent._tool_map["echo"] = lambda **kw: "ok"
    state = {"turn": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        state["turn"] += 1
        if state["turn"] == 1:
            return _msg_response(
                [ToolUseBlock(type="tool_use", id="tu1", name="echo", input={})],
                stop_reason="tool_use",
            )
        return _msg_response([TextBlock(type="text", text="done")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)

    call_count = {"n": 0}

    def counting_loader() -> None:
        call_count["n"] += 1
        return None

    monkeypatch.setattr("cothis.agent._load_agents_md", counting_loader)
    asyncio.run(agent.run("hi"))
    # Two turns (tool call then final answer) → exactly one assembly.
    assert state["turn"] == 2
    assert call_count["n"] == 1


# --- regression: COTHIS_HOME="" resolves to ~/.cothis, not cwd (#33 review) -


def test_load_agents_md_empty_cothis_home_does_not_read_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``COTHIS_HOME=""`` must behave like cli.py: unset → ``~/.cothis``.

    Without mirroring cli.py's resolution, the empty string became
    ``Path('.')`` (cwd), so the project file was double-read under a
    misleading ``<agents_md type="user-cothis">`` tag.
    """
    home = tmp_path / "home"
    cothis_default = home / ".cothis"
    project = tmp_path / "project"
    for d in (home, cothis_default, project):
        d.mkdir(parents=True, exist_ok=True)

    # Both layers present; the bug would tag project content as user-cothis.
    (cothis_default / "AGENTS.md").write_text("real cothis rules")
    (project / "AGENTS.md").write_text("project rules")

    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setenv("COTHIS_HOME", "")  # set-but-empty → must fall back
    monkeypatch.setattr("pathlib.Path.cwd", lambda: project)

    result = _load_agents_md()
    assert result is not None
    # user-cothis layer must carry the ~/.cothis content, not the project's.
    assert "real cothis rules" in result
    # project layer still present and correctly tagged.
    assert '<agents_md type="project">' in result
    # And user-cothis did NOT accidentally read the project file.
    cothis_block = result.split('<agents_md type="user-cothis">')[1].split(
        "</agents_md>"
    )[0]
    assert "project rules" not in cothis_block
