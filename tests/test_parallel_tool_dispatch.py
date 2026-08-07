"""Tests for ordered-concurrency tool dispatch (#316 / iteration I30).

``Agent._dispatch_tool_uses`` replaces the two serial ``for block in
tool_uses: await self._execute_tool(block)`` loops (batch ``run`` and
streaming ``run_stream``) with one site that:

* runs non-skill-marker blocks concurrently via
  ``asyncio.gather(return_exceptions=True)`` (N independent tools cost
  roughly the MAX of their wall-clocks, not the SUM);
* serialises skill-marker blocks (``load_skill`` / ``deactivate_skill``)
  through a shared ``asyncio.Lock`` so the ``Session._active_skills``
  check-then-mutate sequences can't race;
* preserves ORIGINAL block order for results and streamed events;
* maps any exception to ``(True, "Error calling ...")`` so one failing
  tool never cancels its siblings.

These tests cover the latency win, the order-preservation guarantee, the
one-fails-isolation guarantee, skill-marker serialisation determinism,
and the single-tool golden event sequence.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from anthropic.types import (
    InputJSONDelta,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    StopReason,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)
from anthropic.types import (
    Message as MessageResponse,
)
from anthropic.types import (
    Usage as MessageUsage,
)
from anthropic.types.message import Message
from anthropic.types.raw_message_delta_event import Delta

from cothis.agent import (
    Agent,
    ContentDelta,
    ToolCallEvent,
    ToolResultEvent,
)
from cothis.tools import tool

if TYPE_CHECKING:
    from pathlib import Path

    from cothis.session import Session


# --- shared fixtures -------------------------------------------------------


def _patched_agent(monkeypatch: pytest.MonkeyPatch) -> Agent:
    """Agent with a mocked provider — no real LLM calls."""
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **kw: MagicMock())
    return Agent(model="x", provider="openrouter", tools=[], max_iterations=5)


def _msg_response(
    content: list[Any], stop_reason: StopReason = "end_turn"
) -> Any:
    return MessageResponse(
        id="msg_1",
        model="test-model",
        role="assistant",
        type="message",
        content=content,
        stop_reason=stop_reason,
        usage=MessageUsage(input_tokens=3, output_tokens=4),
    )


# --- batch path (Agent.run) ------------------------------------------------


def test_concurrent_dispatch_overlaps_independent_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two independent async tools in one turn run concurrently, not serially.

    Deterministic overlap probe (no wall-clock timing, so no CI jitter):
    each tool increments a shared ``in_flight`` counter on entry and
    records the high-water mark before yielding. Under serial dispatch the
    mark stays 1; concurrent dispatch pushes it to 2 because both tools
    are awaiting at once.
    """
    state = {"in_flight": 0, "max_in_flight": 0}

    async def slow(**kw: Any) -> str:
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        try:
            await asyncio.sleep(0.05)  # window that forces overlap
            return "done"
        finally:
            state["in_flight"] -= 1

    agent = _patched_agent(monkeypatch)
    agent._tool_map["slow_a"] = slow
    agent._tool_map["slow_b"] = slow

    turn = {"i": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        turn["i"] += 1
        if turn["i"] == 1:
            return _msg_response(
                [
                    ToolUseBlock(
                        type="tool_use", id="tu1", name="slow_a", input={}
                    ),
                    ToolUseBlock(
                        type="tool_use", id="tu2", name="slow_b", input={}
                    ),
                ],
                stop_reason="tool_use",
            )
        return _msg_response([TextBlock(type="text", text="final")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    answer = asyncio.run(agent.run("hi"))

    assert answer == "final"
    # Both tools were in flight at once — concurrent dispatch overlapped
    # them rather than queueing serially.
    assert state["max_in_flight"] == 2, (
        f"expected both tools to overlap (max_in_flight=2), got "
        f"{state['max_in_flight']} — tools ran serially"
    )


def test_result_order_preserved_when_later_tool_finishes_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tool_result`` blocks land in ORIGINAL block order in ``_messages``.

    The first-scheduled tool sleeps LONGER than the second, so under
    concurrent dispatch the second completes first. Results must still
    merge in original block order (tu1 before tu2), not completion
    order.
    """

    async def long_tool(**kw: Any) -> str:
        await asyncio.sleep(0.08)
        return "from-long"

    async def short_tool(**kw: Any) -> str:
        await asyncio.sleep(0.005)
        return "from-short"

    agent = _patched_agent(monkeypatch)
    agent._tool_map["long_tool"] = long_tool
    agent._tool_map["short_tool"] = short_tool

    state = {"turn": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        state["turn"] += 1
        if state["turn"] == 1:
            return _msg_response(
                [
                    ToolUseBlock(
                        type="tool_use", id="tu1", name="long_tool", input={}
                    ),
                    ToolUseBlock(
                        type="tool_use", id="tu2", name="short_tool", input={}
                    ),
                ],
                stop_reason="tool_use",
            )
        return _msg_response([TextBlock(type="text", text="done")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    asyncio.run(agent.run("hi"))

    tool_result_msg = next(
        m
        for m in agent._messages
        if m["role"] == "user"
        and any(b.get("type") == "tool_result" for b in m["content"])
    )
    results = [b for b in tool_result_msg["content"] if b.get("type") == "tool_result"]
    # Original block order preserved (long_tool=tu1 first, short_tool=tu2
    # second) even though short_tool finished first under the gather.
    assert [r["tool_use_id"] for r in results] == ["tu1", "tu2"]
    assert [r["content"] for r in results] == ["from-long", "from-short"]


def test_one_tool_raising_does_not_cancel_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising tool becomes an ``is_error`` result; siblings complete normally.

    ``return_exceptions=True`` plus the inner ``except`` in
    ``_dispatch_tool_uses`` guarantees the gather always completes, so
    one failing tool can never cancel its siblings.
    """

    async def boom(**kw: Any) -> str:
        raise RuntimeError("kaboom")

    async def ok(**kw: Any) -> str:
        return "ok-value"

    agent = _patched_agent(monkeypatch)
    agent._tool_map["boom"] = boom
    agent._tool_map["ok"] = ok

    state = {"turn": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        state["turn"] += 1
        if state["turn"] == 1:
            return _msg_response(
                [
                    ToolUseBlock(
                        type="tool_use", id="tu1", name="ok", input={}
                    ),
                    ToolUseBlock(
                        type="tool_use", id="tu2", name="boom", input={}
                    ),
                ],
                stop_reason="tool_use",
            )
        return _msg_response([TextBlock(type="text", text="final")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    asyncio.run(agent.run("hi"))

    tool_result_msg = next(
        m
        for m in agent._messages
        if m["role"] == "user"
        and any(b.get("type") == "tool_result" for b in m["content"])
    )
    by_id = {
        b["tool_use_id"]: b for b in tool_result_msg["content"] if b.get("type") == "tool_result"
    }
    # Sibling completed and merged normally.
    assert by_id["tu1"]["content"] == "ok-value"
    assert "is_error" not in by_id["tu1"]
    # Raising tool mapped to (is_error=True, "Error calling ...").
    assert by_id["tu2"]["is_error"] is True
    assert "Error calling" in by_id["tu2"]["content"]
    assert "kaboom" in by_id["tu2"]["content"]


# --- stream path ordering --------------------------------------------------


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


def _msg_delta(stop_reason: StopReason) -> RawMessageDeltaEvent:
    return RawMessageDeltaEvent(
        type="message_delta",
        delta=Delta(stop_reason=stop_reason),
        usage=MessageDeltaUsage(output_tokens=2, input_tokens=1),
    )


def _msg_stop() -> RawMessageStopEvent:
    return RawMessageStopEvent(type="message_stop")


def _stream_from(events: list[Any]) -> Any:
    async def gen() -> Any:
        for e in events:
            yield e

    return gen()


async def _drain(gen: Any) -> list[Any]:
    out: list[Any] = []
    async for ev in gen:
        out.append(ev)
    return out


@pytest.mark.asyncio
async def test_stream_emits_all_calls_then_all_results_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent streaming: every ``ToolCallEvent`` lands before any
    ``ToolResultEvent``; results follow in original block order, paired
    by ``call_id`` (not adjacency)."""

    async def long_tool(**kw: Any) -> str:
        await asyncio.sleep(0.08)
        return "long"

    async def short_tool(**kw: Any) -> str:
        await asyncio.sleep(0.005)
        return "short"

    agent = _patched_agent(monkeypatch)
    agent._tool_map["long_tool"] = long_tool
    agent._tool_map["short_tool"] = short_tool

    def turn1() -> list[Any]:
        return [
            _message_start("m1"),
            _block_start(
                0, ToolUseBlock(type="tool_use", id="tu1", name="long_tool", input={})
            ),
            _block_start(
                1, ToolUseBlock(type="tool_use", id="tu2", name="short_tool", input={})
            ),
            _block_stop(0),
            _block_stop(1),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]

    def turn2() -> list[Any]:
        return [
            _message_start("m2"),
            _block_start(0, TextBlock(type="text", text="")),
            _block_delta(0, TextDelta(type="text_delta", text="done")),
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

    calls = [e for e in out if isinstance(e, ToolCallEvent)]
    results = [e for e in out if isinstance(e, ToolResultEvent)]
    assert len(calls) == 2
    assert len(results) == 2

    # All ToolCallEvents precede all ToolResultEvents.
    last_call_idx = max(
        i for i, e in enumerate(out) if isinstance(e, ToolCallEvent)
    )
    first_result_idx = min(
        i for i, e in enumerate(out) if isinstance(e, ToolResultEvent)
    )
    assert last_call_idx < first_result_idx

    # Calls + results both in original block order (long=tu1, short=tu2),
    # even though short_tool finished first under the gather.
    assert [c.call_id for c in calls] == ["tu1", "tu2"]
    assert [r.call_id for r in results] == ["tu1", "tu2"]
    assert [r.tool for r in results] == ["long_tool", "short_tool"]


@pytest.mark.asyncio
async def test_single_tool_turn_event_sequence_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Golden: a single-tool turn yields ``ToolCallEvent`` then
    ``ToolResultEvent`` — the same sequence as before concurrent
    dispatch. The single-tool case is unchanged because there is nothing
    to overlap with."""

    def add(**kw: Any) -> str:
        return "5"

    agent = _patched_agent(monkeypatch)
    agent._tool_map["add"] = add

    def turn1() -> list[Any]:
        return [
            _message_start("m1"),
            _block_start(
                0, ToolUseBlock(type="tool_use", id="tu1", name="add", input={})
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

    # Single-tool turn: ToolCallEvent then ToolResultEvent then the
    # final-answer ContentDelta — exactly the pre-change sequence.
    assert isinstance(out[0], ToolCallEvent)
    assert out[0].name == "add"
    assert out[0].call_id == "tu1"
    assert isinstance(out[1], ToolResultEvent)
    assert out[1].call_id == "tu1"
    assert out[1].is_error is False
    deltas = [e for e in out if isinstance(e, ContentDelta)]
    assert deltas[0].kind == "text" and deltas[0].text == "5"


# --- skill-marker serialisation -------------------------------------------


def _make_agent_with_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Agent, Session]:
    """Agent + attached real Session for skill-state mutation tests."""
    from cothis.session import Session

    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **kw: MagicMock())
    agent = Agent(model="x", provider="openrouter", tools=[], max_iterations=5)
    db_path = tmp_path / "sessions" / "session.db"
    session = Session.new(db_path, cwd=tmp_path, model="x", flush_sync=True)
    agent.attach_session(session)
    return agent, session


def test_skill_markers_serialise_so_active_skills_not_corrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_skill`` + a normal async tool in one turn resolve cleanly.

    The skill-marker dispatch is serialised (via the shared lock) so the
    ``Session._active_skills`` mutation can't race; the normal async tool
    overlaps in time. Final state: the skill is active and the normal
    tool's result merged.
    """
    log: list[str] = []

    @tool(inject_session=True, skill_marker=True)
    async def sload(name: str, _session: Any) -> str:
        """Load.

        Args:
            name: skill name.
        """
        log.append(f"load:start:{name}")
        await asyncio.sleep(0.03)
        assert _session is not None
        _session._active_skills.add(name)
        log.append(f"load:end:{name}")
        return f"loaded:{name}"

    async def normal(**kw: Any) -> str:
        log.append("normal:start")
        await asyncio.sleep(0.02)
        log.append("normal:end")
        return "normal-ok"

    agent, session = _make_agent_with_session(tmp_path, monkeypatch)
    agent._tool_map["sload"] = sload
    agent._tool_map["normal"] = normal

    state = {"turn": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        state["turn"] += 1
        if state["turn"] == 1:
            return _msg_response(
                [
                    ToolUseBlock(
                        type="tool_use", id="tu1", name="sload", input={"name": "deploy"}
                    ),
                    ToolUseBlock(
                        type="tool_use", id="tu2", name="normal", input={}
                    ),
                ],
                stop_reason="tool_use",
            )
        return _msg_response([TextBlock(type="text", text="done")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    asyncio.run(agent.run("hi"))

    # The normal async tool overlapped with the (serialised) skill load.
    assert "normal:start" in log and "normal:end" in log
    # Skill state mutated exactly once, cleanly.
    assert "deploy" in session.active_skills
    # Normal tool result merged in original block order (tu1=sload, tu2=normal).
    tool_result_msg = next(
        m
        for m in agent._messages
        if m["role"] == "user"
        and any(b.get("type") == "tool_result" for b in m["content"])
    )
    results = [b for b in tool_result_msg["content"] if b.get("type") == "tool_result"]
    assert [r["tool_use_id"] for r in results] == ["tu1", "tu2"]
    assert results[1]["content"] == "normal-ok"


def test_load_then_deactivate_same_skill_pair_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_skill('a')`` + ``deactivate_skill('a')`` in one turn resolve
    to a deterministic final active set regardless of body timing.

    Both blocks are skill-marker tools sharing the serialisation lock,
    so they execute strictly in original block order (load fully
    completes before deactivate starts). Without serialisation the
    check-then-mutate windows could interleave and the final set would
    depend on scheduling. Asserted via an execution log: load's end
    precedes deactivate's start, and the final active set is empty
    (loaded then deactivated).
    """
    log: list[str] = []

    @tool(inject_session=True, skill_marker=True)
    async def sload(name: str, _session: Any) -> str:
        """Load.

        Args:
            name: skill name.
        """
        log.append(f"load:start:{name}")
        await asyncio.sleep(0.03)  # body timing that would race unlocked
        assert _session is not None
        _session._active_skills.add(name)
        log.append(f"load:end:{name}")
        return f"loaded:{name}"

    @tool(inject_session=True, skill_marker=True)
    async def sunload(name: str, _session: Any) -> str:
        """Unload.

        Args:
            name: skill name.
        """
        log.append(f"unload:start:{name}")
        await asyncio.sleep(0.01)
        assert _session is not None
        _session._active_skills.discard(name)
        log.append(f"unload:end:{name}")
        return f"unloaded:{name}"

    agent, session = _make_agent_with_session(tmp_path, monkeypatch)
    agent._tool_map["sload"] = sload
    agent._tool_map["sunload"] = sunload

    state = {"turn": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        state["turn"] += 1
        if state["turn"] == 1:
            return _msg_response(
                [
                    ToolUseBlock(
                        type="tool_use", id="tu1", name="sload", input={"name": "a"}
                    ),
                    ToolUseBlock(
                        type="tool_use", id="tu2", name="sunload", input={"name": "a"}
                    ),
                ],
                stop_reason="tool_use",
            )
        return _msg_response([TextBlock(type="text", text="done")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    asyncio.run(agent.run("hi"))

    # Serialised: load fully completed before unload started (no
    # interleaving of the check-then-mutate windows).
    assert log.index("load:end:a") < log.index("unload:start:a")
    # Final active set is deterministic: load-then-unload leaves it empty.
    assert "a" not in session.active_skills
    assert session.active_skills == frozenset()

    # Results merged in original block order regardless of body timing.
    tool_result_msg = next(
        m
        for m in agent._messages
        if m["role"] == "user"
        and any(b.get("type") == "tool_result" for b in m["content"])
    )
    results = [b for b in tool_result_msg["content"] if b.get("type") == "tool_result"]
    assert [r["tool_use_id"] for r in results] == ["tu1", "tu2"]
    assert results[0]["content"] == "loaded:a"
    assert results[1]["content"] == "unloaded:a"
