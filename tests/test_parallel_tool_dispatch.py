"""Tests for ordered-concurrency tool dispatch (#316).

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
import logging
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


def _patched_agent(
    monkeypatch: pytest.MonkeyPatch, *, max_concurrent_tools: int = 8
) -> Agent:
    """Agent with a mocked provider — no real LLM calls."""
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **kw: MagicMock())
    return Agent(
        model="x",
        provider="openrouter",
        tools=[],
        max_iterations=5,
        max_concurrent_tools=max_concurrent_tools,
    )


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


# --- bounded live concurrency (max_concurrent_tools) -----------------------
#
# ``_dispatch_tool_uses`` now acquires an ``asyncio.Semaphore`` INSIDE each
# gathered coroutine so peak live tool executions within one fan-out turn
# never exceed ``max_concurrent_tools``. Result ORDER (submission order),
# the ``duration_ms`` capture, and the ``(True, "Error calling ...")`` error
# mapping are all preserved. These tests are fully DETERMINISTIC — they use
# shared counters and ``asyncio.Event`` gates, never wall-clock sleeps, for
# their assertions. The ``asyncio.wait_for(..., timeout=...)`` calls below
# are hang-guards (only fire on a regression that would otherwise deadlock
# the suite), not timing assertions.

_GUARD = 10.0  # seconds; only hit on a deadlock regression, never on pass


def _tool_result_blocks(agent: Agent) -> list[dict[str, Any]]:
    msg = next(
        m
        for m in agent._messages
        if m["role"] == "user"
        and any(b.get("type") == "tool_result" for b in m["content"])
    )
    return [b for b in msg["content"] if b.get("type") == "tool_result"]


@pytest.mark.asyncio
async def test_peak_live_concurrency_capped_at_configured_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cap=4`` with a 12-tool fan-out never exceeds 4 live executions.

    Each tool increments a shared counter on entry, asserts ``counter <=
    cap`` (fails fast if the bound ever breaks), records the high-water
    mark, then blocks on a shared gate. Exactly ``cap`` acquire the
    semaphore and block; the other ``total - cap`` queue on the semaphore.
    Asserted BEFORE the gate is released — proves the cap is the configured
    value (not accidental serialisation → 1, not unbounded → 12).
    """
    cap = 4
    total = 12
    state: dict[str, int] = {"counter": 0, "high_water": 0}
    reached_cap = asyncio.Event()
    proceed = asyncio.Event()

    def make_tool(idx: int) -> Any:
        async def gated(**kw: Any) -> str:
            state["counter"] += 1
            assert state["counter"] <= cap, (
                f"live executions {state['counter']} exceeded cap {cap}"
            )
            state["high_water"] = max(state["high_water"], state["counter"])
            if state["counter"] == cap:
                reached_cap.set()
            await proceed.wait()
            state["counter"] -= 1
            return f"r{idx}"

        return gated

    agent = _patched_agent(monkeypatch, max_concurrent_tools=cap)
    for i in range(total):
        agent._tool_map[f"tool{i}"] = make_tool(i)

    turn = {"i": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        turn["i"] += 1
        if turn["i"] == 1:
            return _msg_response(
                [
                    ToolUseBlock(
                        type="tool_use", id=f"tu{i}", name=f"tool{i}", input={}
                    )
                    for i in range(total)
                ],
                stop_reason="tool_use",
            )
        return _msg_response([TextBlock(type="text", text="final")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    task = asyncio.create_task(agent.run("hi"))

    await asyncio.wait_for(reached_cap.wait(), timeout=_GUARD)
    assert state["counter"] == cap, (
        f"expected exactly {cap} tools live, got {state['counter']}"
    )
    assert state["high_water"] == cap, (
        f"expected high-water {cap}, got {state['high_water']}"
    )

    proceed.set()  # release all; queued tools then acquire in turn
    answer = await task
    assert answer == "final"

    # All ``total`` results merged in ORIGINAL block order, none cancelled.
    results = _tool_result_blocks(agent)
    assert [r["tool_use_id"] for r in results] == [f"tu{i}" for i in range(total)]
    assert [r["content"] for r in results] == [f"r{i}" for i in range(total)]


@pytest.mark.asyncio
async def test_below_cap_concurrency_unaffected_by_semaphore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the default cap (8) and 2 slow async tools, both overlap.

    Directly re-asserts the existing overlap guarantee under the new code
    path: the semaphore is invisible when fan-out < cap (no latency
    regression vs the unbounded dispatch). Deterministic via an in-flight
    high-water counter + a gate event, no wall-clock.
    """
    state: dict[str, int] = {"in_flight": 0, "max_in_flight": 0}
    both_in = asyncio.Event()
    proceed = asyncio.Event()

    async def slow(**kw: Any) -> str:
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        if state["in_flight"] == 2:
            both_in.set()
        await proceed.wait()
        state["in_flight"] -= 1
        return "done"

    agent = _patched_agent(monkeypatch)  # default cap 8 > 2-tool fan-out
    agent._tool_map["slow_a"] = slow
    agent._tool_map["slow_b"] = slow

    turn = {"i": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        turn["i"] += 1
        if turn["i"] == 1:
            return _msg_response(
                [
                    ToolUseBlock(type="tool_use", id="tu1", name="slow_a", input={}),
                    ToolUseBlock(type="tool_use", id="tu2", name="slow_b", input={}),
                ],
                stop_reason="tool_use",
            )
        return _msg_response([TextBlock(type="text", text="final")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    task = asyncio.create_task(agent.run("hi"))

    await asyncio.wait_for(both_in.wait(), timeout=_GUARD)
    assert state["max_in_flight"] == 2, (
        f"expected both tools to overlap (2), got {state['max_in_flight']} "
        f"— semaphore serialised a below-cap fan-out"
    )
    proceed.set()
    answer = await task
    assert answer == "final"


@pytest.mark.asyncio
async def test_result_order_preserved_under_contention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cap=2`` with 4 tools: results land in submission order even though
    completion order is scrambled.

    Each tool blocks on a per-index gate; the gates are opened so completion
    order (1, 0, 3, 2) differs from submission order (0, 1, 2, 3).
    ``tool_result`` ids must still be [tu0, tu1, tu2, tu3]. Driven via a
    per-acquisition pulse event, no wall-clock.
    """
    cap = 2
    total = 4
    acquired: list[int] = []
    acquire_pulse = asyncio.Event()
    releases = [asyncio.Event() for _ in range(total)]

    def make_tool(idx: int) -> Any:
        async def t(**kw: Any) -> str:
            acquired.append(idx)
            acquire_pulse.set()
            await releases[idx].wait()
            return f"r{idx}"

        return t

    async def wait_acquired(n: int) -> None:
        while len(acquired) < n:
            await asyncio.wait_for(acquire_pulse.wait(), timeout=_GUARD)
            acquire_pulse.clear()

    agent = _patched_agent(monkeypatch, max_concurrent_tools=cap)
    for i in range(total):
        agent._tool_map[f"tool{i}"] = make_tool(i)

    turn = {"i": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        turn["i"] += 1
        if turn["i"] == 1:
            return _msg_response(
                [
                    ToolUseBlock(
                        type="tool_use", id=f"tu{i}", name=f"tool{i}", input={}
                    )
                    for i in range(total)
                ],
                stop_reason="tool_use",
            )
        return _msg_response([TextBlock(type="text", text="final")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    task = asyncio.create_task(agent.run("hi"))

    # First ``cap`` tools acquire (tool0, tool1 in FIFO semaphore order).
    await wait_acquired(cap)
    assert set(acquired[:cap]) == {0, 1}
    # Scramble completion order: release tool1 before tool0, then tool3
    # before tool2 — completion (1, 0, 3, 2) != submission (0, 1, 2, 3).
    releases[1].set()
    await wait_acquired(cap + 1)  # tool1 done → tool2 acquires
    releases[0].set()
    await wait_acquired(cap + 2)  # tool0 done → tool3 acquires
    releases[3].set()  # tool3 done before tool2
    releases[2].set()  # tool2 done last
    answer = await task
    assert answer == "final"

    # Results merged in ORIGINAL block order despite scrambled completion.
    results = _tool_result_blocks(agent)
    assert [r["tool_use_id"] for r in results] == [f"tu{i}" for i in range(total)]
    assert [r["content"] for r in results] == [f"r{i}" for i in range(total)]


@pytest.mark.asyncio
async def test_error_isolation_releases_semaphore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cap=2`` with 6 tools, one raising: the raiser maps to
    ``(is_error=True, "Error calling ...")`` at its ORIGINAL index, siblings
    complete normally, and results merge in original order.

    The semaphore's available-permit count returns to ``cap`` after the run
    — the direct proof that ``async with`` released on the exception path.
    (Had it leaked, a permit would be permanently consumed; the run would
    also eventually stall once enough errored tools drained the pool.)
    """
    cap = 2
    total = 6
    raise_idx = 2  # tool2 raises

    def make_tool(idx: int) -> Any:
        async def t(**kw: Any) -> str:
            if idx == raise_idx:
                raise RuntimeError("kaboom")
            return f"r{idx}"

        return t

    agent = _patched_agent(monkeypatch, max_concurrent_tools=cap)
    for i in range(total):
        agent._tool_map[f"tool{i}"] = make_tool(i)

    turn = {"i": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        turn["i"] += 1
        if turn["i"] == 1:
            return _msg_response(
                [
                    ToolUseBlock(
                        type="tool_use", id=f"tu{i}", name=f"tool{i}", input={}
                    )
                    for i in range(total)
                ],
                stop_reason="tool_use",
            )
        return _msg_response([TextBlock(type="text", text="final")])

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    # ``agent.run`` would stall if the semaphore leaked on raise, because the
    # errored tool's permit would never return to the pool.
    answer = await asyncio.wait_for(agent.run("hi"), timeout=_GUARD)
    assert answer == "final"

    results = _tool_result_blocks(agent)
    assert [r["tool_use_id"] for r in results] == [f"tu{i}" for i in range(total)]
    by_id = {b["tool_use_id"]: b for b in results}
    # Siblings complete normally.
    for i in range(total):
        if i == raise_idx:
            continue
        assert by_id[f"tu{i}"]["content"] == f"r{i}"
        assert "is_error" not in by_id[f"tu{i}"]
    # Raiser mapped to (is_error=True, "Error calling ...") at its index.
    assert by_id[f"tu{raise_idx}"]["is_error"] is True
    assert "Error calling" in by_id[f"tu{raise_idx}"]["content"]
    assert "kaboom" in by_id[f"tu{raise_idx}"]["content"]
    # The semaphore released every permit — none leaked on the raise path.
    assert agent._dispatch_semaphore._value == cap


# --- COTHIS_MAX_CONCURRENT_TOOLS override-or-None --------------------------
# These cover the ``model_post_init`` env-var branch — the only tuner for the
# non-CLI construction sites (the ``worker`` subprocess and ``acp_bridge``).
# They deliberately OMIT ``max_concurrent_tools`` from the constructor (unlike
# :func:`_patched_agent`, which always passes it) so the field is absent from
# ``model_fields_set`` and the override path runs. No asyncio is needed: the
# semaphore's initial ``_value`` equals the resolved cap.


def test_env_override_sets_cap_when_no_explicit_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``COTHIS_MAX_CONCURRENT_TOOLS`` sets the cap when no kwarg is passed."""
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **kw: MagicMock())
    monkeypatch.setenv("COTHIS_MAX_CONCURRENT_TOOLS", "3")
    agent = Agent(model="x", provider="openrouter", tools=[], max_iterations=5)
    assert agent.max_concurrent_tools == 3
    assert agent._dispatch_semaphore._value == 3


def test_env_override_ignores_non_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-positive env value falls back to the default — ``Semaphore(0)``
    would block every dispatch forever."""
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **kw: MagicMock())
    monkeypatch.setenv("COTHIS_MAX_CONCURRENT_TOOLS", "0")
    agent = Agent(model="x", provider="openrouter", tools=[], max_iterations=5)
    assert agent.max_concurrent_tools == 8
    assert agent._dispatch_semaphore._value == 8


def test_explicit_kwarg_suppresses_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``max_concurrent_tools`` kwarg wins over the env var: the
    field is in ``model_fields_set``, so the override branch is skipped."""
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **kw: MagicMock())
    monkeypatch.setenv("COTHIS_MAX_CONCURRENT_TOOLS", "3")
    agent = Agent(
        model="x",
        provider="openrouter",
        tools=[],
        max_iterations=5,
        max_concurrent_tools=4,
    )
    assert agent.max_concurrent_tools == 4
    assert agent._dispatch_semaphore._value == 4


# --- tool_timeout: per-body wall-clock bound --------------------------------
# A tiny ``tool_timeout`` (0.05s) plus a tool awaiting a never-set
# ``asyncio.Event`` gives a deterministic hang with NO real wall-clock sleep.
# ``_execute_tool`` is the single dispatch funnel; calling it directly tests
# the timeout wrap point without an LLM round-trip. Every hang call is
# wrapped in ``asyncio.wait_for(..., _GUARD)`` so a regression (timeout fails
# to fire) fails the test fast instead of hanging the suite.


def _timeout_agent(
    monkeypatch: pytest.MonkeyPatch, **kw: Any
) -> Agent:
    """Agent with a mocked provider and caller-supplied ``tool_timeout`` /
    ``max_concurrent_tools``. Does NOT default ``tool_timeout`` so the env
    tests can omit it from ``model_fields_set``."""
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **k: MagicMock())
    return Agent(
        model="x",
        provider="openrouter",
        tools=[],
        max_iterations=5,
        **kw,
    )


@pytest.mark.asyncio
async def test_tool_timeout_fires_on_hanging_async_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool that never resolves is cut off at ``tool_timeout`` and mapped to
    the error tuple with the exact "timed out after <s>s" wording — the same
    ``(is_error, "Error calling ...")`` shape a raised exception produces."""
    agent = _timeout_agent(monkeypatch, tool_timeout=0.05)

    never = asyncio.Event()

    async def hang(**kw: Any):
        await never.wait()  # never set -> deterministic hang

    agent._tool_map["hang"] = hang

    is_error, output = await asyncio.wait_for(
        agent._execute_tool({"name": "hang", "input": {}}), timeout=_GUARD
    )
    assert is_error is True
    assert output == "Error calling hang: timed out after 0.05s"


@pytest.mark.asyncio
async def test_tool_timeout_does_not_fire_when_tool_completes_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool that returns well under ``tool_timeout`` yields a normal result —
    the bound never interferes with prompt tools."""
    agent = _timeout_agent(monkeypatch, tool_timeout=0.05)

    async def quick(**kw: Any) -> str:
        return "ok"

    agent._tool_map["quick"] = quick

    is_error, output = await agent._execute_tool({"name": "quick", "input": {}})
    assert is_error is False
    assert output == "ok"


def test_tool_timeout_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default is ``None`` — no timeout = today's behavior (backward
    compat). No env var set so the override branch is a no-op too."""
    monkeypatch.delenv("COTHIS_TOOL_TIMEOUT", raising=False)
    agent = _timeout_agent(monkeypatch)
    assert agent.tool_timeout is None


@pytest.mark.asyncio
async def test_none_tool_timeout_never_enters_asyncio_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``tool_timeout is None`` the ``nullcontext`` branch runs, so
    ``asyncio.timeout`` is never even entered. Monkeypatch it to count calls
    and assert the counter stays 0; a fast tool still returns normally. We do
    NOT await a never-set event under None (that would hang the test forever,
    which is exactly the backward-compat point)."""
    agent = _timeout_agent(monkeypatch)
    assert agent.tool_timeout is None

    calls = {"n": 0}
    real_timeout = asyncio.timeout

    def counting_timeout(delay: Any) -> Any:
        calls["n"] += 1
        return real_timeout(delay)

    monkeypatch.setattr(asyncio, "timeout", counting_timeout)

    async def quick(**kw: Any) -> str:
        return "ok"

    agent._tool_map["quick"] = quick
    is_error, output = await agent._execute_tool({"name": "quick", "input": {}})
    assert is_error is False
    assert output == "ok"
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_tool_timeout_configurable_via_ctor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tool_timeout=0.05`` passed to the ctor is honored: the hanging tool
    is cut off and the ctor value is visible on the field."""
    agent = _timeout_agent(monkeypatch, tool_timeout=0.05)
    assert agent.tool_timeout == 0.05

    never = asyncio.Event()

    async def hang(**kw: Any):
        await never.wait()

    agent._tool_map["hang"] = hang

    is_error, output = await asyncio.wait_for(
        agent._execute_tool({"name": "hang", "input": {}}), timeout=_GUARD
    )
    assert is_error is True
    assert "timed out after 0.05s" in output


@pytest.mark.asyncio
async def test_timed_out_tool_does_not_cancel_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One tool timing out must not cancel or reorder its sibling: tool A
    awaits a never-set event (times out at 0.05s), tool B returns immediately.
    Both outcomes land in SUBMISSION order; B is a normal result, A is the
    timeout error tuple. This is the per-``_execute_tool``-coroutine guarantee
    that keeps the gather isolation intact under the timeout."""
    agent = _timeout_agent(
        monkeypatch, max_concurrent_tools=8, tool_timeout=0.05
    )

    never = asyncio.Event()

    async def hang(**kw: Any):
        await never.wait()

    async def quick(**kw: Any) -> str:
        return "ok"

    agent._tool_map["hang"] = hang
    agent._tool_map["quick"] = quick

    blocks = [
        {"type": "tool_use", "id": "tu_a", "name": "hang", "input": {}},
        {"type": "tool_use", "id": "tu_b", "name": "quick", "input": {}},
    ]
    outcomes = await asyncio.wait_for(
        agent._dispatch_tool_uses(blocks), timeout=_GUARD
    )

    assert len(outcomes) == 2
    # Submission order preserved: hang (index 0) timed out.
    assert outcomes[0][0] is True
    assert "timed out after 0.05s" in outcomes[0][1]
    # Sibling completed normally — not cancelled.
    assert outcomes[1][0] is False
    assert outcomes[1][1] == "ok"


@pytest.mark.asyncio
async def test_semaphore_released_after_tool_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``max_concurrent_tools=1`` the timed-out tool's permit must return
    to the pool (the timeout is caught inside ``_execute_tool``, so it returns
    normally and the ``async with`` releases). After the timeout, a follow-up
    dispatch must acquire immediately — no deadlock. (Had the permit leaked,
    the second dispatch would block forever and ``_GUARD`` would fire.)"""
    cap = 1
    agent = _timeout_agent(
        monkeypatch, max_concurrent_tools=cap, tool_timeout=0.05
    )

    never = asyncio.Event()

    async def hang(**kw: Any):
        await never.wait()

    agent._tool_map["hang"] = hang

    outcomes = await asyncio.wait_for(
        agent._dispatch_tool_uses(
            [{"type": "tool_use", "id": "tu1", "name": "hang", "input": {}}]
        ),
        timeout=_GUARD,
    )
    assert outcomes[0][0] is True
    assert "timed out after 0.05s" in outcomes[0][1]
    # Permit fully restored — the dispatch-semaphore invariant holds through the timeout path.
    assert agent._dispatch_semaphore._value == cap

    # Follow-up acquires: a quick tool dispatches without blocking.
    async def quick(**kw: Any) -> str:
        return "ok"

    agent._tool_map["quick"] = quick
    outcomes2 = await asyncio.wait_for(
        agent._dispatch_tool_uses(
            [{"type": "tool_use", "id": "tu2", "name": "quick", "input": {}}]
        ),
        timeout=_GUARD,
    )
    assert outcomes2[0][0] is False
    assert outcomes2[0][1] == "ok"
    assert agent._dispatch_semaphore._value == cap


# --- COTHIS_TOOL_TIMEOUT override-or-None -----------------------------------
# Mirror the ``COTHIS_MAX_CONCURRENT_TOOLS`` override suite: omit the kwarg so
# the field is absent from ``model_fields_set`` and the override branch runs.
# No asyncio needed — the resolved ``tool_timeout`` float is the proof.


def test_env_override_sets_timeout_when_no_explicit_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``COTHIS_TOOL_TIMEOUT`` sets the bound when no kwarg is passed."""
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **k: MagicMock())
    monkeypatch.setenv("COTHIS_TOOL_TIMEOUT", "0.05")
    agent = Agent(model="x", provider="openrouter", tools=[], max_iterations=5)
    assert agent.tool_timeout == 0.05


@pytest.mark.parametrize("bad", ["abc", "0", "-1"])
def test_env_override_ignores_bad_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    bad: str,
) -> None:
    """Non-numeric (``abc``) and non-positive (``0`` / ``-1``) env values are
    rejected with a warning and leave ``tool_timeout`` at ``None`` (no timeout
    = today's behavior, never a silently-broken zero/negative bound)."""
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **k: MagicMock())
    monkeypatch.setenv("COTHIS_TOOL_TIMEOUT", bad)
    with caplog.at_level(logging.WARNING):
        agent = Agent(
            model="x", provider="openrouter", tools=[], max_iterations=5
        )
    assert agent.tool_timeout is None
    assert any(
        "COTHIS_TOOL_TIMEOUT" in rec.getMessage() and rec.levelno == logging.WARNING
        for rec in caplog.records
    )


def test_explicit_kwarg_suppresses_timeout_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``tool_timeout`` kwarg wins over the env var: the field is
    in ``model_fields_set``, so the override branch is skipped."""
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **k: MagicMock())
    monkeypatch.setenv("COTHIS_TOOL_TIMEOUT", "99")
    agent = Agent(
        model="x",
        provider="openrouter",
        tools=[],
        max_iterations=5,
        tool_timeout=0.05,
    )
    assert agent.tool_timeout == 0.05


