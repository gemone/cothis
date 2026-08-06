"""Tests for ``SessionWorker`` (#225).

The worker owns one Agent + binds a loopback WebSocket that accepts
control messages (``run_turn`` / ``ping`` / ``shutdown``). Handshake
requires a valid bearer token on the ``Authorization`` header; missing
or wrong token → HTTP 401 + connection rejected.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
import websockets

if TYPE_CHECKING:
    from pathlib import Path


def _mock_agent() -> Any:
    """Agent stub whose ``run_stream`` yields one delta + closes."""
    from cothis.agent import ContentDelta, ToolCallEvent

    async def _run_stream(prompt: str):
        yield ContentDelta(kind="text", text="hello ")
        yield ContentDelta(kind="text", text="world")
        yield ToolCallEvent(
            name="fs.read", arguments={"path": "a.py"}, call_id="tu_test",
        )

    agent = MagicMock()
    agent.run_stream = _run_stream
    agent.aclose = MagicMock(return_value=asyncio.sleep(0))
    agent._session = None
    agent._bus = None
    return agent


# ---------------------------------------------------------------------
# Lifecycle: start + stop
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_starts_and_binds_loopback_port() -> None:
    """``start`` binds a WS server on 127.0.0.1 + returns a usable URI."""
    from cothis.worker import SessionWorker

    worker = SessionWorker(_mock_agent())
    try:
        uri = await worker.start()
        assert uri is not None
        assert uri.startswith("ws://127.0.0.1:")
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_worker_token_is_url_safe() -> None:
    """Bearer token is non-empty + URL-safe (generated via ``secrets``)."""
    from cothis.worker import SessionWorker

    worker = SessionWorker(_mock_agent())
    assert isinstance(worker.token, str)
    assert len(worker.token) >= 32
    assert all(c.isalnum() or c in "-_" for c in worker.token)


# ---------------------------------------------------------------------
# Auth: missing + invalid token → 401
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_rejects_missing_token() -> None:
    """Handshake without Authorization header → 401."""
    from cothis.worker import SessionWorker

    worker = SessionWorker(_mock_agent())
    uri = await worker.start()
    try:
        with pytest.raises(websockets.exceptions.InvalidStatus) as exc:
            async with websockets.connect(uri):
                pass
        assert exc.value.response.status_code == 401
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_worker_rejects_invalid_token() -> None:
    """Handshake with wrong bearer token → 401."""
    from cothis.worker import SessionWorker

    worker = SessionWorker(_mock_agent())
    uri = await worker.start()
    try:
        with pytest.raises(websockets.exceptions.InvalidStatus) as exc:
            async with websockets.connect(
                uri, additional_headers={"Authorization": "Bearer wrong"}
            ):
                pass
        assert exc.value.response.status_code == 401
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_worker_accepts_valid_token() -> None:
    """Handshake with correct bearer token succeeds."""
    from cothis.worker import SessionWorker

    worker = SessionWorker(_mock_agent())
    uri = await worker.start()
    try:
        async with websockets.connect(
            uri, additional_headers={"Authorization": f"Bearer {worker.token}"}
        ):
            pass  # handshake succeeded
    finally:
        await worker.stop()


# ---------------------------------------------------------------------
# Concurrent handshake cap (#264)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_handshakes_enforce_max_conns_cap() -> None:
    """AC #264: 5 simultaneous handshakes against a cap=4 transport → 4×101 + 1×503.

    Without the fix (increment in ``conn_handler`` rather than ``process_request``)
    the check-then-increment gap spans the handshake response being sent, so all
    N callers observe ``_active_conns < cap`` and pass. With the fix the
    increment lands atomically with the cap check (no ``await`` between them),
    so exactly ``max_concurrent_conns`` callers upgrade and the rest see 503.
    """
    from cothis.worker import SessionWorker
    from cothis.ws import WebSocketServerTransport

    transport = WebSocketServerTransport(max_concurrent_conns=4)
    worker = SessionWorker(_mock_agent(), transport=transport)
    uri = await worker.start()
    auth_header = {"Authorization": f"Bearer {worker.token}"}

    async def attempt() -> str:
        try:
            async with websockets.connect(uri, additional_headers=auth_header):
                # Hold the connection open so the slot stays claimed while
                # sibling handshakes arrive. The slow recv blocks until we
                # gather results from all attempts and stop the worker.
                try:
                    await asyncio.wait_for(
                        # Block until the worker stops + closes peers, OR a
                        # 2s safety timeout (well below the test's overall
                        # budget).
                        asyncio.sleep(2.0),
                        timeout=2.5,
                    )
                except TimeoutError:
                    pass
                return "ok"
        except websockets.exceptions.InvalidStatus as exc:
            if exc.response.status_code == 503:
                return "rejected-503"
            raise

    try:
        results = await asyncio.gather(*(attempt() for _ in range(5)))
    finally:
        await worker.stop()

    assert results.count("ok") == 4, (
        f"expected 4 successful upgrades, got {results.count('ok')}; "
        f"full results: {sorted(results)}"
    )
    assert results.count("rejected-503") == 1, (
        f"expected 1 503 rejection, got {results.count('rejected-503')}; "
        f"full results: {sorted(results)}"
    )


# ---------------------------------------------------------------------
# Control messages
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_ping_pong() -> None:
    """``ping`` from client → ``pong`` from worker."""
    from cothis.worker import SessionWorker

    worker = SessionWorker(_mock_agent())
    uri = await worker.start()
    try:
        async with websockets.connect(
            uri, additional_headers={"Authorization": f"Bearer {worker.token}"}
        ) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            assert json.loads(raw) == {"type": "pong"}
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_worker_shutdown_closes_cleanly() -> None:
    """``shutdown`` closes the connection + stops the worker."""
    from cothis.worker import SessionWorker

    worker = SessionWorker(_mock_agent())
    uri = await worker.start()
    try:
        async with websockets.connect(
            uri, additional_headers={"Authorization": f"Bearer {worker.token}"}
        ) as ws:
            await ws.send(json.dumps({"type": "shutdown"}))
            # Connection should close from the worker side.
            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=2.0)
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_worker_run_turn_emits_assistant_delta_and_tool_call() -> None:
    """``run_turn`` drives ``Agent.run_stream`` and forwards each delta."""
    from cothis.worker import SessionWorker

    worker = SessionWorker(_mock_agent())
    uri = await worker.start()
    try:
        async with websockets.connect(
            uri, additional_headers={"Authorization": f"Bearer {worker.token}"}
        ) as ws:
            await ws.send(json.dumps({"type": "run_turn", "prompt": "hi"}))
            received: list[dict[str, Any]] = []
            # Drain the full turn (#I24): turn_started + 3 stream events +
            # turn_finished = 5 frames. Filter the new turn_* bookend frames
            # so the assertions below still target the 3 stream events.
            while len(received) < 5:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                received.append(json.loads(raw))
            stream = [
                m for m in received
                if m["type"] not in ("turn_started", "turn_finished")
            ]
            assert stream[0] == {"type": "assistant_delta", "kind": "text", "text": "hello "}
            assert stream[1] == {"type": "assistant_delta", "kind": "text", "text": "world"}
            assert stream[2] == {
                "type": "tool_call_started",
                "tool": "fs.read",
                "arguments": {"path": "a.py"},
                "call_id": "tu_test",
            }
    finally:
        await worker.stop()


# ---------------------------------------------------------------------
# Turn-lifecycle frames + interrupt (#I24)
# ---------------------------------------------------------------------


def _rich_mock_agent() -> Any:
    """Agent stub carrying model / session / budget / skills signals.

    Used by the #I24 turn-frame tests so ``turn_finished``'s payload
    (model / session_id / pressure / active_skills) carries non-default
    values that the assertions can pin down. Yields one text delta + ends.
    """
    from cothis.agent import ContentDelta
    from cothis.ai.context_budget import ContextBudget, PressureLevel

    async def _run_stream(prompt: str):
        yield ContentDelta(kind="text", text="ok")

    session = MagicMock()
    session.session_id = "abcdef0123456789abcdef0123456789"
    session.active_skills = frozenset({"git-commit", "reviewer"})

    agent = MagicMock()
    agent.run_stream = _run_stream
    agent.model = "m1"
    agent._session = session
    agent.session = session
    agent.context_budget = MagicMock(
        return_value=ContextBudget(
            used_tokens=100,
            capacity_tokens=1000,
            available_tokens=900,
            ratio=0.1,
            pressure=PressureLevel.NONE,
        ),
    )
    agent.aclose = MagicMock(return_value=asyncio.sleep(0))
    return agent


@pytest.mark.asyncio
async def test_run_turn_emits_turn_started_and_turn_finished_frames() -> None:
    """``run_turn`` yields ``turn_started`` then a terminal ``turn_finished`` (#I24).

    The ``turn_finished`` payload carries the post-turn model / session_id /
    pressure / active_skills snapshot read from the agent. This is the
    authoritative refresh the TUI uses to repaint the footer + reconcile
    run-state to idle.
    """
    from cothis.worker import SessionWorker

    worker = SessionWorker(_rich_mock_agent())
    uri = await worker.start()
    try:
        async with websockets.connect(
            uri, additional_headers={"Authorization": f"Bearer {worker.token}"}
        ) as ws:
            await ws.send(json.dumps({"type": "run_turn", "prompt": "hi"}))
            received: list[dict[str, Any]] = []
            while len(received) < 3:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                received.append(json.loads(raw))
            assert received[0]["type"] == "turn_started"
            finished = received[-1]
            assert finished["type"] == "turn_finished"
            assert finished["model"] == "m1"
            assert finished["session_id"] == "abcdef0123456789abcdef0123456789"
            assert finished["pressure"] == "none"
            assert finished["active_skills"] == ["git-commit", "reviewer"]
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_interrupt_turn_cancels_active_turn_and_emits_turn_finished() -> None:
    """``interrupt_turn`` cancels ``_active_turn`` + still yields ``turn_finished`` (#I24).

    The finally fires despite cancellation — that terminal-frame guarantee
    is what the TUI relies on to return to idle. Reuses the same cancel
    primitive as run_turn-supersede + disconnect.
    """
    from cothis.worker import SessionWorker

    blocker = asyncio.Event()

    async def _blocking_run_stream(prompt: str):
        # Block until interrupted (or the test's safety timeout). The
        # unreachable ``yield`` makes this an async generator.
        await blocker.wait()
        yield  # pragma: no cover

    agent = _rich_mock_agent()
    agent.run_stream = _blocking_run_stream
    worker = SessionWorker(agent)
    uri = await worker.start()
    try:
        async with websockets.connect(
            uri, additional_headers={"Authorization": f"Bearer {worker.token}"}
        ) as ws:
            await ws.send(json.dumps({"type": "run_turn", "prompt": "hi"}))
            # Wait for turn_started so we know the turn task is in flight
            # before interrupting.
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            assert json.loads(raw)["type"] == "turn_started"
            await ws.send(json.dumps({"type": "interrupt_turn"}))
            # The interrupt handler cancels + awaits the task; the finally
            # then emits turn_finished.
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            assert json.loads(raw)["type"] == "turn_finished"
            # The interrupt handler clears ``_active_turn``.
            assert worker._active_turn is None or worker._active_turn.done()
    finally:
        blocker.set()
        await worker.stop()
