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
        yield ToolCallEvent(name="fs.read", arguments={"path": "a.py"})

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
            # Loop until the turn ends (mock_agent yields exactly 3 events).
            while len(received) < 3:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                received.append(json.loads(raw))
            assert received[0] == {"type": "assistant_delta", "kind": "text", "text": "hello "}
            assert received[1] == {"type": "assistant_delta", "kind": "text", "text": "world"}
            assert received[2] == {
                "type": "tool_call_started",
                "tool": "fs.read",
                "arguments": {"path": "a.py"},
            }
    finally:
        await worker.stop()
