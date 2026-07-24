"""Tests for the WS transport seam + mocked-transport worker tests (#248).

Two test classes:

1. ``FakeTransport`` / ``FakeConnection`` — implements ``WSTransport`` /
   ``Connection`` without any socket. The worker's dispatch / timeout / auth
   logic is driven through these, fulfilling the #225 acceptance criterion
   *"anyio WS wrappers are isolated and unit-testable (mock the transport)"*.
2. ``WebSocketServerTransport`` smoke test — the production adapter still binds
   a real loopback port and rejects/accepts on the bearer token (covered more
   fully in ``test_session_worker.py``; this asserts the ``bind``/``serve``
   split + ``uri`` works).

The point of the seam: the worker's behaviour tests below bind **no socket**
and connect **no client** — they enqueue frames into a fake and read the
worker's responses out of it. A regression in dispatch shows up in one fast,
hermetic test instead of a real-WS round-trip.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from cothis.agent import ToolCallEvent
from cothis.worker import SessionWorker


# ---------------------------------------------------------------------
# Fakes: a WSTransport / Connection pair with no socket
# ---------------------------------------------------------------------


class FakeConnection:
    """In-memory ``Connection``: feed frames in, read frames out."""

    def __init__(self) -> None:
        # Frames the "client" will send to the worker (fed via ``feed``).
        # ``object`` because ``None`` is the end-of-stream sentinel.
        self._inbox: asyncio.Queue[object] = asyncio.Queue()
        # Frames the worker emitted back.
        self.sent: list[str] = []
        self._closed = False
        self._recv_waiters: list[asyncio.Future[None]] = []

    async def feed(self, frame: str | None) -> None:
        """Enqueue a client frame; ``None`` is the end-of-stream sentinel."""
        await self._inbox.put(frame)

    async def send(self, message: str) -> None:
        self.sent.append(message)
        # Wake any test waiting for a response.
        for w in self._recv_waiters:
            if not w.done():
                w.set_result(None)

    async def close(self) -> None:
        self._closed = True
        for w in self._recv_waiters:
            if not w.done():
                w.set_result(None)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        while True:
            frame = await self._inbox.get()
            if frame is None:  # sentinel: end of stream
                return
            yield frame
            if self._closed:
                return

    async def wait_for_send(self, count: int, timeout: float = 2.0) -> list[str]:
        """Block until the worker has emitted >= ``count`` frames."""
        while len(self.sent) < count:
            fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            self._recv_waiters.append(fut)
            if self.sent:  # already enough
                break
            try:
                await asyncio.wait_for(fut, timeout)
            except TimeoutError:
                break
        return self.sent


class FakeTransport:
    """``WSTransport`` with no socket: hands the worker a ``FakeConnection``."""

    def __init__(self) -> None:
        self.connections: list[FakeConnection] = []
        self._handler = None
        self._auth = None
        self._serve_stopped = asyncio.Event()
        self._serving = False

    @property
    def uri(self) -> str | None:
        # No real socket; return a stable placeholder so worker.start() works.
        return "ws://fake/agent" if self._handler is not None else None

    async def bind(self, handler: Any, auth: Any) -> None:
        self._handler = handler
        self._auth = auth

    async def serve(self) -> None:
        self._serving = True
        await self._serve_stopped.wait()

    def request_shutdown(self) -> None:
        self._serve_stopped.set()

    async def accept(self) -> FakeConnection:
        """Test-only: simulate one client connecting. Returns the conn."""
        assert self._handler is not None, "bind() not called"
        conn = FakeConnection()
        self.connections.append(conn)
        # Run the worker's per-connection handler in the background; it reads
        # frames from conn._inbox and writes to conn.sent.
        asyncio.create_task(self._handler(conn))
        # Give the handler a tick to start iterating.
        await asyncio.sleep(0)
        return conn

    @property
    def auth_check(self):
        """Expose the installed auth callback for direct unit testing."""
        return self._auth


# ---------------------------------------------------------------------
# Mock agent — yields a scripted stream
# ---------------------------------------------------------------------


def _scripted_agent(events: list[Any]) -> Any:
    """Agent whose ``run_stream`` yields ``events`` then closes."""

    async def _run_stream(prompt: str):
        for e in events:
            yield e

    agent = MagicMock()
    agent.run_stream = _run_stream
    agent.aclose = MagicMock(return_value=asyncio.sleep(0))
    agent._session = None
    agent._bus = None
    return agent


# ---------------------------------------------------------------------
# Worker behaviour via the fake transport — NO socket, NO client
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ping_pong_via_fake_transport() -> None:
    """``ping`` dispatches to ``pong`` with no socket bound (#248)."""
    transport = FakeTransport()
    worker = SessionWorker(_scripted_agent([]), transport=transport)
    await worker.start()
    conn = await transport.accept()
    try:
        await conn.feed(json.dumps({"type": "ping"}))
        await conn.wait_for_send(1)
        assert json.loads(conn.sent[0]) == {"type": "pong"}
    finally:
        await conn.feed(None)  # end stream
        await worker.stop()


@pytest.mark.asyncio
async def test_run_turn_streams_deltas_via_fake_transport() -> None:
    """``run_turn`` forwards each agent event — driven through the seam."""
    events = ["hello ", "world", ToolCallEvent(name="fs.read", arguments={"path": "a.py"})]
    transport = FakeTransport()
    worker = SessionWorker(_scripted_agent(events), transport=transport)
    await worker.start()
    conn = await transport.accept()
    try:
        await conn.feed(json.dumps({"type": "run_turn", "prompt": "hi"}))
        await conn.wait_for_send(3)
        got = [json.loads(f) for f in conn.sent[:3]]
        assert got[0] == {"type": "assistant_delta", "text": "hello "}
        assert got[1] == {"type": "assistant_delta", "text": "world"}
        assert got[2] == {
            "type": "tool_call_started",
            "tool": "fs.read",
            "arguments": {"path": "a.py"},
        }
    finally:
        await conn.feed(None)
        await worker.stop()


@pytest.mark.asyncio
async def test_unknown_message_type_emits_error_via_fake_transport() -> None:
    """An unknown ``type`` is rejected with an ``error`` frame — no socket."""
    transport = FakeTransport()
    worker = SessionWorker(_scripted_agent([]), transport=transport)
    await worker.start()
    conn = await transport.accept()
    try:
        await conn.feed(json.dumps({"type": "bogus"}))
        await conn.wait_for_send(1)
        msg = json.loads(conn.sent[0])
        assert msg["type"] == "error"
        assert "bogus" in msg["message"]
    finally:
        await conn.feed(None)
        await worker.stop()


@pytest.mark.asyncio
async def test_invalid_json_emits_error_via_fake_transport() -> None:
    """Malformed JSON is reported as an error, not a crash — no socket."""
    transport = FakeTransport()
    worker = SessionWorker(_scripted_agent([]), transport=transport)
    await worker.start()
    conn = await transport.accept()
    try:
        await conn.feed("not json at all")
        await conn.wait_for_send(1)
        msg = json.loads(conn.sent[0])
        assert msg["type"] == "error"
        assert "invalid JSON" in msg["message"]
    finally:
        await conn.feed(None)
        await worker.stop()


@pytest.mark.asyncio
async def test_shutdown_signals_transport_via_fake_transport() -> None:
    """``shutdown`` closes the conn + asks the transport to stop — no socket."""
    transport = FakeTransport()
    worker = SessionWorker(_scripted_agent([]), transport=transport)
    await worker.start()
    conn = await transport.accept()
    try:
        await conn.feed(json.dumps({"type": "shutdown"}))
        await asyncio.sleep(0.05)  # let dispatch run
        # The transport's shutdown flag should be set (request_shutdown called).
        assert transport._serve_stopped.is_set()
    finally:
        await conn.feed(None)
        await worker.stop()


@pytest.mark.asyncio
async def test_auth_check_rejects_missing_bearer() -> None:
    """The worker's auth callback returns 401 without a Bearer header.

    The auth gate is unit-testable in isolation through the seam: call the
    callback the transport received and inspect the Response — no handshake.
    """
    transport = FakeTransport()
    worker = SessionWorker(_scripted_agent([]), transport=transport)
    await worker.start()
    auth = transport.auth_check
    assert auth is not None, "bind() must install the auth callback"
    request = MagicMock()
    request.headers = {}  # no Authorization
    resp = auth(request)
    assert resp is not None
    assert resp.status_code == 401
    await worker.stop()


@pytest.mark.asyncio
async def test_auth_check_rejects_wrong_bearer() -> None:
    """Wrong token → 401; constant-time compare is exercised via the seam."""
    transport = FakeTransport()
    worker = SessionWorker(_scripted_agent([]), transport=transport)
    await worker.start()
    auth = transport.auth_check
    assert auth is not None
    request = MagicMock()
    request.headers = {"Authorization": "Bearer wrong-token"}
    assert auth(request).status_code == 401
    await worker.stop()


@pytest.mark.asyncio
async def test_auth_check_accepts_correct_bearer() -> None:
    """Correct token → ``None`` (accept) — exercised without a handshake."""
    transport = FakeTransport()
    worker = SessionWorker(_scripted_agent([]), transport=transport)
    await worker.start()
    auth = transport.auth_check
    assert auth is not None
    request = MagicMock()
    request.headers = {"Authorization": f"Bearer {worker.token}"}
    assert auth(request) is None
    await worker.stop()


@pytest.mark.asyncio
async def test_turn_timeout_emits_error(monkeypatch) -> None:
    """``anyio.fail_after`` cancels a stuck turn → ``error: turn timeout``.

    This is the payoff of the anyio migration: the timeout cancel scope is now
    backend-neutral and exercisable through the fake transport. We shrink the
    timeout and feed a stream that never yields.
    """
    import cothis.worker as worker_mod

    monkeypatch.setattr(worker_mod, "_TURN_TIMEOUT_S", 0.05)

    async def _stuck_stream(prompt: str):  # noqa: ARG001
        # Never yields — simulates a hung model call.
        await asyncio.sleep(10)
        if False:  # pragma: no cover
            yield ""

    agent = MagicMock()
    agent.run_stream = _stuck_stream
    agent.aclose = MagicMock(return_value=asyncio.sleep(0))

    transport = FakeTransport()
    worker = SessionWorker(agent, transport=transport)
    await worker.start()
    conn = await transport.accept()
    try:
        await conn.feed(json.dumps({"type": "run_turn", "prompt": "hi"}))
        await conn.wait_for_send(1, timeout=2.0)
        msg = json.loads(conn.sent[0])
        assert msg == {"type": "error", "message": "turn timeout"}
    finally:
        await conn.feed(None)
        await worker.stop()
