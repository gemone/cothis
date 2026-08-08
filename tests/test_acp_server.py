"""Tests for ``cothis.protocol.acp.ACPServer`` — handshake + command dispatch.

Hermetic: a fake in-memory ``ByteConnection`` captures sent frames; a fake
``SessionBackend`` records calls and (for ``prompt``) streams one delta. No
sockets, no network, no real agent.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from cothis.protocol.acp import ACPServer
from cothis.protocol.messages import (
    PROTOCOL_VERSION,
    AssistantDelta,
    BackendError,
    ModelDescriptor,
    ModelRef,
    ProtocolError,
    SessionSnapshot,
    SessionSummary,
    ThinkingLevel,
)
from cothis.protocol.wire import (
    ServerMessageDecoder,
    encode_client_message,
)


class FakeConnection:
    """Captures sent frames and yields the chunks it was seeded with."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.sent = bytearray()
        self.closed = False

    async def send(self, chunk: bytes) -> None:
        self.sent += chunk

    async def close(self, final_chunk: bytes | None = None) -> None:
        if final_chunk is not None:
            self.sent += final_chunk
        self.closed = True

    def __aiter__(self) -> FakeConnection:
        return self

    async def __anext__(self) -> bytes:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _QueueConn:
    """A connection backed by an asyncio.Queue for async chunk delivery.

    Lets a test feed frames dynamically (after the previous frame has been
    processed) and await server output mid-turn — needed for the abort
    mid-turn scenario where a blocking prompt must start before the abort
    arrives.
    """

    def __init__(self) -> None:
        self._q: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.sent = bytearray()
        self.closed = False

    async def send(self, chunk: bytes) -> None:
        self.sent += chunk

    async def close(self, final_chunk: bytes | None = None) -> None:
        if final_chunk is not None:
            self.sent += final_chunk
        self.closed = True

    def __aiter__(self) -> _QueueConn:
        return self

    async def __anext__(self) -> bytes:
        item = await self._q.get()
        if item is None:
            raise StopAsyncIteration
        return item

    def feed(self, chunk: bytes) -> None:
        self._q.put_nowait(chunk)

    def feed_eof(self) -> None:
        self._q.put_nowait(None)


class FakeBackend:
    def __init__(
        self,
        *,
        block: asyncio.Event | None = None,
        started: asyncio.Event | None = None,
    ) -> None:
        self.created: list[SessionSnapshot] = []
        self.prompt_calls: list[tuple[str, str]] = []
        self.abort_calls: list[str] = []
        self.set_model_calls: list[tuple[str, ModelRef]] = []
        self.set_thinking_calls: list[tuple[str, ThinkingLevel]] = []
        self._block = block
        self._started = started

    async def models(self) -> list[ModelDescriptor]:
        # The honest advertisement: a single configured model with limits
        # the bundled metadata can resolve for it.
        return [
            ModelDescriptor(
                provider="openrouter",
                id="openai/gpt-oss-120b",
                maxOutputTokens=32768,
                contextWindow=131072,
            )
        ]

    async def list_sessions(self) -> list[SessionSummary]:
        return [
            SessionSummary(
                id=s.id,
                cwd=s.cwd,
                phase="idle",
                model=s.model,
                thinkingLevel="off",
                createdAt=0,
                updatedAt=0,
            )
            for s in self.created
        ]

    async def create_session(
        self,
        cwd: str | None,
        name: str | None,
        model: ModelRef | None,
        thinking_level: Any,
    ) -> SessionSnapshot:
        snap = SessionSnapshot(
            id=uuid.uuid4().hex,
            cwd=cwd or "/",
            phase="idle",
            model=ModelRef(provider="p", id="m"),
            thinkingLevel="off",
            createdAt=0,
            updatedAt=0,
            revision=0,
            transcript=[],
        )
        self.created.append(snap)
        return snap

    async def prompt(self, session_id: str, text: str, emit: Any) -> SessionSnapshot:
        self.prompt_calls.append((session_id, text))
        if not any(s.id == session_id for s in self.created):
            raise BackendError(
                ProtocolError(code="not_found", message=f"session {session_id!r} not found")
            )
        await emit(
            AssistantDelta(
                type="assistant_delta",
                messageId="m1",
                contentIndex=0,
                kind="text",
                delta="Hi",
            )
        )
        if self._started is not None:
            self._started.set()
        if self._block is not None:
            try:
                await self._block.wait()
            except asyncio.CancelledError:
                snap = next(s for s in self.created if s.id == session_id)
                return snap.model_copy(update={"revision": 42})
        snap = next(s for s in self.created if s.id == session_id)
        return snap.model_copy(update={"revision": 1})

    async def abort(self, session_id: str) -> SessionSnapshot:
        self.abort_calls.append(session_id)
        if not any(s.id == session_id for s in self.created):
            raise BackendError(
                ProtocolError(code="not_found", message=f"session {session_id!r} not found")
            )
        return next(s for s in self.created if s.id == session_id)

    async def set_model(self, session_id: str, model: ModelRef) -> SessionSnapshot:
        self.set_model_calls.append((session_id, model))
        if not any(s.id == session_id for s in self.created):
            raise BackendError(
                ProtocolError(code="not_found", message=f"session {session_id!r} not found")
            )
        snap = next(s for s in self.created if s.id == session_id)
        return snap.model_copy(update={"model": model})

    async def set_thinking(
        self, session_id: str, level: ThinkingLevel
    ) -> SessionSnapshot:
        self.set_thinking_calls.append((session_id, level))
        if not any(s.id == session_id for s in self.created):
            raise BackendError(
                ProtocolError(code="not_found", message=f"session {session_id!r} not found")
            )
        snap = next(s for s in self.created if s.id == session_id)
        return snap.model_copy(update={"thinkingLevel": level})


def _enc(d: dict) -> bytes:
    return encode_client_message(d)


def _decode(conn: FakeConnection | _QueueConn) -> list:
    return ServerMessageDecoder().push(bytes(conn.sent))


async def _serve(server: ACPServer, frames: list[bytes]) -> list:
    conn = FakeConnection(frames)
    await server.serve_connection(conn)
    return _decode(conn)


def _hello(token: str = "secret", version: int = PROTOCOL_VERSION) -> bytes:
    return _enc({"type": "hello", "version": version, "token": token})


@pytest.mark.asyncio
async def test_handshake_success_sends_server_hello_with_snapshot() -> None:
    backend = FakeBackend()
    server = ACPServer(backend, token="secret")
    [msg] = await _serve(server, [_hello()])
    assert msg.type == "hello"
    assert msg.version == PROTOCOL_VERSION
    assert msg.snapshot.serverId == "cothis"
    assert msg.snapshot.sessions == []
    # The handshake snapshot advertises the backend's configured model.
    assert msg.snapshot.models == await backend.models()


@pytest.mark.asyncio
async def test_handshake_bad_token_returns_hello_error() -> None:
    server = ACPServer(FakeBackend(), token="secret")
    [msg] = await _serve(server, [_hello(token="WRONG")])
    assert msg.type == "hello_error"
    assert msg.error.code == "auth"


@pytest.mark.asyncio
async def test_handshake_bad_version_returns_version_error() -> None:
    server = ACPServer(FakeBackend(), token="secret")
    [msg] = await _serve(server, [_hello(version=PROTOCOL_VERSION + 1)])
    assert msg.type == "hello_error"
    assert msg.error.code == "version"


@pytest.mark.asyncio
async def test_first_message_must_be_hello() -> None:
    server = ACPServer(FakeBackend(), token="secret")
    [msg] = await _serve(
        server,
        [_enc({"type": "request", "id": "r", "request": {"command": "list"}})],
    )
    assert msg.type == "hello_error"
    assert msg.error.code == "invalid_request"


@pytest.mark.asyncio
async def test_create_then_list() -> None:
    backend = FakeBackend()
    server = ACPServer(backend, token="secret")
    out = await _serve(
        server,
        [
            _hello(),
            _enc({"type": "request", "id": "r1", "request": {"command": "create", "cwd": "/tmp"}}),
        ],
    )
    create_reply = out[1]
    assert create_reply.type == "response" and create_reply.ok
    assert create_reply.result.command == "create"
    sid = create_reply.result.session.id

    out2 = await _serve(
        ACPServer(backend, token="secret"),
        [_hello(), _enc({"type": "request", "id": "r2", "request": {"command": "list"}})],
    )
    list_reply = out2[1]
    assert list_reply.result.command == "list"
    assert [s.id for s in list_reply.result.sessions] == [sid]


@pytest.mark.asyncio
async def test_prompt_streams_progress_then_response() -> None:
    backend = FakeBackend()
    server = ACPServer(backend, token="secret")
    created = await _serve(
        server,
        [_hello(), _enc({"type": "request", "id": "c", "request": {"command": "create", "cwd": "/"}})],
    )
    sid = created[1].result.session.id

    out = await _serve(
        ACPServer(backend, token="secret"),
        [
            _hello(),
            _enc(
                {
                    "type": "request",
                    "id": "p",
                    "request": {"command": "prompt", "sessionId": sid, "text": "hello"},
                }
            ),
        ],
    )
    # ServerHello, progress event, prompt response.
    assert [m.type for m in out] == ["hello", "event", "response"]
    event = out[1].event
    assert event.type == "session_progress"
    assert event.progress.type == "assistant_delta"
    assert event.progress.delta == "Hi"
    response = out[2]
    assert response.ok and response.result.command == "prompt"
    assert backend.prompt_calls == [(sid, "hello")]


@pytest.mark.asyncio
async def test_unsupported_command_returns_invalid_request() -> None:
    # ``steer`` is defined in the schema but still unsupported (deferred).
    server = ACPServer(FakeBackend(), token="secret")
    out = await _serve(
        server,
        [_hello(), _enc({"type": "request", "id": "r", "request": {"command": "steer", "sessionId": "s", "text": "x"}})],
    )
    reply = out[1]
    assert reply.type == "response" and not reply.ok
    assert reply.error.code == "invalid_request"


@pytest.mark.asyncio
async def test_prompt_unknown_session_returns_not_found() -> None:
    server = ACPServer(FakeBackend(), token="secret")
    out = await _serve(
        server,
        [
            _hello(),
            _enc(
                {
                    "type": "request",
                    "id": "r",
                    "request": {"command": "prompt", "sessionId": "nope", "text": "x"},
                }
            ),
        ],
    )
    reply = next(m for m in out if m.type == "response")
    assert not reply.ok
    assert reply.error.code == "not_found"


@pytest.mark.asyncio
async def test_batched_hello_and_request_in_one_chunk() -> None:
    # A client may write hello + request in a single OS write; the server
    # must promote to ready mid-chunk and still dispatch the request.
    backend = FakeBackend()
    server = ACPServer(backend, token="secret")
    blob = _hello() + _enc({"type": "request", "id": "r", "request": {"command": "create", "cwd": "/"}})
    out = await _serve(server, [blob])
    assert [m.type for m in out] == ["hello", "response"]
    assert out[1].ok and out[1].result.command == "create"


@pytest.mark.asyncio
async def test_server_rejects_empty_token() -> None:
    with pytest.raises(ValueError):
        ACPServer(FakeBackend(), token="")


# ---------------------------------------------------------------------------
# abort / set_model / set_thinking dispatch
# ---------------------------------------------------------------------------


async def _create_session(backend: FakeBackend) -> str:
    """Create a session via the server and return its id."""
    created = await _serve(
        ACPServer(backend, token="secret"),
        [_hello(), _enc({"type": "request", "id": "c", "request": {"command": "create", "cwd": "/"}})],
    )
    return created[1].result.session.id


@pytest.mark.asyncio
async def test_abort_with_no_active_turn_returns_snapshot() -> None:
    # REQUIRED no-op-safe: aborting a session with NO active turn returns
    # ok + the current snapshot (not an error).
    backend = FakeBackend()
    sid = await _create_session(backend)

    out = await _serve(
        ACPServer(backend, token="secret"),
        [_hello(), _enc({"type": "request", "id": "a", "request": {"command": "abort", "sessionId": sid}})],
    )
    reply = out[1]
    assert reply.type == "response" and reply.ok
    assert reply.result.command == "abort"
    assert reply.result.session.id == sid
    assert backend.abort_calls == [sid]


@pytest.mark.asyncio
async def test_abort_unknown_session_returns_not_found() -> None:
    server = ACPServer(FakeBackend(), token="secret")
    out = await _serve(
        server,
        [_hello(), _enc({"type": "request", "id": "a", "request": {"command": "abort", "sessionId": "nope"}})],
    )
    reply = next(m for m in out if m.type == "response")
    assert not reply.ok
    assert reply.error.code == "not_found"


@pytest.mark.asyncio
async def test_abort_mid_turn_cancels_active_prompt() -> None:
    # The blocking prompt must start before the abort arrives, so use a
    # queue-backed connection that feeds frames dynamically.
    started = asyncio.Event()
    block = asyncio.Event()
    backend = FakeBackend(block=block, started=started)
    sid = await _create_session(backend)

    server = ACPServer(backend, token="secret")
    conn = _QueueConn()
    serve_task = asyncio.create_task(server.serve_connection(conn))
    conn.feed(_hello())
    conn.feed(
        _enc(
            {
                "type": "request",
                "id": "p",
                "request": {"command": "prompt", "sessionId": sid, "text": "hi"},
            }
        )
    )
    # Wait until the backend has emitted progress and is now blocking.
    await asyncio.wait_for(started.wait(), timeout=2.0)
    # Now send the abort — the prompt task is mid-turn.
    conn.feed(
        _enc({"type": "request", "id": "a", "request": {"command": "abort", "sessionId": sid}})
    )
    conn.feed_eof()
    await serve_task

    msgs = _decode(conn)
    responses = [m for m in msgs if m.type == "response"]
    prompt_resp = next(r for r in responses if r.id == "p")
    abort_resp = next(r for r in responses if r.id == "a")
    # Both land as ok=True.
    assert prompt_resp.ok and prompt_resp.result.command == "prompt"
    assert abort_resp.ok and abort_resp.result.command == "abort"
    # The prompt response is sent before the abort response (abort awaits
    # the prompt task before sending its own response).
    assert msgs.index(prompt_resp) < msgs.index(abort_resp)
    assert backend.abort_calls == [sid]


@pytest.mark.asyncio
async def test_second_prompt_while_turn_active_returns_busy() -> None:
    started = asyncio.Event()
    block = asyncio.Event()
    backend = FakeBackend(block=block, started=started)
    sid = await _create_session(backend)

    server = ACPServer(backend, token="secret")
    conn = _QueueConn()
    serve_task = asyncio.create_task(server.serve_connection(conn))
    conn.feed(_hello())
    conn.feed(
        _enc(
            {
                "type": "request",
                "id": "p1",
                "request": {"command": "prompt", "sessionId": sid, "text": "hi"},
            }
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2.0)
    # A second prompt for the same session while the first is active → busy.
    conn.feed(
        _enc(
            {
                "type": "request",
                "id": "p2",
                "request": {"command": "prompt", "sessionId": sid, "text": "hi2"},
            }
        )
    )
    conn.feed_eof()
    await serve_task

    msgs = _decode(conn)
    busy = next(m for m in msgs if m.type == "response" and m.id == "p2")
    assert not busy.ok
    assert busy.error.code == "busy"


@pytest.mark.asyncio
async def test_set_model_dispatch() -> None:
    backend = FakeBackend()
    sid = await _create_session(backend)

    out = await _serve(
        ACPServer(backend, token="secret"),
        [
            _hello(),
            _enc(
                {
                    "type": "request",
                    "id": "m",
                    "request": {
                        "command": "set_model",
                        "sessionId": sid,
                        "model": {"provider": "x", "id": "y"},
                    },
                }
            ),
        ],
    )
    reply = out[1]
    assert reply.ok and reply.result.command == "set_model"
    assert reply.result.session.model == ModelRef(provider="x", id="y")
    assert backend.set_model_calls == [(sid, ModelRef(provider="x", id="y"))]


@pytest.mark.asyncio
async def test_set_thinking_dispatch() -> None:
    backend = FakeBackend()
    sid = await _create_session(backend)

    out = await _serve(
        ACPServer(backend, token="secret"),
        [
            _hello(),
            _enc(
                {
                    "type": "request",
                    "id": "t",
                    "request": {
                        "command": "set_thinking",
                        "sessionId": sid,
                        "thinkingLevel": "high",
                    },
                }
            ),
        ],
    )
    reply = out[1]
    assert reply.ok and reply.result.command == "set_thinking"
    assert reply.result.session.thinkingLevel == "high"
    assert backend.set_thinking_calls == [(sid, "high")]
