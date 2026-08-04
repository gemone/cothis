"""Tests for ``cothis.protocol.acp.ACPServer`` — handshake + command dispatch.

Hermetic: a fake in-memory ``ByteConnection`` captures sent frames; a fake
``SessionBackend`` records calls and (for ``prompt``) streams one delta. No
sockets, no network, no real agent.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from cothis.protocol.acp import ACPServer
from cothis.protocol.messages import (
    PROTOCOL_VERSION,
    AssistantDelta,
    BackendError,
    ModelRef,
    ProtocolError,
    SessionSnapshot,
    SessionSummary,
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


class FakeBackend:
    def __init__(self) -> None:
        self.created: list[SessionSnapshot] = []
        self.prompt_calls: list[tuple[str, str]] = []

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
        snap = next(s for s in self.created if s.id == session_id)
        return snap.model_copy(update={"revision": 1})


def _enc(d: dict) -> bytes:
    return encode_client_message(d)


def _decode(conn: FakeConnection) -> list:
    return ServerMessageDecoder().push(bytes(conn.sent))


async def _serve(server: ACPServer, frames: list[bytes]) -> list:
    conn = FakeConnection(frames)
    await server.serve_connection(conn)
    return _decode(conn)


def _hello(token: str = "secret", version: int = PROTOCOL_VERSION) -> bytes:
    return _enc({"type": "hello", "version": version, "token": token})


@pytest.mark.asyncio
async def test_handshake_success_sends_server_hello_with_snapshot() -> None:
    server = ACPServer(FakeBackend(), token="secret")
    [msg] = await _serve(server, [_hello()])
    assert msg.type == "hello"
    assert msg.version == PROTOCOL_VERSION
    assert msg.snapshot.serverId == "cothis"
    assert msg.snapshot.sessions == []


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
    server = ACPServer(FakeBackend(), token="secret")
    out = await _serve(
        server,
        [_hello(), _enc({"type": "request", "id": "r", "request": {"command": "abort", "sessionId": "s"}})],
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
