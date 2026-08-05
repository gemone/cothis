"""Tests for ``cothis.protocol.acp_client.ACPClient`` — client over ACP.

Hermetic, in-process loopback: a pair of :class:`ByteConnection`s wired
together on one event loop so a real :class:`ACPClient` drives a real
:class:`ACPServer` + ``FakeBackend``. No subprocess, no sockets, no network.

The loopback pair mirrors the real transport contract: ``send`` on one side
delivers bytes to the other side's inbound iterator; ``close`` flushes a
final chunk then signals EOF.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cothis.protocol.acp import ACPServer
from cothis.protocol.acp_client import (
    ACPClient,
    ACPHandshakeError,
    ACPRequestError,
)
from cothis.protocol.messages import (
    PROTOCOL_VERSION,
    AssistantDelta,
    BackendError,
    ModelDescriptor,
    ModelRef,
    ProtocolError,
    SessionSnapshot,
    SessionSummary,
)

# A sentinel the loopback uses to signal EOF on a peer's inbound queue. Kept
# distinct from any bytes chunk so the iterator can stop cleanly.
_EOF = object()


class _LoopbackConn:
    """One side of an in-memory :class:`ByteConnection` pair."""

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self._peer: _LoopbackConn | None = None
        self.closed = False

    def _set_peer(self, peer: _LoopbackConn) -> None:
        self._peer = peer

    async def send(self, chunk: bytes) -> None:
        assert self._peer is not None
        await self._peer._inbox.put(chunk)

    async def close(self, final_chunk: bytes | None = None) -> None:
        assert self._peer is not None
        if final_chunk is not None:
            await self._peer._inbox.put(final_chunk)
        await self._peer._inbox.put(_EOF)
        self.closed = True

    def __aiter__(self) -> _LoopbackConn:
        return self

    async def __anext__(self) -> bytes:
        item = await self._inbox.get()
        if item is _EOF:
            raise StopAsyncIteration
        return item


class _Loopback:
    """A bidirectional byte pipe: ``server`` <-> ``client``."""

    def __init__(self) -> None:
        self.server = _LoopbackConn()
        self.client = _LoopbackConn()
        self.server._set_peer(self.client)
        self.client._set_peer(self.server)


class _FakeBackend:
    """In-memory backend: records calls, streams one delta on ``prompt``."""

    def __init__(self) -> None:
        self.created: list[SessionSnapshot] = []
        self.prompt_calls: list[tuple[str, str]] = []
        self._advertised = ModelDescriptor(
            provider="openrouter",
            id="openai/gpt-oss-120b",
            maxOutputTokens=32768,
            contextWindow=131072,
        )

    async def models(self) -> list[ModelDescriptor]:
        return [self._advertised]

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
            id="sess-1",
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

    async def prompt(
        self, session_id: str, text: str, emit: Any
    ) -> SessionSnapshot:
        self.prompt_calls.append((session_id, text))
        if not any(s.id == session_id for s in self.created):
            raise BackendError(
                ProtocolError(
                    code="not_found", message=f"session {session_id!r} not found"
                )
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


async def _serve_in_background(server: ACPServer, loop: _Loopback) -> asyncio.Task:
    task = asyncio.create_task(server.serve_connection(loop.server))
    return task


@pytest.mark.asyncio
async def test_client_handshake_success_returns_advertised_snapshot() -> None:
    backend = _FakeBackend()
    server = ACPServer(backend, token="secret")
    loop = _Loopback()
    server_task = await _serve_in_background(server, loop)
    try:
        client = ACPClient(loop.client, token="secret")
        snapshot = await client.connect()
        assert snapshot.serverId == "cothis"
        assert snapshot.protocolVersion == PROTOCOL_VERSION
        # The advertised model is the backend's configured one.
        assert snapshot.models == await backend.models()
        assert client.connection_id == "cothis"
        await client.aclose()
    finally:
        await server_task


@pytest.mark.asyncio
async def test_client_handshake_auth_failure_raises_typed_error() -> None:
    server = ACPServer(_FakeBackend(), token="secret")
    loop = _Loopback()
    server_task = await _serve_in_background(server, loop)
    try:
        client = ACPClient(loop.client, token="WRONG")
        with pytest.raises(ACPHandshakeError) as exc_info:
            await client.connect()
        assert exc_info.value.error.code == "auth"
    finally:
        await server_task


@pytest.mark.asyncio
async def test_client_create_then_list_round_trip() -> None:
    backend = _FakeBackend()
    server = ACPServer(backend, token="secret")
    loop = _Loopback()
    server_task = await _serve_in_background(server, loop)
    try:
        client = ACPClient(loop.client, token="secret")
        await client.connect()
        created = await client.create_session(cwd="/tmp")
        assert created.id == "sess-1"
        sessions = await client.list_sessions()
        assert [s.id for s in sessions] == ["sess-1"]
        await client.aclose()
    finally:
        await server_task


@pytest.mark.asyncio
async def test_client_prompt_streams_progress_then_completes() -> None:
    backend = _FakeBackend()
    server = ACPServer(backend, token="secret")
    loop = _Loopback()
    server_task = await _serve_in_background(server, loop)
    try:
        client = ACPClient(loop.client, token="secret")
        await client.connect()
        created = await client.create_session(cwd="/")
        progress: list[Any] = []
        async for item in client.prompt(created.id, "hello"):
            progress.append(item)
        # The one delta streamed by the backend is delivered in order.
        assert len(progress) == 1
        assert progress[0].type == "assistant_delta"
        assert progress[0].delta == "Hi"
        assert backend.prompt_calls == [(created.id, "hello")]
        # The post-turn snapshot from the ok response is exposed.
        assert client.last_prompt_snapshot is not None
        assert client.last_prompt_snapshot.id == created.id
        assert client.last_prompt_snapshot.revision == 1
        await client.aclose()
    finally:
        await server_task


@pytest.mark.asyncio
async def test_client_prompt_unknown_session_surfaces_request_error() -> None:
    server = ACPServer(_FakeBackend(), token="secret")
    loop = _Loopback()
    server_task = await _serve_in_background(server, loop)
    try:
        client = ACPClient(loop.client, token="secret")
        await client.connect()
        with pytest.raises(ACPRequestError) as exc_info:
            async for _ in client.prompt("nope", "x"):
                pass
        assert exc_info.value.error.code == "not_found"
        await client.aclose()
    finally:
        await server_task
