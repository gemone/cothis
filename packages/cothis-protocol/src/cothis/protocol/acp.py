"""``cothis.protocol.acp`` — the Agent Client Protocol server core.

Transport-neutral: the server speaks the wire format
(:mod:`cothis.protocol.wire`) over a :class:`ByteConnection` — an object
that can send bytes, close, and be iterated for inbound byte chunks. That is
the same shape as ``cothis.ws.Connection``, so a WebSocket adapter (or a
stdio adapter) plugs in without touching this module.

The agent itself is behind a :class:`SessionBackend` interface so the server
is unit-testable with a fake backend; the production bridge that drives
``Agent.run_stream`` lives in the agent package.

Lifecycle per connection: read frames → the first must be ``hello``; the
server validates the bearer token (timing-safe) and protocol version, replies
``hello`` with a :class:`ServerSnapshot` or ``hello_error`` + close. After
that each ``request`` runs a command and returns a ``response``; ``prompt``
streams ``session_progress`` events as the turn runs.

I9 scope: ``list`` / ``create`` / ``prompt`` are implemented; the remaining
commands are defined in the schema but answered with ``invalid_request``.
CBOR, persistence, snapshot revision/broadcast, and an idle handshake timer
are follow-ups.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from cothis.protocol.messages import (
    PROTOCOL_VERSION,
    BackendError,
    ClientHello,
    ModelRef,
    PromptCommand,
    ProtocolError,
    RequestEnvelope,
    ServerSnapshot,
    SessionSnapshot,
    SessionSummary,
    ThinkingLevel,
    TranscriptProgress,
)
from cothis.protocol.wire import (
    ClientMessageDecoder,
    FrameError,
    ProtocolValidationError,
    encode_server_message,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

#: Callback a backend invokes to stream one progress update during a turn.
ProgressEmitter = Callable[[TranscriptProgress], "Awaitable[None]"]


@runtime_checkable
class ByteConnection(Protocol):
    """A connected, ordered byte sink + inbound chunk source.

    Mirrors ``cothis.ws.Connection`` so either transport can back the server.
    """

    closed: bool

    def send(self, chunk: bytes) -> Awaitable[None]: ...
    def close(self, final_chunk: bytes | None = None) -> Awaitable[None]: ...
    def __aiter__(self) -> AsyncIterator[bytes]: ...


@runtime_checkable
class SessionBackend(Protocol):
    """The agent-facing surface the server drives.

    Implementations are async. ``prompt`` receives an *emit* callback to
    stream :class:`TranscriptProgress` updates as the turn runs and returns
    the authoritative post-turn snapshot. A backend signals a protocol-level
    failure by raising :class:`ProtocolError`.
    """

    async def list_sessions(self) -> list[SessionSummary]: ...
    async def create_session(
        self,
        cwd: str | None,
        name: str | None,
        model: ModelRef | None,
        thinking_level: ThinkingLevel | None,
    ) -> SessionSnapshot: ...
    async def prompt(
        self, session_id: str, text: str, emit: ProgressEmitter
    ) -> SessionSnapshot: ...


class ACPServer:
    """Drives one or more :class:`ByteConnection` clients through ACP."""

    def __init__(
        self,
        backend: SessionBackend,
        *,
        token: str,
        server_id: str = "cothis",
        models: list[Any] | None = None,
        max_frame_length: int | None = None,
    ) -> None:
        if not token:
            raise ValueError("ACPServer token must not be empty")
        self._backend = backend
        self._token_digest = hashlib.sha256(token.encode("utf-8")).digest()
        self.id = server_id
        self._models = list(models or [])
        self._max_frame = max_frame_length
        # revision is fixed at 0 in I9 (no broadcast machinery yet).
        self._revision = 0

    # ------------------------------------------------------------------ snapshot

    async def _server_snapshot(self) -> dict[str, Any]:
        sessions = await self._backend.list_sessions()
        return {
            "serverId": self.id,
            "protocolVersion": PROTOCOL_VERSION,
            "revision": self._revision,
            "sessions": [s.model_dump(mode="json") for s in sessions],
            "models": self._models,
        }

    # ------------------------------------------------------------------ serving

    async def serve_connection(self, conn: ByteConnection) -> None:
        """Run one client connection to completion (handshake → requests).

        A single decode loop handles both phases so that a chunk carrying
        ``hello`` *and* follow-on requests (clients may batch them in one
        write) is processed correctly — the hello promotes the connection to
        ``ready`` and the rest of the same chunk dispatch as requests.
        """
        decoder = ClientMessageDecoder(max_frame_length=self._max_frame)
        stage = "awaitingHello"
        try:
            async for chunk in conn:
                for message in decoder.push(chunk):
                    if stage == "awaitingHello":
                        if not isinstance(message, ClientHello):
                            await self._fail_handshake(
                                conn,
                                ProtocolError(
                                    code="invalid_request",
                                    message="The first client message must be hello",
                                ),
                            )
                            return
                        if not await self._handshake(conn, message):
                            return
                        stage = "ready"
                        continue
                    # ready
                    if isinstance(message, ClientHello):
                        await self._fail_with(
                            conn,
                            ProtocolError(
                                code="invalid_request",
                                message="hello may only be sent as the first message",
                            ),
                        )
                        return
                    await self._handle_request(conn, message)
        except (FrameError, ProtocolValidationError) as exc:
            logger.debug("connection failed framing/validation: %s", exc)
            await self._fail_handshake(
                conn,
                ProtocolError(
                    code="invalid_request", message="Malformed protocol frame"
                ),
            )
        finally:
            if not conn.closed:
                await conn.close()

    # ------------------------------------------------------------------ handshake

    def _authenticate(self, hello: ClientHello) -> bool:
        digest = hashlib.sha256(hello.token.encode("utf-8")).digest()
        return hmac.compare_digest(digest, self._token_digest)

    async def _handshake(self, conn: ByteConnection, hello: ClientHello) -> bool:
        if not self._authenticate(hello):
            await self._fail_handshake(
                conn, ProtocolError(code="auth", message="Authentication failed")
            )
            return False
        if hello.version != PROTOCOL_VERSION:
            await self._fail_handshake(
                conn,
                ProtocolError(
                    code="version",
                    message=(
                        f"Unsupported protocol version {hello.version}; "
                        f"expected {PROTOCOL_VERSION}"
                    ),
                ),
            )
            return False
        snapshot = await self._server_snapshot()
        return await self._send(
            conn,
            {
                "type": "hello",
                "version": PROTOCOL_VERSION,
                "connectionId": self.id,
                "snapshot": snapshot,
            },
        )

    async def _fail_handshake(self, conn: ByteConnection, error: ProtocolError) -> None:
        frame = encode_server_message(
            {"type": "hello_error", "error": error.model_dump(mode="json")},
            max_frame_length=self._max_frame,
        )
        try:
            await conn.close(frame)
        except Exception:  # closing errors are not actionable
            logger.debug("error closing connection after hello_error", exc_info=True)

    async def _fail_with(self, conn: ByteConnection, error: ProtocolError) -> None:
        if not conn.closed:
            await conn.close(
                encode_server_message(
                    {"type": "hello_error", "error": error.model_dump(mode="json")},
                    max_frame_length=self._max_frame,
                )
            )

    # ------------------------------------------------------------------ requests

    async def _handle_request(
        self, conn: ByteConnection, envelope: RequestEnvelope
    ) -> None:
        req = envelope.request
        try:
            if req.command == "list":
                sessions = await self._backend.list_sessions()
                result: Any = {
                    "command": "list",
                    "sessions": [s.model_dump(mode="json") for s in sessions],
                }
            elif req.command == "create":
                snap = await self._backend.create_session(
                    req.cwd, req.name, req.model, req.thinkingLevel
                )
                result = {"command": "create", "session": snap.model_dump(mode="json")}
            elif req.command == "prompt":
                snap = await self._cmd_prompt(conn, req)
                result = {"command": "prompt", "session": snap.model_dump(mode="json")}
            else:
                raise BackendError(
                    ProtocolError(
                        code="invalid_request",
                        message=f"command '{req.command}' is not supported by this server",
                    )
                )
        except BackendError as exc:
            await self._send(
                conn,
                {
                    "type": "response",
                    "id": envelope.id,
                    "ok": False,
                    "error": exc.error.model_dump(mode="json"),
                },
            )
            return
        except Exception:
            logger.exception("error executing command %s", req.command)
            await self._send(
                conn,
                {
                    "type": "response",
                    "id": envelope.id,
                    "ok": False,
                    "error": {
                        "code": "invalid_request",
                        "message": "Internal server error",
                    },
                },
            )
            return
        await self._send(
            conn, {"type": "response", "id": envelope.id, "ok": True, "result": result}
        )

    async def _cmd_prompt(
        self, conn: ByteConnection, cmd: PromptCommand
    ) -> SessionSnapshot:
        async def emit(progress: TranscriptProgress) -> None:
            await self._send(
                conn,
                {
                    "type": "event",
                    "event": {
                        "type": "session_progress",
                        "sessionId": cmd.sessionId,
                        "progress": progress.model_dump(mode="json"),
                    },
                },
            )

        return await self._backend.prompt(cmd.sessionId, cmd.text, emit)

    # ------------------------------------------------------------------ low-level

    async def _send(self, conn: ByteConnection, message: dict[str, Any]) -> bool:
        if conn.closed:
            return False
        try:
            frame = encode_server_message(message, max_frame_length=self._max_frame)
        except ProtocolValidationError:
            logger.exception("failed to encode outgoing message")
            if not conn.closed:
                await conn.close()
            return False
        try:
            await conn.send(frame)
            return True
        except Exception:
            logger.debug("send failed; closing connection", exc_info=True)
            if not conn.closed:
                await conn.close()
            return False


__all__ = [
    "ACPServer",
    "ByteConnection",
    "SessionBackend",
    "ProgressEmitter",
]
