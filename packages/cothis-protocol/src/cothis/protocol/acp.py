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

Scope: ``list`` / ``create`` / ``prompt`` are implemented; later work adds
``abort`` / ``set_model`` / ``set_thinking``. The remaining commands
(``attach`` / ``detach`` / ``steer``) are defined in the schema but answered
with ``invalid_request``. CBOR, persistence, snapshot revision/broadcast, and
an idle handshake timer are follow-ups.

``prompt`` runs as a background task so the read loop keeps consuming frames —
that is what makes same-connection ``abort`` reachable mid-turn. Each
connection owns a per-session in-flight registry; the connection that started
a turn is the one that may abort it (cross-connection abort is deferred).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from cothis.protocol.messages import (
    PROTOCOL_VERSION,
    AbortCommand,
    BackendError,
    ClientHello,
    ModelDescriptor,
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

#: How long :meth:`ACPServer.serve_connection` waits for in-flight prompt
#: tasks to finish after the read loop exits before force-cancelling them.
#: Bounds teardown so a stuck turn cannot keep a closing connection alive
#: indefinitely; a turn about to finish completes within the bound.
_DRAIN_TIMEOUT: float = 0.5


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
    async def models(self) -> list[ModelDescriptor]:
        """Models this server advertises in the handshake snapshot.

        Implementations return the model(s) they are configured to serve,
        enriched with the limits they can resolve. ``[]`` is honest for a
        backend that advertises nothing.
        """
        ...
    async def abort(self, session_id: str) -> SessionSnapshot:
        """Return the authoritative post-abort snapshot of a session.

        No-op-safe: the server cancels any active prompt task before calling
        this; the backend just reports state. Raises ``not_found`` (via
        ``_get``) for an unknown session — never for a missing *turn*.
        """
        ...
    async def set_model(
        self, session_id: str, model: ModelRef
    ) -> SessionSnapshot:
        """Re-target the session at a new model; returns the new snapshot.

        Takes effect on the next ``prompt``. May be received mid-turn (the
        read loop is free); the change applies to the next turn.
        """
        ...
    async def set_thinking(
        self, session_id: str, level: ThinkingLevel
    ) -> SessionSnapshot:
        """Update the session's thinking level; returns the new snapshot.

        Takes effect on the next ``prompt``.
        """
        ...


class ACPServer:
    """Drives one or more :class:`ByteConnection` clients through ACP."""

    def __init__(
        self,
        backend: SessionBackend,
        *,
        token: str,
        server_id: str = "cothis",
        max_frame_length: int | None = None,
    ) -> None:
        if not token:
            raise ValueError("ACPServer token must not be empty")
        self._backend = backend
        self._token_digest = hashlib.sha256(token.encode("utf-8")).digest()
        self.id = server_id
        self._max_frame = max_frame_length
        # revision is fixed at 0 (no broadcast machinery yet).
        self._revision = 0

    # ------------------------------------------------------------------ snapshot

    async def _server_snapshot(self) -> dict[str, Any]:
        sessions = await self._backend.list_sessions()
        models = await self._backend.models()
        return {
            "serverId": self.id,
            "protocolVersion": PROTOCOL_VERSION,
            "revision": self._revision,
            "sessions": [s.model_dump(mode="json") for s in sessions],
            "models": [m.model_dump(mode="json") for m in models],
        }

    # ------------------------------------------------------------------ serving

    async def serve_connection(self, conn: ByteConnection) -> None:
        """Run one client connection to completion (handshake → requests).

        A single decode loop handles both phases so that a chunk carrying
        ``hello`` *and* follow-on requests (clients may batch them in one
        write) is processed correctly — the hello promotes the connection to
        ``ready`` and the rest of the same chunk dispatch as requests.

        ``prompt`` runs as a background task (registered per session in
        ``inflight``) so the read loop keeps consuming frames — that is what
        makes same-connection ``abort`` reachable mid-turn. On exit, any
        still-running prompt task is given a short bounded wait to finalise
        (so its response lands) and then force-cancelled.
        """
        decoder = ClientMessageDecoder(max_frame_length=self._max_frame)
        stage = "awaitingHello"
        # Per-connection in-flight prompt registry: sessionId → active task.
        # The connection that started a turn owns cancelling it.
        inflight: dict[str, asyncio.Task[None]] = {}
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
                    await self._handle_request(conn, message, inflight)
        except (FrameError, ProtocolValidationError) as exc:
            logger.debug("connection failed framing/validation: %s", exc)
            await self._fail_handshake(
                conn,
                ProtocolError(
                    code="invalid_request", message="Malformed protocol frame"
                ),
            )
        finally:
            await self._drain_inflight(inflight)
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
        self,
        conn: ByteConnection,
        envelope: RequestEnvelope,
        inflight: dict[str, asyncio.Task[None]],
    ) -> None:
        req = envelope.request
        rid = envelope.id
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
            elif req.command == "set_model":
                snap = await self._backend.set_model(req.sessionId, req.model)
                result = {
                    "command": "set_model",
                    "session": snap.model_dump(mode="json"),
                }
            elif req.command == "set_thinking":
                snap = await self._backend.set_thinking(
                    req.sessionId, req.thinkingLevel
                )
                result = {
                    "command": "set_thinking",
                    "session": snap.model_dump(mode="json"),
                }
            elif req.command == "prompt":
                # prompt runs as a background task so the read loop stays free
                # to receive an abort on the same connection, mid-turn. The
                # task sends its own response and pops its registry entry.
                if req.sessionId in inflight:
                    await self._respond_error(
                        conn,
                        rid,
                        ProtocolError(
                            code="busy",
                            message=(
                                f"a turn is already active on session "
                                f"{req.sessionId!r}"
                            ),
                        ),
                    )
                    return
                task = asyncio.create_task(self._cmd_prompt(conn, rid, req, inflight))
                inflight[req.sessionId] = task
                return
            elif req.command == "abort":
                await self._cmd_abort(conn, rid, req, inflight)
                return
            else:
                raise BackendError(
                    ProtocolError(
                        code="invalid_request",
                        message=(
                            f"command '{req.command}' is not supported by this server"
                        ),
                    )
                )
        except BackendError as exc:
            await self._respond_error(conn, rid, exc.error)
            return
        except Exception:
            logger.exception("error executing command %s", req.command)
            await self._respond_error(
                conn,
                rid,
                ProtocolError(
                    code="invalid_request", message="Internal server error"
                ),
            )
            return
        await self._respond_ok(conn, rid, result)

    async def _cmd_prompt(
        self,
        conn: ByteConnection,
        rid: str,
        cmd: PromptCommand,
        inflight: dict[str, asyncio.Task[None]],
    ) -> None:
        """Run one prompt turn as a background task; send the response itself.

        Registered in ``inflight`` so an ``abort`` on the same session can
        cancel it. Pops its own entry in ``finally``. If cancelled mid-turn,
        the backend finalises the assistant item as ``aborted`` and returns
        the snapshot, which is sent as a normal ``ok=True`` response.
        """

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

        try:
            snap = await self._backend.prompt(cmd.sessionId, cmd.text, emit)
            await self._respond_ok(
                conn, rid, {"command": "prompt", "session": snap.model_dump(mode="json")}
            )
        except BackendError as exc:
            await self._respond_error(conn, rid, exc.error)
        except asyncio.CancelledError:
            # The backend swallows cancellation at its boundary (finalises the
            # assistant item as aborted and returns the snapshot); reaching
            # here means it did not — propagate so the task ends cancelled.
            raise
        except Exception:
            logger.exception("error executing prompt")
            await self._respond_error(
                conn,
                rid,
                ProtocolError(
                    code="invalid_request", message="Internal server error"
                ),
            )
        finally:
            inflight.pop(cmd.sessionId, None)

    async def _cmd_abort(
        self,
        conn: ByteConnection,
        rid: str,
        cmd: AbortCommand,
        inflight: dict[str, asyncio.Task[None]],
    ) -> None:
        """Abort an active turn on this connection; no-op-safe if none active.

        If a prompt task is registered for this session, cancel and await it
        (the backend finalises the assistant item as ``aborted`` and the task
        sends its own ``ok=True`` response). Then read the post-abort snapshot
        and send the ``abort`` response. Aborting a session with no active
        turn skips the cancellation and returns the current snapshot.
        """
        task = inflight.get(cmd.sessionId)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug(
                    "aborted prompt task raised during finalisation", exc_info=True
                )
            inflight.pop(cmd.sessionId, None)
        try:
            snap = await self._backend.abort(cmd.sessionId)
        except BackendError as exc:
            await self._respond_error(conn, rid, exc.error)
            return
        except Exception:
            logger.exception("error executing abort")
            await self._respond_error(
                conn,
                rid,
                ProtocolError(
                    code="invalid_request", message="Internal server error"
                ),
            )
            return
        await self._respond_ok(
            conn, rid, {"command": "abort", "session": snap.model_dump(mode="json")}
        )

    async def _drain_inflight(
        self, inflight: dict[str, asyncio.Task[None]]
    ) -> None:
        """Let in-flight prompt tasks finalise before the connection closes.

        A short bounded wait lets a task that is about to finish (e.g. the
        backend returned a snapshot and the task is sending its response)
        complete cleanly so its response lands. Tasks still running after the
        wait are cancelled so the connection cannot leak a task that outlives
        it.
        """
        if not inflight:
            return
        tasks = list(inflight.values())
        done, pending = await asyncio.wait(tasks, timeout=_DRAIN_TIMEOUT)
        if not pending:
            return
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug(
                    "in-flight prompt task raised during drain", exc_info=True
                )

    # ------------------------------------------------------------------ low-level

    async def _respond_ok(
        self, conn: ByteConnection, rid: str, result: dict[str, Any]
    ) -> bool:
        return await self._send(
            conn, {"type": "response", "id": rid, "ok": True, "result": result}
        )

    async def _respond_error(
        self, conn: ByteConnection, rid: str, error: ProtocolError
    ) -> bool:
        return await self._send(
            conn,
            {
                "type": "response",
                "id": rid,
                "ok": False,
                "error": error.model_dump(mode="json"),
            },
        )

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
