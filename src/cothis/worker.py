"""``cothis.worker`` — SessionWorker process entrypoint (#225).

Owns one ``Agent`` + drives a WebSocket transport (``cothis.ws``) that accepts
control messages (``run_turn`` / ``attach_input`` / ``detach_input`` /
``shutdown`` / ``ping``) and emits stream messages (``assistant_delta`` /
``tool_call_started`` / ``tool_call_result_pointer`` / ``pong`` / ``error``).

Handshake requires a valid bearer token on the ``Authorization`` header.
Missing or wrong token → HTTP 401, connection rejected. The token is generated
via ``secrets.token_urlsafe``; the Supervisor (#227) receives it from the spawn
call and passes it back to the TUI via an IPC channel.

The worker talks to its WS surface through the ``WSTransport`` seam
(``cothis.ws``), not to ``websockets`` directly (#248). The transport is
injectable so the worker's message-handling logic is unit-testable with a
mock transport — no socket bound. The only concurrency primitive the worker
reaches for is ``anyio.fail_after`` (backend-neutral cancel scope) for the
turn timeout; it names no ``asyncio`` symbol. See ADR-0017 §6.
"""

from __future__ import annotations

import json
import logging
import secrets
from typing import TYPE_CHECKING, Any

import anyio

from cothis.agent import Agent, ToolCallEvent
from cothis.ws import (
    AuthCheck,
    Connection,
    WebSocketServerTransport,
    WSTransport,
    _http_401,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

_TURN_TIMEOUT_S = 300


class SessionWorker:
    """One per session. Owns Agent + drives a WS transport.

    Lifecycle:

    - ``__init__`` generates the bearer token; no network yet.
    - ``start()`` binds the transport (random loopback port), returns the URI.
    - ``serve_forever()`` runs the accept loop until ``shutdown`` arrives or
      ``stop()`` is called.
    - ``stop()`` signals the transport to shut down and closes ``Agent``.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        transport: WSTransport | None = None,
        host: str = "127.0.0.1",
    ) -> None:
        self._agent = agent
        # cothis: default transport is the ``websockets`` adapter; tests pass a
        # fake so the dispatch/timeout logic runs without a socket (#248).
        self._transport: WSTransport = transport or WebSocketServerTransport(host=host)
        self._token = secrets.token_urlsafe(32)
        self._bound = False

    @property
    def token(self) -> str:
        """The bearer token the client must present."""
        return self._token

    @property
    def uri(self) -> str | None:
        """WS URI once ``start`` has bound the port; ``None`` otherwise."""
        return self._transport.uri

    async def start(self) -> str:
        """Bind the transport; return the URI. Idempotent.

        Delegates socket binding to ``transport.bind``; the URI is available
        the moment ``bind`` returns.
        """
        if not self._bound:
            await self._transport.bind(self._handle_conn, self._check_auth)
            self._bound = True
        uri = self._transport.uri
        if uri is None:  # pragma: no cover - transport violated its contract
            raise RuntimeError("transport bound but uri is None")
        return uri

    def _check_auth(self, request: Any):
        """Handshake gate: ``Authorization: Bearer *** required.

        Returns ``None`` to accept, an HTTP ``Response`` (401) to reject. The
        transport calls this synchronously during the WS handshake.
        """
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return _http_401()
        # Constant-time compare; token length is bounded (32+ chars).
        if not secrets.compare_digest(auth[len("Bearer "):], self._token):
            return _http_401()
        return None

    async def _handle_conn(self, conn: Connection) -> None:
        """Dispatch control messages until the connection closes."""
        try:
            async for raw in conn:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await conn.send(
                        json.dumps({"type": "error", "message": "invalid JSON"})
                    )
                    continue
                if not isinstance(msg, dict) or "type" not in msg:
                    await conn.send(
                        json.dumps({"type": "error", "message": "expected {type: ...}"})
                    )
                    continue
                await self._dispatch(conn, msg)
                if msg["type"] == "shutdown":
                    return
        except Exception as exc:  # noqa: BLE001
            logger.warning("SessionWorker connection error: %s", exc)

    async def _dispatch(self, conn: Connection, msg: dict[str, Any]) -> None:
        """One control message → one or more WS responses."""
        typ = msg["type"]
        if typ == "ping":
            await conn.send(json.dumps({"type": "pong"}))
        elif typ == "shutdown":
            await conn.close()
            self._transport.request_shutdown()
        elif typ == "run_turn":
            await self._stream_turn(conn, msg.get("prompt", ""))
        elif typ in ("attach_input", "detach_input"):
            # Real terminal attach lands with #230; accept + ignore for now.
            logger.debug("SessionWorker got %r (terminal attach deferred)", typ)
        else:
            await conn.send(
                json.dumps({"type": "error", "message": f"unknown type: {typ!r}"})
            )

    async def _stream_turn(self, conn: Connection, prompt: str) -> None:
        """Drive ``Agent.run_stream`` and forward each event to the client.

        Bounded by ``_TURN_TIMEOUT_S`` (via ``anyio.fail_after`` — backend-
        neutral cancel scope, #248) so a stuck tool or model stream can't hold
        the connection indefinitely. Errors are logged server-side + a generic
        ``"internal error"`` goes to the client (loopback-only is not a license
        to leak exception details).
        """
        try:
            with anyio.fail_after(_TURN_TIMEOUT_S):
                async for event in self._agent.run_stream(prompt):
                    if isinstance(event, str):
                        await conn.send(
                            json.dumps({"type": "assistant_delta", "text": event})
                        )
                    elif isinstance(event, ToolCallEvent):
                        await conn.send(
                            json.dumps({
                                "type": "tool_call_started",
                                "tool": event.name,
                                "arguments": event.arguments,
                            })
                        )
        except TimeoutError:
            logger.warning("SessionWorker turn timed out after %ds", _TURN_TIMEOUT_S)
            await conn.send(json.dumps({"type": "error", "message": "turn timeout"}))
        except Exception:  # noqa: BLE001
            logger.exception("Agent.run_stream failed")
            await conn.send(json.dumps({"type": "error", "message": "internal error"}))

    async def serve_forever(self) -> None:
        """Run the accept loop until ``shutdown`` arrives or ``stop()``."""
        if not self._bound:
            await self.start()
        await self._transport.serve()

    async def stop(self) -> None:
        """Close the transport + Agent. Idempotent."""
        self._transport.request_shutdown()
        aclose = getattr(self._agent, "aclose", None)
        if aclose is not None:
            await aclose()


__all__ = ["AuthCheck", "SessionWorker", "WSTransport"]
