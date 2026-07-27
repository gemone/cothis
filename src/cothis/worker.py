"""``cothis.worker`` — SessionWorker process entrypoint (#225).

Owns one ``Agent`` + drives a WebSocket transport (``cothis.ws``) that accepts
control messages (``run_turn`` / ``attach_input`` / ``detach_input`` /
``shutdown`` / ``ping`` / ``resolve_ask``) and emits stream messages
(``assistant_delta`` / ``tool_call_started`` / ``tool_call_result_pointer`` /
``ask_user_request`` / ``pong`` / ``error``).

Handshake requires a valid bearer token on the ``Authorization`` header.
Missing or wrong token → HTTP 401, connection rejected. The token is generated
via ``secrets.token_urlsafe``; the Supervisor (#227) receives it from the spawn
call and passes it back to the TUI via an IPC channel.

The worker talks to its WS surface through the ``WSTransport`` seam
(``cothis.ws``), not to ``websockets`` directly (#248). The transport is
injectable so the worker's message-handling logic is unit-testable with a
mock transport — no socket bound. The only concurrency primitives the worker
reaches for are ``anyio.fail_after`` (backend-neutral cancel scope for the
turn timeout) and ``asyncio.create_task`` (background dispatch of ``run_turn``
so control messages stay readable during a turn, #316). See ADR-0017 §6.

Interactive ``ask_user`` flow (#229 B+D-3): a tool inside ``Agent.run_stream``
calls ``Agent._ask_user`` → the worker-installed ``_on_ask_user`` callback
fires synchronously → an ``ask_user_request`` frame is scheduled to the active
WS client. The client replies with ``resolve_ask``; the worker's handler
forwards it to ``Agent.resolve_ask`` which resolves the Future the tool is
awaiting. The agent knows nothing about the WS surface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from typing import TYPE_CHECKING, Any

import anyio

from cothis.agent import (
    Agent,
    AskUserRequestEvent,
    ContentDelta,
    ToolCallEvent,
    ToolResultEvent,
)
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
        # #229 B+D-3: conn the active turn's tools should send ask_user_request
        # to. Set in _handle_conn, cleared on exit. The callback installed
        # below reads this so the agent (which knows nothing about the WS
        # surface) can emit the event.
        self._active_conn: Connection | None = None
        # Sync callback the agent calls from _ask_user. Reads _active_conn at
        # call time (a fresh closure per worker would also work; method ref
        # is simpler and lets tests monkeypatch if needed). ``setattr`` keeps
        # the dynamic-attribute installation out of the ``Agent`` class's
        # typed surface (the agent reads it via ``getattr``).
        setattr(agent, "_on_ask_user", self._emit_ask_user_request)

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
        if not secrets.compare_digest(auth[len("Bearer ") :], self._token):
            return _http_401()
        return None

    def _emit_ask_user_request(self, event: AskUserRequestEvent) -> None:
        """Forward an ``AskUserRequestEvent`` to the active WS client.

        Called synchronously by ``Agent._ask_user`` from inside a tool (which
        runs inside ``_stream_turn``). The send itself is async, so we
        schedule it on the running loop with ``create_task`` — fire-and-
        forget. The agent then awaits its Future; when the client replies
        with ``resolve_ask``, ``_dispatch`` calls ``agent.resolve_ask`` and
        the tool unblocks.

        If no conn is active, the request is dropped (with a warning). The
        agent's Future will simply never resolve; the turn will hit
        ``_TURN_TIMEOUT_S`` and report a timeout. That's the correct
        behaviour when nobody is listening — better than crashing the tool.
        """
        conn = self._active_conn
        if conn is None:
            logger.warning(
                "ask_user event with no active conn (ask_id=%s); "
                "Future will not resolve — turn will time out",
                event.ask_id,
            )
            return

        # ``conn.send`` is ``Awaitable[None]`` (per the ``Connection`` protocol);
        # ``create_task`` wants a ``Coroutine``. Wrap the send in a local
        # async function so the types line up — and so the work is clearly
        # its own schedulable unit.
        async def _send_ask() -> None:
            await conn.send(
                json.dumps(
                    {
                        "type": "ask_user_request",
                        "ask_id": event.ask_id,
                        "prompt": event.prompt,
                        "choices": event.choices,
                    }
                )
            )

        asyncio.create_task(_send_ask())

    async def _handle_conn(self, conn: Connection) -> None:
        """Dispatch control messages until the connection closes.

        ``run_turn`` is dispatched as a background task so control
        messages (``resolve_ask``, ``ping``, ``shutdown``) can be
        processed while the turn runs (#229 deadlock prevention).
        Without this, a tool that blocks on a Future (waiting for
        ``resolve_ask``) would deadlock — the ``async for`` loop can't
        read the next message until ``_stream_turn`` returns.
        """
        self._active_turn: asyncio.Task[None] | None = None
        self._active_conn = conn
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
                typ = msg["type"]
                if typ == "run_turn":
                    # Cancel any active turn before starting a new one
                    # (the client shouldn't send concurrent turns, but
                    # guard against it anyway).
                    if self._active_turn and not self._active_turn.done():
                        self._active_turn.cancel()
                    self._active_turn = asyncio.create_task(self._dispatch(conn, msg))
                else:
                    await self._dispatch(conn, msg)
                if typ == "shutdown":
                    if self._active_turn and not self._active_turn.done():
                        self._active_turn.cancel()
                    return
        except Exception as exc:  # noqa: BLE001
            logger.warning("SessionWorker connection error: %s", exc)
        finally:
            self._active_conn = None
            # cothis: cancel + await the in-flight turn on disconnect (#396).
            # Without this, a client that drops mid-turn leaves the background
            # turn task running (driving the agent for a dead client) and
            # un-awaited (asyncio "Task exception was never retrieved").
            if self._active_turn is not None and not self._active_turn.done():
                self._active_turn.cancel()
                try:
                    await self._active_turn
                except (asyncio.CancelledError, Exception):
                    pass

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
        elif typ == "resolve_ask":
            # #229 B+D-3: client's reply resolves the agent's pending
            # Future (created by ``_ask_user``). ``agent.resolve_ask`` is
            # a no-op if the ask_id is unknown or already done, so an
            # orphan / duplicate reply is silently absorbed. Skip if the
            # payload is malformed — agent has nothing to resolve to.
            ask_id = msg.get("ask_id")
            if isinstance(ask_id, str):
                self._agent.resolve_ask(ask_id, msg.get("value"))
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
                    if isinstance(event, ContentDelta):
                        await conn.send(
                            json.dumps(
                                {
                                    "type": "assistant_delta",
                                    "kind": event.kind,
                                    "text": event.text,
                                }
                            )
                        )
                    elif isinstance(event, ToolCallEvent):
                        await conn.send(
                            json.dumps(
                                {
                                    "type": "tool_call_started",
                                    "tool": event.name,
                                    "arguments": event.arguments,
                                    "call_id": event.call_id,
                                }
                            )
                        )
                    elif isinstance(event, ToolResultEvent):
                        await conn.send(
                            json.dumps(
                                {
                                    "type": "tool_call_result_pointer",
                                    "tool": event.tool,
                                    "is_error": event.is_error,
                                    "duration_ms": event.duration_ms,
                                    "pointer": event.result_pointer,
                                    "call_id": event.call_id,
                                }
                            )
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
