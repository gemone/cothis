"""``cothis.worker`` — SessionWorker process entrypoint (#225).

Owns one ``Agent`` + drives a WebSocket transport (``cothis.ws``) that accepts
control messages (``run_turn`` / ``attach_input`` / ``detach_input`` /
``shutdown`` / ``ping`` / ``resolve_ask`` / ``interrupt_turn``) and emits
stream messages (``assistant_delta`` / ``tool_call_started`` /
``tool_call_result_pointer`` / ``ask_user_request`` / ``pong`` / ``error`` /
``turn_started`` / ``turn_finished``).

``turn_started`` opens a turn; ``turn_finished`` (carrying the post-turn
model / session_id / context pressure / active_skills snapshot) closes it
on every exit path (normal end, timeout, error, interrupt). The TUI relies
on ``turn_finished`` to reconcile its run-state to idle and refresh the
status footer. ``interrupt_turn`` cancels ``_active_turn`` — the same
abort primitive already used by run_turn-supersede (#316) and disconnect
(#396) — exposed behind a control message rather than a parallel path.

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
        # Interrupt-vs-timeout tie-breaker. ``interrupt_turn`` flips
        # this to ``True`` BEFORE cancelling ``_active_turn``; ``_stream_turn``
        # resets it to ``False`` at turn start. A ``CancelledError`` reaching
        # ``_stream_turn`` with the flag set is an intentional interrupt (no
        # error frame); without it, the cancel is treated as a turn timeout.
        # See ``_stream_turn`` for why the flag is needed (cancel-scope
        # cancellation and ``interrupt_turn`` surface as the same type).
        self._interrupting: bool = False
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
            # Emit ``turn_started`` before the stream begins so the TUI can
            # flip its run-state to "running" + render the footer's state
            # cell. The matching ``turn_finished`` is emitted from
            # ``_stream_turn``'s finally (covers normal end, timeout,
            # error, AND interrupt) — the TUI uses that frame to return
            # to idle + refresh the footer's data cells.
            await conn.send(json.dumps({"type": "turn_started"}))
            await self._stream_turn(conn, msg.get("prompt", ""))
        elif typ == "interrupt_turn":
            # Reuse the existing abort primitive (the same cancel used by
            # run_turn-supersede at #316 and disconnect in ``_handle_conn``'s
            # finally at #396) so there's no parallel abort path. The
            # ``_stream_turn`` finally emits ``turn_finished`` once the
            # cancelled task unwinds, which is the frame the TUI awaits to
            # return to idle. Guarded for the no-active-turn case: an Esc
            # when idle is a benign no-op, not an error.
            #
            # Set ``_interrupting`` BEFORE cancelling so ``_stream_turn``'s
            # ``except asyncio.CancelledError`` can tell this intentional
            # cancel apart from a turn-timeout cancel (both arrive as the
            # same type). The interrupt path emits ``turn_finished`` only —
            # no ``error`` frame — because interrupt is a user action, not
            # a fault.
            if self._active_turn is not None and not self._active_turn.done():
                self._interrupting = True
                self._active_turn.cancel()
                try:
                    await self._active_turn
                except (asyncio.CancelledError, Exception):
                    pass
            self._active_turn = None
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

        Exit paths + the frames each emits (the TUI reconciles run-state to
        idle off the terminal ``turn_finished`` from ``finally``):

        - normal end        → stream events + ``turn_finished``
        - timeout           → ``error: turn timeout`` + ``turn_finished``
        - run_stream raises → ``error: internal error`` + ``turn_finished``
        - interrupt_turn    → ``turn_finished`` only (no error — intentional)

        Timeout vs. interrupt disambiguation: ``anyio.fail_after``'s deadline
        normally surfaces as ``TimeoutError`` (the ``except TimeoutError``
        branch below — the common path). On some Python / async-generator
        setups the cancel-scope cancellation can leak as a raw
        ``asyncio.CancelledError`` instead; ``interrupt_turn`` cancels the
        same task, so the two are indistinguishable by type alone. The
        ``_interrupting`` flag (set by the interrupt dispatch before
        cancelling, reset at the top of this method on each turn) is the
        tie-breaker: a ``CancelledError`` with the flag set is an intentional
        interrupt (no error frame); without it, the cancellation is treated
        as a timeout leak and ``error: turn timeout`` is emitted. The
        handler does not re-raise — falling through to ``finally`` keeps the
        terminal-frame send off the cancellation path so the TUI footer
        reliably reconciles to idle.
        """
        # Reset at turn start: a CancelledError during this turn is a
        # timeout unless ``interrupt_turn`` flips ``_interrupting`` first.
        self._interrupting = False
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
        except asyncio.CancelledError:
            # ``CancelledError`` is a ``BaseException`` — it bypasses the
            # ``except Exception`` handler below. Reach here when
            # ``anyio.fail_after``'s cancellation surfaces unconverted (the
            # ``TimeoutError`` branch above is the common path) OR when
            # ``interrupt_turn`` / run_turn-supersede / disconnect cancels
            # the task. The ``_interrupting`` flag distinguishes intentional
            # interrupt (no error frame — interrupt is a user action, not a
            # fault) from a timeout leak (emit the ``error: turn timeout``
            # frame). Either way, fall through to ``finally`` rather than
            # re-raising: re-raise would subject the terminal-frame send to
            # cancellation re-delivery on some Python versions and risk
            # losing the ``turn_finished`` frame the TUI footer reconciles
            # on. Swallowing ends the task cleanly; callers
            # (``_handle_conn`` finally, interrupt dispatch, run_turn
            # supersede) await + absorb both ``CancelledError`` and normal
            # completion, so the task's cancelled-vs-done state is unused.
            if not self._interrupting:
                logger.warning(
                    "SessionWorker turn cancelled (treated as timeout) "
                    "after %ds",
                    _TURN_TIMEOUT_S,
                )
                await conn.send(
                    json.dumps({"type": "error", "message": "turn timeout"})
                )
        except Exception:  # noqa: BLE001
            logger.exception("Agent.run_stream failed")
            await conn.send(json.dumps({"type": "error", "message": "internal error"}))
        finally:
            # Terminal frame on every exit path. The TUI reconciles its
            # run-state to idle off this frame and re-reads the post-turn
            # context pressure + active_skills (which may have mutated
            # during the turn via load_skill/deactivate_skill tool calls).
            # ``pressure`` carries the ``PressureLevel`` value string
            # (cleanly JSON-serialisable — ``PressureLevel`` is a ``str``
            # Enum) or ``None`` when the budget is unknown.
            await self._emit_turn_finished(conn)

    async def _emit_turn_finished(self, conn: Connection) -> None:
        """Send the terminal ``turn_finished`` frame, guarded against a closing conn.

        Reads model / session_id / context pressure / active_skills from
        ``self._agent`` (and ``self._agent.session`` / ``_session``). Every
        field is validated for JSON-safety (``str`` / ``None`` / ``list[str]``)
        and falls back to a safe default rather than raising — the snapshot
        is best-effort, and a non-serialisable value on any agent shape must
        not break the terminal-frame guarantee the TUI relies on.

        The send is shielded (``asyncio.shield``) and wrapped in try/except:
        a ``CancelledError`` reaching ``_stream_turn``'s ``finally`` (timeout
        or interrupt) can be re-delivered at the await point on some Python
        versions, which would abort the send mid-flight and lose the terminal
        frame. ``shield`` runs the send in an inner task that survives outer
        cancellation, so the frame reaches the client even as the task
        unwinds; the outer ``CancelledError`` is swallowed (the frame is the
        point — the task's cancel state is settled by the caller).
        """
        agent = self._agent
        # ``session`` is exposed as ``_session`` (PrivateAttr) on real
        # agents; try both spellings defensively so test stubs + future
        # surfaces both work.
        session = getattr(agent, "session", None) or getattr(agent, "_session", None)
        session_id = self._safe_session_id(session)
        active_skills = self._safe_active_skills(session)
        pressure = self._safe_pressure(agent)
        model = self._safe_model(agent)
        payload = json.dumps(
            {
                "type": "turn_finished",
                "model": model,
                "session_id": session_id,
                "pressure": pressure,
                "active_skills": active_skills,
            }
        )
        try:
            # ``shield`` so task-cancellation re-delivery at this await
            # doesn't abort the send; the inner task completes + delivers
            # the frame, the outer ``CancelledError`` is caught below.
            await asyncio.shield(conn.send(payload))
        except asyncio.CancelledError:
            # Outer task unwinding (timeout / interrupt / disconnect); the
            # shielded send completes in the background. Don't mask the
            # cancel as a fresh error — the TUI reconciles on this frame
            # (or on the next turn's ``turn_started`` if the conn was
            # already tearing down).
            pass
        except Exception:  # noqa: BLE001 — conn mid-close, best-effort send
            # The conn may be mid-close after cancellation (the interrupt
            # handler cancels ``_active_turn`` which raises out of the
            # ``run_stream`` loop into the finally above). Swallow so a
            # closing conn doesn't mask the cancel as a fresh error.
            pass

    @staticmethod
    def _safe_session_id(session: Any) -> str | None:
        """Read ``session.session_id`` if it's a real string, else ``None``."""
        if session is None:
            return None
        try:
            sid = session.session_id
        except Exception:  # noqa: BLE001 — best-effort snapshot read
            return None
        return sid if isinstance(sid, str) else None

    @staticmethod
    def _safe_active_skills(session: Any) -> list[str]:
        """Read ``session.active_skills`` as a sorted ``list[str]``; ``[]`` on any failure."""
        if session is None:
            return []
        try:
            skills = session.active_skills
        except Exception:  # noqa: BLE001 — best-effort snapshot read
            return []
        try:
            return sorted(str(s) for s in skills)
        except Exception:  # noqa: BLE001 — non-iterable / malformed
            return []

    @staticmethod
    def _safe_pressure(agent: Any) -> str | None:
        """Read ``agent.context_budget().pressure.value``; ``None`` on any failure."""
        try:
            budget = agent.context_budget()
        except Exception:  # noqa: BLE001 — best-effort snapshot read
            return None
        if budget is None:
            return None
        try:
            pressure = budget.pressure
        except Exception:  # noqa: BLE001 — best-effort snapshot read
            return None
        if pressure is None:
            return None
        try:
            value = pressure.value
        except Exception:  # noqa: BLE001 — best-effort snapshot read
            return None
        return value if isinstance(value, str) else None

    @staticmethod
    def _safe_model(agent: Any) -> str | None:
        """Read ``agent.model`` if it's a real string, else ``None``."""
        try:
            model = agent.model
        except Exception:  # noqa: BLE001 — best-effort snapshot read
            return None
        return model if isinstance(model, str) else None

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
