"""``cothis.worker`` — SessionWorker process entrypoint (#225).

Owns one ``Agent`` + binds a loopback WebSocket that accepts control
messages (``run_turn`` / ``attach_input`` / ``detach_input`` /
``shutdown`` / ``ping``) and emits stream messages (``assistant_delta``
/ ``tool_call_started`` / ``tool_call_result_pointer`` / ``pong`` /
``error``).

Handshake requires a valid bearer token on the ``Authorization``
header. Missing or wrong token → HTTP 401, connection rejected. The
token is generated via ``secrets.token_urlsafe``; the Supervisor
(#227) will receive it from the spawn call and pass it back to the
TUI via an IPC channel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from cothis.agent import Agent, ToolCallEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


_WS_PATH = "/agent"
_TURN_TIMEOUT_S = 300
_MAX_CONCURRENT_CONNS = 4


def _http_401() -> Response:
    """Fresh 401 Response per handshake (``Headers`` is mutable)."""
    return Response(401, "Unauthorized", Headers())


def _http_503() -> Response:
    """Fresh 503 Response for over-capacity handshakes."""
    return Response(503, "Service Unavailable", Headers())


class SessionWorker:
    """One per session. Owns Agent + binds WS on loopback.

    Lifecycle:

    - ``__init__`` generates the bearer token; no network yet.
    - ``start()`` binds the WS server on ``127.0.0.1:0`` (random port),
      returns a URI the Supervisor hands to the TUI.
    - ``serve_forever()`` blocks until ``shutdown`` arrives or ``stop()``
      is called.
    - ``stop()`` closes the server + ``Agent.aclose()``.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        host: str = "127.0.0.1",
    ) -> None:
        self._agent = agent
        self._host = host
        self._token = secrets.token_urlsafe(32)
        self._server: Any = None
        self._stop_event = asyncio.Event()
        self._active_conns: int = 0

    @property
    def token(self) -> str:
        """The bearer token the client must present."""
        return self._token

    @property
    def uri(self) -> str | None:
        """WS URI once ``start`` has bound the port; ``None`` otherwise."""
        if self._server is None:
            return None
        sockets = self._server.sockets
        if not sockets:
            return None
        port = sockets[0].getsockname()[1]
        return f"ws://{self._host}:{port}{_WS_PATH}"

    async def start(self) -> str:
        """Bind the WS server on a random loopback port; return the URI."""
        self._server = await serve(
            self._handle_conn,
            self._host,
            0,
            process_request=self._check_auth,
        )
        if self.uri is None:
            raise RuntimeError("WS server bound but no sockets found")
        return self.uri

    async def _check_auth(
        self,
        conn: ServerConnection,
        request: Any,
    ) -> Response | None:
        """Handshake gate: ``Authorization: Bearer <token>`` required.

        websockets v16 calls ``process_request(conn, request)`` where
        ``request.path`` is the URL path and ``request.headers`` is the
        ``Headers`` mapping.
        """
        if request.path != _WS_PATH:
            return _http_401()
        if self._active_conns >= _MAX_CONCURRENT_CONNS:
            return _http_503()
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return _http_401()
        # Constant-time compare; token length is bounded (32+ chars).
        if not secrets.compare_digest(auth[len("Bearer "):], self._token):
            return _http_401()
        return None

    async def _handle_conn(self, conn: ServerConnection) -> None:
        """Dispatch control messages until the connection closes."""
        self._active_conns += 1
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
        finally:
            self._active_conns -= 1

    async def _dispatch(self, conn: ServerConnection, msg: dict[str, Any]) -> None:
        """One control message → one or more WS responses."""
        typ = msg["type"]
        if typ == "ping":
            await conn.send(json.dumps({"type": "pong"}))
        elif typ == "shutdown":
            await conn.close()
            self._stop_event.set()
        elif typ == "run_turn":
            await self._stream_turn(conn, msg.get("prompt", ""))
        elif typ in ("attach_input", "detach_input"):
            # Real terminal attach lands with #230; accept + ignore for now.
            logger.debug("SessionWorker got %r (terminal attach deferred)", typ)
        else:
            await conn.send(
                json.dumps({"type": "error", "message": f"unknown type: {typ!r}"})
            )

    async def _stream_turn(self, conn: ServerConnection, prompt: str) -> None:
        """Drive ``Agent.run_stream`` and forward each event to the client.

        Bounded by ``_TURN_TIMEOUT_S`` so a stuck tool or model stream
        can't hold the connection indefinitely. Errors are logged
        server-side + a generic ``"internal error"`` goes to the client
        (loopback-only is not a license to leak exception details).
        """
        try:
            async with asyncio.timeout(_TURN_TIMEOUT_S):
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
        """Block until ``shutdown`` arrives or ``stop()`` is called."""
        await self._stop_event.wait()

    async def stop(self) -> None:
        """Close the server + Agent. Idempotent."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._stop_event.set()
        aclose = getattr(self._agent, "aclose", None)
        if aclose is not None:
            await aclose()


# ---------------------------------------------------------------------
# CLI entrypoint: ``python -m cothis.worker --session <id> ...``
# ---------------------------------------------------------------------
#
# Spawned by ``Supervisor.spawn_worker`` (#227 follow-up) and directly
# runnable by a user. On bind it prints exactly one JSON line —
# ``{"uri": ..., "token": ...}`` — to stdout; the Supervisor reads that
# line to learn where the worker is listening + the bearer token to
# pass to the TUI. Everything else goes to stderr so the JSON line is
# unambiguous even when the worker logs.
#
# cothis: ``argparse``, not ``typer``. This process is spawned by the
# Supervisor, never typed by a human; typer's 30ms startup cost buys
# nothing here and counts against the worker-bind latency budget.
# ``run_turn`` itself needs an Agent; this entrypoint is the minimum
# glue to get one listening on a WS.

def _build_arg_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="cothis.worker",
        description="Run one cothis session worker (WS server on loopback).",
    )
    parser.add_argument(
        "--session",
        required=True,
        help="32-char hex session id to attach this worker to.",
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="any-llm provider key (e.g. openrouter, openai, anthropic).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier for the chosen provider.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="LLM round-trip cap per turn (default: 30).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Output-token cap. Default: resolved from litellm metadata.",
    )
    return parser


def _emit_ready(uri: str, token: str) -> None:
    """Print exactly one ``{"uri", "token"}`` JSON line to stdout.

    ``ensure_ascii=False`` per the project's text-boundary default; the
    token is ASCII regardless, but the URI host could be non-ASCII in
    exotic setups and the line is the Supervisor's parse target.
    """
    import sys

    line = json.dumps({"uri": uri, "token": token}, ensure_ascii=False)
    # ``flush=True`` — the Supervisor blocks on readline() and the pipe
    # is block-buffered by default under subprocess.Popen.
    print(line, flush=True, file=sys.stdout)


async def _serve(argv: list[str] | None = None) -> None:
    """Parse argv, build Agent+Session, bind worker, serve until shutdown."""
    args = _build_arg_parser().parse_args(argv)

    # Must run before cothis.agent imports any_llm — mirrors cli.py.
    os.environ.setdefault("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

    from cothis.agent import Agent
    from cothis.cli import (
        _PROJECT_TOOLS_DIR,
        DEFAULT_SYSTEM_PROMPT,
        _resolve_db_path,
        _user_tools_dir,
        _validate_session_id_arg,
    )
    from cothis.session import Session
    from cothis.tools import discover_tools

    _validate_session_id_arg(args.session)
    db_path = _resolve_db_path()
    session = Session.load(db_path, args.session, cwd=Path.cwd())
    try:
        agent = Agent(
            model=args.model,
            provider=args.provider,
            tools=discover_tools(_PROJECT_TOOLS_DIR, _user_tools_dir()),
            system=DEFAULT_SYSTEM_PROMPT,
            max_iterations=args.max_iterations,
            max_tokens=args.max_tokens,
            cwd=Path.cwd(),
        )
        agent.attach_session(session)
        worker = SessionWorker(agent)
        uri = await worker.start()
        _emit_ready(uri, worker.token)
        try:
            await worker.serve_forever()
        finally:
            await worker.stop()
    finally:
        session.close()


def run(argv: list[str] | None = None) -> None:
    """Entry point: ``python -m cothis.worker``.

    ``argv`` defaults to ``sys.argv[1:]``. Exits non-zero on bind failure
    or if the session id is malformed/missing.
    """
    asyncio.run(_serve(argv))


if __name__ == "__main__":
    run()
