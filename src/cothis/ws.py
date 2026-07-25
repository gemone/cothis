"""``cothis.ws`` — WebSocket transport seam for the SessionWorker (#248).

The SessionWorker translates ``Agent.run_stream`` events into WS control
messages. That translation is the worker's job; *binding a socket, doing the
handshake, and shuttling frames* is the transport's job. This module is the
seam between them: a small ``WSTransport`` protocol the worker depends on,
plus one production adapter (``WebSocketServerTransport``) wrapping
``websockets``.

Two adapters means a real seam, not a hypothetical one — production wires the
``websockets`` adapter; tests inject a fake (``tests/test_ws_transport.py``)
so ``worker`` logic is exercised without a socket. This is the acceptance
criterion deferred from #225: *anyio WS wrappers are isolated and
unit-testable (mock the transport)*.

anyio is the runtime abstraction: the adapter uses ``anyio.Event`` for its
bind/shutdown signals and the worker uses ``anyio.fail_after`` for its turn
timeout, so neither side names ``asyncio`` directly. The ``websockets``
library is asyncio-only, but it runs unchanged under anyio's asyncio backend;
if a second backend ever appears, only the adapter in this file needs to
change — the worker is already backend-neutral. See ADR-0017 §6.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import anyio
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

# The auth callback inspects the WS handshake request and returns ``None`` to
# accept or an HTTP ``Response`` (401/503) to reject. Defined here so the
# transport and worker share one name for it.
AuthCheck = Callable[[Any], "Response | None"]


def _http_401() -> Response:
    """Fresh 401 Response per handshake (``Headers`` is mutable)."""
    return Response(401, "Unauthorized", Headers())


def _http_503() -> Response:
    """Fresh 503 Response for over-capacity handshakes."""
    return Response(503, "Service Unavailable", Headers())


@runtime_checkable
class Connection(Protocol):
    """One open WS connection — the only surface the worker uses per peer.

    Narrowed from ``websockets``' ``ServerConnection`` to the three things the
    worker actually does: iterate inbound frames, send an outbound frame, and
    close. A fake transport implements these three and nothing else.
    """

    def send(self, message: str) -> Awaitable[None]: ...
    def close(self) -> Awaitable[None]: ...
    def __aiter__(self) -> AsyncIterator[str]: ...


@runtime_checkable
class WSTransport(Protocol):
    """Server-side WS transport the SessionWorker depends on (#248).

    Two-phase lifecycle so ``worker.start()`` can read ``uri`` before
    ``serve_forever()`` blocks:

    - ``bind(handler, auth)`` binds the socket + returns (sets ``uri``).
      Called once from ``worker.start``.
    - ``serve()`` runs the accept loop until ``request_shutdown``. Called once
      from ``worker.serve_forever``.
    """

    @property
    def uri(self) -> str | None: ...

    async def bind(
        self,
        handler: Callable[[Connection], Awaitable[None]],
        auth: AuthCheck,
    ) -> None:
        """Bind the socket + install the handler/auth gate (non-blocking)."""
        ...

    async def serve(self) -> None:
        """Run the accept loop until ``request_shutdown``; then close."""
        ...

    def request_shutdown(self) -> None:
        """Signal ``serve`` to return (idempotent, non-blocking)."""
        ...


async def _str_iter(source: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Yield each frame from ``source`` as ``str`` (decode bytes if needed).

    ``websockets`` yields ``Data`` (``str | bytes``); the worker only sends
    text frames and JSON-decodes inbound bytes as UTF-8, so coerce to ``str``
    here and keep ``Connection.__aiter__`` text-typed.
    """
    async for frame in source:
        yield frame.decode() if isinstance(frame, bytes) else frame


class _WSConn:
    """Adapter from ``websockets.asyncio.server.ServerConnection`` to our
    ``Connection`` protocol. Exposes send/close/``__aiter__`` and nothing else.
    """

    __slots__ = ("_conn",)

    def __init__(self, conn: ServerConnection) -> None:
        self._conn = conn

    async def send(self, message: str) -> None:
        await self._conn.send(message)

    async def close(self) -> None:
        await self._conn.close()

    def __aiter__(self) -> AsyncIterator[str]:
        return _str_iter(self._conn.__aiter__())


class WebSocketServerTransport:
    """Production adapter over ``websockets`` (#248).

    Binds ``127.0.0.1:0`` (random loopback port — see ADR-0017 §2), enforces a
    bearer-token handshake via the ``auth`` callback passed to ``bind``, and
    caps concurrent connections so a flood of handshakes can't exhaust the
    worker. ``bind`` opens the socket (``uri`` becomes available); ``serve``
    then runs the accept loop until ``request_shutdown``.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        path: str = "/agent",
        max_concurrent_conns: int = 4,
    ) -> None:
        self._host = host
        self._path = path
        self._max_conns = max_concurrent_conns
        self._server: Any = None
        self._active_conns: int = 0

    @property
    def uri(self) -> str | None:
        if self._server is None:
            return None
        sockets = self._server.sockets
        if not sockets:
            return None
        port = sockets[0].getsockname()[1]
        return f"ws://{self._host}:{port}{self._path}"

    async def bind(
        self,
        handler: Callable[[Connection], Awaitable[None]],
        auth: AuthCheck,
    ) -> None:
        async def conn_handler(conn: ServerConnection) -> None:
            self._active_conns += 1
            try:
                await handler(_WSConn(conn))
            finally:
                self._active_conns -= 1

        def process_request(conn: ServerConnection, request: Any) -> Response | None:
            if request.path != self._path:
                return _http_401()
            if self._active_conns >= self._max_conns:
                return _http_503()
            return auth(request)

        # ``serve`` binds immediately and returns; ``serve_forever`` runs the
        # accept loop (called separately from ``serve`` below).
        self._server = await serve(
            conn_handler,
            self._host,
            0,
            process_request=process_request,
        )

    async def serve(self) -> None:
        """Run the accept loop until ``request_shutdown``; then close.

        Idempotent shutdown: if ``serve`` is entered without a prior ``bind``
        it returns immediately.
        """
        if self._server is None:
            return
        try:
            await self._server.serve_forever()
        except anyio.get_cancelled_exc_class():
            raise
        finally:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def request_shutdown(self) -> None:
        """Trigger ``serve`` to return by stopping the underlying server.

        ``serve_forever`` returns once the server stops serving, so closing
        the socket here is what unblocks ``serve`` — no separate signal needed.
        """
        if self._server is not None:
            self._server.close()
