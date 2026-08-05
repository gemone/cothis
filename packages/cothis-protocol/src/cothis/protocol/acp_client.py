"""``cothis.protocol.acp_client`` — the Agent Client Protocol client core.

Transport-neutral peer of :mod:`cothis.protocol.acp`. An :class:`ACPClient`
owns one :class:`~cothis.protocol.acp.ByteConnection`, performs the ``hello``
handshake, drives the three implemented commands (``list`` / ``create`` /
``prompt``), and demultiplexes the inbound server-message stream into per-id
responses and per-session progress events.

It depends only on the wire codec (:mod:`cothis.protocol.wire`) and the
message models (:mod:`cothis.protocol.messages`) — the same primitives the
server uses. A stdlib :func:`connect_stdio` helper spawns a ``cothis acp``
subprocess and returns a connected client; no third-party dependency.

The remaining commands (``attach`` / ``detach`` / ``steer`` / ``abort`` /
``set_model`` / ``set_thinking``) are answered ``invalid_request`` by the
server today; the demultiplex here already supports them and they land
one-for-one when the server does.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from cothis.protocol.messages import (
    PROTOCOL_VERSION,
    ClientHello,
    CreateResult,
    ListResult,
    ModelRef,
    PromptResult,
    ProtocolError,
    ResponseEnvelope,
    ServerHello,
    ServerHelloError,
    ServerSnapshot,
    SessionSnapshot,
    SessionSummary,
    ThinkingLevel,
    TranscriptProgress,
)
from cothis.protocol.wire import (
    FrameError,
    ProtocolValidationError,
    ServerMessageDecoder,
    encode_client_message,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cothis.protocol.acp import ByteConnection

logger = logging.getLogger(__name__)

#: Default ceiling for how long :func:`connect_stdio` waits for the handshake.
DEFAULT_START_TIMEOUT = 10.0

#: How long :meth:`_SubprocessByteConnection.close` waits for the child to
#: exit after its stdin is closed before force-killing it — bounds reaping so
#: a wedged server child can never leak as a zombie on the success path.
_SUBPROCESS_CLOSE_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ACPError(Exception):
    """Base class for ACP client failures (transport, framing, lifecycle)."""


class ACPHandshakeError(ACPError):
    """The server rejected the ``hello`` (auth / version / malformed).

    Carries the :class:`ProtocolError` the server returned in its
    ``hello_error``, so callers can branch on ``.error.code``.
    """

    def __init__(self, error: ProtocolError) -> None:
        self.error = error
        super().__init__(f"handshake failed: {error.code} ({error.message})")


class ACPRequestError(ACPError):
    """A command returned ``ok=False`` (e.g. ``not_found``).

    Carries the :class:`ProtocolError` from the response envelope.
    """

    def __init__(self, error: ProtocolError) -> None:
        self.error = error
        super().__init__(f"request failed: {error.code} ({error.message})")


class _ReaderClosed:
    """Sentinel put on a prompt queue when the reader task has exited.

    Signals an in-flight prompt generator to stop; carries the exception
    (or message) explaining why the stream ended.
    """

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ACPClient:
    """A client that drives one ACP server over a :class:`ByteConnection`.

    Typical use::

        client = ACPClient(conn, token="...")
        snapshot = await client.connect()
        snap = await client.create_session(cwd="/tmp")
        async for progress in client.prompt(snap.id, "hello"):
            ...
        await client.aclose()

    The inbound server-message stream is read by a background task started
    by :meth:`connect`: ``ResponseEnvelope``\\ s are routed by ``id`` to the
    awaiting caller, and ``session_progress`` events are routed to the
    active prompt generator for that session.
    """

    def __init__(
        self,
        conn: ByteConnection,
        *,
        token: str,
        max_frame_length: int | None = None,
    ) -> None:
        if not token:
            raise ValueError("ACPClient token must not be empty")
        self._conn = conn
        self._token = token
        self._decoder = ServerMessageDecoder(max_frame_length=max_frame_length)
        self._max_frame = max_frame_length
        self._loop = asyncio.get_running_loop()
        # Handshake result.
        self.connection_id: str | None = None
        self.snapshot: ServerSnapshot | None = None
        #: Populated when a ``prompt`` generator exhausts on the matching response.
        self.last_prompt_snapshot: SessionSnapshot | None = None
        # Reader state.
        self._reader_task: asyncio.Task[None] | None = None
        # id -> future resolving to the ResponseEnvelope (non-streaming requests).
        self._requests: dict[str, asyncio.Future[ResponseEnvelope]] = {}
        # Active prompt routing. Each prompt registers its queue under both the
        # session id (for session_progress events) and the request id (for the
        # terminal response), so the reader can route either without branching.
        self._progress_routes: dict[str, asyncio.Queue[Any]] = {}
        self._response_routes: dict[str, asyncio.Queue[Any]] = {}

    # ------------------------------------------------------------------ lifecycle

    async def connect(self) -> ServerSnapshot:
        """Send ``hello`` and await the server's ``hello`` (or ``hello_error``).

        On success stores ``connection_id`` / ``snapshot`` on the instance,
        starts the background reader, and returns the handshake
        :class:`ServerSnapshot`. A ``hello_error`` raises
        :class:`ACPHandshakeError`.
        """
        hello = ClientHello(
            type="hello", version=PROTOCOL_VERSION, token=self._token
        )
        await self._conn.send(
            encode_client_message(
                hello.model_dump(mode="json"), max_frame_length=self._max_frame
            )
        )

        greeting: ServerHello | None = None
        carry: list[Any] = []
        async for chunk in self._conn:
            for message in self._decoder.push(chunk):
                if greeting is None:
                    if isinstance(message, ServerHello):
                        greeting = message
                    elif isinstance(message, ServerHelloError):
                        raise ACPHandshakeError(message.error)
                    else:
                        raise ACPError(
                            f"unexpected message before hello: {message.type!r}"
                        )
                else:
                    carry.append(message)
            if greeting is not None:
                break

        if greeting is None:
            raise ACPError("connection closed before handshake completed")

        self.connection_id = greeting.connectionId
        self.snapshot = greeting.snapshot
        # The reader owns the stream from here. Any messages that arrived in
        # the same chunk as the hello (clients may batch, servers may pipeline)
        # are fed to it before it reads the next chunk.
        self._reader_task = self._loop.create_task(self._read_loop(carry))
        return greeting.snapshot

    async def aclose(self) -> None:
        """Cancel the reader and close the connection.

        In-flight requests are failed by the reader's cancellation handler.
        Safe to call once; subsequent calls are no-ops.
        """
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except Exception:
                # The reader raised a real error (framing/transport); it has
                # already failed in-flight callers via _fail_all. Swallow.
                logger.debug("reader task ended with error during aclose", exc_info=True)
            except asyncio.CancelledError:
                # Expected: we cancelled the reader just above, so awaiting it
                # re-raises its own cancellation — swallow that. But if the
                # *outer* task running aclose was itself cancelled mid-await,
                # propagate so the caller's cancellation unwinds promptly
                # rather than being delayed by the conn.close() below.
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise
        if not self._conn.closed:
            try:
                await self._conn.close()
            except Exception:
                logger.debug("error closing connection during aclose", exc_info=True)

    # ------------------------------------------------------------------ commands

    async def list_sessions(self) -> list[SessionSummary]:
        """List the sessions the server currently holds."""
        response = await self._roundtrip({"command": "list"})
        result = response.result
        if isinstance(result, ListResult):
            return list(result.sessions)
        raise ACPError(f"unexpected list result shape: {type(result).__name__}")

    async def create_session(
        self,
        *,
        cwd: str | None = None,
        name: str | None = None,
        model: ModelRef | None = None,
        thinking_level: ThinkingLevel | None = None,
    ) -> SessionSnapshot:
        """Create a new session and return its authoritative snapshot."""
        request: dict[str, Any] = {"command": "create"}
        if cwd is not None:
            request["cwd"] = cwd
        if name is not None:
            request["name"] = name
        if model is not None:
            request["model"] = model.model_dump(mode="json")
        if thinking_level is not None:
            request["thinkingLevel"] = thinking_level
        response = await self._roundtrip(request)
        result = response.result
        if isinstance(result, CreateResult):
            return result.session
        raise ACPError(f"unexpected create result shape: {type(result).__name__}")

    def prompt(self, session_id: str, text: str) -> AsyncIterator[TranscriptProgress]:
        """Stream ``TranscriptProgress`` for one prompt turn.

        Sends the ``prompt`` request, yields each ``session_progress`` event's
        progress as it arrives, and stops when the matching ``ok=True``
        response lands. An ``ok=False`` response raises
        :class:`ACPRequestError` mid-iteration. The post-turn
        :class:`SessionSnapshot` from the response is available as the
        generator's ``send``-time ``value`` via :attr:`last_prompt_snapshot`
        once the generator has exhausted.

        Only one prompt may be active per session id at a time: progress
        events are routed by ``sessionId``, so a second concurrent prompt on
        the same session is rejected with :class:`ACPError` rather than
        silently mis-delivered. (The server serializes turns per session
        regardless.)
        """
        return self._prompt(session_id, text)

    async def _prompt(
        self, session_id: str, text: str
    ) -> AsyncIterator[TranscriptProgress]:
        if session_id in self._progress_routes:
            raise ACPError(
                f"a prompt is already active on session {session_id!r}; "
                "only one prompt may be active per session at a time"
            )
        req_id = self._next_id()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._progress_routes[session_id] = queue
        self._response_routes[req_id] = queue
        await self._send_request(
            req_id,
            {"command": "prompt", "sessionId": session_id, "text": text},
        )
        try:
            while True:
                item = await queue.get()
                if isinstance(item, _ReaderClosed):
                    raise item.exc
                if isinstance(item, ResponseEnvelope):
                    if not item.ok:
                        assert item.error is not None
                        raise ACPRequestError(item.error)
                    result = item.result
                    if isinstance(result, PromptResult):
                        # Capture the post-turn snapshot for the caller.
                        self.last_prompt_snapshot = result.session
                    return
                # TranscriptProgress event.
                yield item
        finally:
            self._progress_routes.pop(session_id, None)
            self._response_routes.pop(req_id, None)

    # ------------------------------------------------------------------ internals

    def _next_id(self) -> str:
        return uuid.uuid4().hex

    async def _send_request(self, req_id: str, request: dict[str, Any]) -> None:
        envelope = {
            "type": "request",
            "id": req_id,
            "request": request,
        }
        await self._conn.send(
            encode_client_message(envelope, max_frame_length=self._max_frame)
        )

    async def _roundtrip(self, request: dict[str, Any]) -> ResponseEnvelope:
        req_id = self._next_id()
        future: asyncio.Future[ResponseEnvelope] = self._loop.create_future()
        self._requests[req_id] = future
        try:
            await self._send_request(req_id, request)
            response = await future
        finally:
            self._requests.pop(req_id, None)
        if not response.ok:
            assert response.error is not None
            raise ACPRequestError(response.error)
        return response

    async def _read_loop(self, carry: list[Any]) -> None:
        try:
            for message in carry:
                self._route(message)
            async for chunk in self._conn:
                for message in self._decoder.push(chunk):
                    self._route(message)
            # Stream ended without error — fail in-flight callers so they
            # don't hang on a future that will never resolve.
            self._fail_all(ACPError("connection closed by server"))
        except asyncio.CancelledError:
            self._fail_all(ACPError("connection closed"))
            raise
        except (FrameError, ProtocolValidationError) as exc:
            self._fail_all(ACPError(f"protocol stream error: {exc}"))
        except Exception as exc:
            self._fail_all(ACPError(f"reader error: {exc}"))

    def _route(self, message: Any) -> None:
        mtype = getattr(message, "type", None)
        if mtype == "response":
            # An active prompt consumes its own response via its queue.
            prompt_queue = self._response_routes.get(message.id)
            if prompt_queue is not None:
                prompt_queue.put_nowait(message)
                return
            future = self._requests.get(message.id)
            if future is not None and not future.done():
                future.set_result(message)
            return
        if mtype == "event":
            event = message.event
            kind = getattr(event, "type", None)
            if kind == "server_snapshot":
                self.snapshot = event.snapshot
            elif kind == "session_progress":
                queue = self._progress_routes.get(event.sessionId)
                if queue is not None:
                    queue.put_nowait(event.progress)
            # session_snapshot / session_removed are not consumed by the
            # client today (deferred); drop them.
            return
        # hello / hello_error post-handshake: the server does not send them,
        # but ignore defensively rather than tear down a healthy connection.

    def _fail_all(self, exc: BaseException) -> None:
        for future in self._requests.values():
            if not future.done():
                future.set_exception(exc)
        for queue in self._response_routes.values():
            queue.put_nowait(_ReaderClosed(exc))


# ---------------------------------------------------------------------------
# stdio subprocess helper
# ---------------------------------------------------------------------------


class _SubprocessByteConnection:
    """A :class:`ByteConnection` backed by a child process's stdin/stdout.

    Mirrors the server's ``_StdioByteConnection`` shape but owns a *child*
    process rather than the process's own stdio: ``send`` writes the child's
    stdin, ``__aiter__`` reads its stdout, and ``close`` closes stdin to
    signal the child to finish (its read loop then sees EOF).
    """

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc
        self.closed = False

    async def send(self, chunk: bytes) -> None:
        if self._proc.stdin is None:
            raise ACPError("subprocess stdin is not available")
        self._proc.stdin.write(chunk)
        await self._proc.stdin.drain()

    async def close(self, final_chunk: bytes | None = None) -> None:
        if final_chunk is not None and self._proc.stdin is not None:
            self._proc.stdin.write(final_chunk)
            await self._proc.stdin.drain()
        self.closed = True
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        # Reap the child so connect_stdio can never leak a zombie on the
        # success path: closing stdin signals the server to exit, then we
        # wait for it (bounded) and force-kill if it doesn't drain. This
        # keeps cleanup symmetric with the handshake-failure path in
        # connect_stdio (which also kills + waits).
        try:
            await asyncio.wait_for(
                self._proc.wait(), timeout=_SUBPROCESS_CLOSE_TIMEOUT
            )
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()

    def __aiter__(self) -> _SubprocessByteConnection:
        return self

    async def __anext__(self) -> bytes:
        if self._proc.stdout is None:
            raise StopAsyncIteration
        chunk = await self._proc.stdout.read(65536)
        if not chunk:
            raise StopAsyncIteration
        return chunk


async def connect_stdio(
    argv: list[str],
    token: str,
    *,
    max_frame_length: int | None = None,
    start_timeout: float = DEFAULT_START_TIMEOUT,
) -> ACPClient:
    """Spawn ``argv`` (typically ``["cothis", "acp", "--token", token]``).

    Wraps the child's stdin/stdout in a :class:`_SubprocessByteConnection`,
    builds an :class:`ACPClient`, and drives the handshake, returning a
    connected client. Stdlib only (``asyncio.subprocess``). The caller owns
    the returned client and should :meth:`~ACPClient.aclose` it (which
    closes stdin so the child exits gracefully).
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    conn = _SubprocessByteConnection(proc)
    client = ACPClient(conn, token=token, max_frame_length=max_frame_length)
    try:
        await asyncio.wait_for(client.connect(), timeout=start_timeout)
    except BaseException:
        # On handshake failure clean up the child so it doesn't linger.
        proc.kill()
        await proc.wait()
        raise
    return client


__all__ = [
    "ACPError",
    "ACPHandshakeError",
    "ACPRequestError",
    "ACPClient",
    "connect_stdio",
    "DEFAULT_START_TIMEOUT",
]
