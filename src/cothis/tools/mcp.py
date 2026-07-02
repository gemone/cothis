"""MCP (Model Context Protocol) server + client-tool subsystem.

Extracted from the original ``tools.py``. Holds the ``type: mcp.stdio`` /
``type: mcp.http`` YAML builders and the runtime that turns a declared server
into dispatchable tools: ``MCPServer`` (a producer handle) expands at Agent
startup into one ``MCPClientTool`` per remote tool. See ADR-0005 (deferred
connect) and ADR-0006 (``mcp:`` name prefix).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from cothis.tools.core import (
    _check_unknown_keys,
    _HookableTool,
    _require,
    _resolve_executable,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from typing import Any, NoReturn

logger = logging.getLogger("cothis.tools")


# Known keys on a ``type: mcp.stdio`` declaration. ``command`` is the server
# executable (a string), ``args`` its CLI arguments (a list), ``env`` the
# subprocess environment (a mapping). Disjoint from the shell-tool schema
# (``_TOOL_KEYS``) — MCP is a different ``type:``, routed before ``_compile``.
_MCP_STDIO_KEYS = {"type", "name", "description", "command", "args", "env"}

# Known keys on a ``type: mcp.http`` declaration. ``url`` is the remote server
# endpoint (a string), ``headers`` the HTTP headers sent on every request (a
# mapping — may carry secrets like ``Authorization``, so never logged).
_MCP_HTTP_KEYS = {"type", "name", "description", "url", "headers"}


def _normalize_mcp_result(result: Any) -> str:
    """Flatten an MCP ``CallToolResult`` into a single string for the LLM.

    Rules (issue #16, story 31):
    - Join every content block's ``.text`` with newlines.
    - Empty content list → ``"(no output)"`` (the tool ran but said nothing).
    - ``isError`` true → prefix ``"Error: "`` so the model sees it as a
      failure it can act on, not a normal result.

    Non-text content blocks (images, embedded resources) are skipped — cothis
    surfaces text to the model today.
    cothis: ceiling — image/resource blocks are dropped, not base64-inlined,
    so ``"(no output)"`` covers *two* cases the spec names as one: a truly
    empty content list, AND a non-empty list that carries only non-text
    blocks (both look like "nothing to say" to a text-only agent). Upgrade
    path: map non-text blocks into the message content array once the agent
    loop carries multimodal tool results — then the two cases diverge and
    text-less-but-non-empty results stop collapsing to ``"(no output)"``.
    """
    parts = [
        block.text
        for block in result.content
        if getattr(block, "text", None) is not None
    ]
    body = "\n".join(parts) if parts else "(no output)"
    # ``isError`` is camelCase on the MCP pydantic model (verified against
    # mcp 1.28.1 — ``CallToolResult`` fields: content/structuredContent/isError).
    if result.isError:
        return f"Error: {body}"
    return body


class MCPClientTool(_HookableTool):
    """A single remote MCP tool, dispatched over a shared server session.

    Produced by ``MCPServer.start`` — one instance per remote tool returned
    by ``tools/list``. Inherits ``_HookableTool`` so ``_execute`` runs its
    hook chains uniformly with every other tool (CONTEXT.md "no per-source
    branching in ``_execute``"). Carries a pre-built ``__cothis_schema__``
    from the server's ``inputSchema`` (already OpenAI-compatible JSON Schema).

    Two names (ADR-0006): ``__name__`` is ``{label}.{remote}`` (e.g.
    ``context7.query-docs``) everywhere cothis-side; ``_remote_name`` is the
    bare name sent to the server in ``call_tool``.
    """

    __name__: str
    __doc__: str
    __cothis_schema__: dict[str, Any]

    def __init__(
        self,
        server: MCPServer,
        label: str,
        remote_name: str,
        description: str,
        input_schema: dict[str, Any] | None,
    ) -> None:
        super().__init__()
        self._remote_name = remote_name
        self.__name__ = f"{label}.{remote_name}"
        self.__doc__ = description or f"MCP tool: {self.__name__}"
        self.__cothis_schema__ = {
            "type": "function",
            "function": {
                "name": self.__name__,
                "description": self.__doc__,
                # ``inputSchema`` from the MCP server is already an OpenAI-
                # compatible JSON Schema object; pass it straight through.
                "parameters": input_schema or {"type": "object", "properties": {}},
            },
        }
        self._server = server
        # Diagnostics parity with YAML/Python tools (``_all_tools`` reads it).
        self._source = server._source

    async def __call__(self, **kwargs: Any) -> str:
        session = self._server._session
        if session is None:
            # The server failed to start (its tools shouldn't have been
            # registered) or was closed. Surface as an error the model sees.
            raise RuntimeError(
                f"MCP tool {self.__name__!r}: server session is not active"
            )
        result = await session.call_tool(self._remote_name, kwargs)
        return _normalize_mcp_result(result)


def _flatten_exc(exc: BaseException) -> str:
    """Describe an exception, unwrapping ``ExceptionGroup``s to the real cause.

    anyio runs the MCP transport inside a task group, so a connection/protocol
    failure surfaces as an ``ExceptionGroup`` whose own message — ``"unhandled
    errors in a TaskGroup (1 sub-exception)"`` — hides what actually went
    wrong. Recurse into ``.exceptions`` and join the leaf messages so the
    startup warning names something the operator (and the model) can act on.
    """
    subs = getattr(exc, "exceptions", None)
    if subs:
        return "; ".join(_flatten_exc(s) for s in subs)
    return f"{type(exc).__name__}: {exc}"


def _scrub_url(url: str) -> str:
    """Strip userinfo and query from a url for safe logging.

    A url may carry credentials in the userinfo (``https://token@host``) or
    in the query string (``?api_key=secret``); both are dropped so the
    diagnostic keeps only ``scheme://host:port/path`` (story 32 — the
    ``diagnostic`` is the only url-derived string that reaches a log).
    """
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    # Drop userinfo (everything before the last ``@``) from the netloc, but
    # keep the netloc string itself — rebuilding from ``parts.hostname`` would
    # lose IPv6 brackets (``hostname`` returns ``::1`` for ``[::1]:8000``),
    # producing a malformed url in the log. Query/fragment are stripped too.
    netloc = parts.netloc.rsplit("@", 1)[-1] if "@" in parts.netloc else parts.netloc
    return urlunsplit(parts._replace(netloc=netloc, query="", fragment=""))


class MCPServer(_HookableTool):
    """A handle to an MCP server declared via ``type: mcp.stdio`` or
    ``type: mcp.http`` YAML.

    Not a callable tool itself — it's a *producer* of tools. Flows through
    discovery (``load_tools_from_layer``) and ``_all_tools`` as an opaque
    item (it satisfies the ``Tool`` protocol minimally: ``__name__`` +
    ``__call__``), then the Agent resolves it at startup: connect the
    server, ``tools/list``, expand into one ``MCPClientTool`` per remote
    tool (ADR-0005). ``__name__`` is a diagnostic label (``mcp:`` + ``name:``
    or the file stem), prefixed so it can never collide with the names of the
    tools it produces — or with any other dispatchable tool in the registry.

    Session lifecycle is manual (``start`` / ``aclose``) because it spans
    methods: ``async with`` sugar can't hold a context open from one call to
    the next. ``start`` enters the connection context; ``aclose`` exits it.
    Both must run in the same task (anyio cancel-scope rule) — the Agent
    awaits both from its own event loop, so this holds. See ADR-0005.

    The *transport* is the only thing that differs between MCP kinds, so it's
    the only injected piece: ``open_transport`` is a zero-arg callable
    returning an async context manager that yields a ``(read, write)`` stream
    pair; ``_default_connect`` wraps those streams in a ``ClientSession``
    uniformly. The stdio and http builders each supply the matching
    ``open_transport``; everything downstream (session, discovery, dispatch,
    normalization) is shared. ``diagnostic`` is a secret-free string (command
    + args, or url — never ``env``/``headers``) logged if the server fails to
    start.

    ``connect`` is an injection seam for tests: a zero-arg callable returning
    an async context manager that yields an *initialized* ``ClientSession``,
    bypassing ``open_transport`` entirely. Production leaves it ``None``;
    tests pass the SDK's in-memory transport.
    """

    __name__: str

    def __init__(
        self,
        *,
        name: str,
        open_transport: Callable[[], Any] | None = None,
        diagnostic: str = "",
        connect: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__()
        self.__name__ = name
        # Production transport factory (async CM yielding ``(read, write)``);
        # ``None`` in pure-seam tests, where ``connect`` supplies the session.
        self._open_transport = open_transport
        # Secret-free detail for the failure log (never ``env``/``headers``).
        self._diagnostic = diagnostic
        self._connect = connect
        # Live connection context + session, set by ``start``, cleared by
        # ``aclose``. ``None`` until started / after close.
        self._cm: Any = None
        self._session: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        # Satisfies the ``Tool`` protocol structurally but must never be
        # dispatched: the Agent filters ``MCPServer`` out of ``_tool_map``
        # and only registers the ``MCPClientTool`` instances it produces.
        raise RuntimeError(
            f"MCP server {self.__name__!r} is a server handle, not a callable tool"
        )

    @asynccontextmanager
    async def _default_connect(self) -> AsyncIterator[Any]:
        """Production transport: open ``open_transport``, wrap in a session.

        Transport-agnostic: ``open_transport`` yields the ``(read, write)``
        streams (stdio subprocess or http connection); this method wraps them
        in a ``ClientSession`` and initializes it. The lazy ``ClientSession``
        import keeps ``import cothis.tools`` cheap (the SDK pulls anyio +
        pydantic-settings + starlette); only a real MCP server pays.
        """
        from mcp import ClientSession

        # Only reached when there's no ``connect`` seam, in which case the
        # builders always supply ``open_transport``. Assert it for the type
        # checker (and to fail loudly if a future caller forgets both).
        assert self._open_transport is not None
        async with self._open_transport() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def start(self) -> list[MCPClientTool]:
        """Connect, list remote tools, return them wrapped as ``MCPClientTool``.

        Enters the connection context (held open on ``self._cm`` until
        ``aclose``), initializes the session, and calls ``tools/list``. On any
        failure (command not found, subprocess crash, connection refused,
        protocol error) logs at ``WARNING`` naming the server + its
        ``diagnostic`` — never ``env``/``headers`` secrets (story 32) —
        unwinds any partial context, and returns ``[]`` so the rest of the
        agent's tools still load (story 30).
        """
        cm = self._connect() if self._connect is not None else self._default_connect()
        try:
            self._session = await cm.__aenter__()
            listed = await self._session.list_tools()
        except Exception as exc:  # noqa: BLE001 — any startup failure is non-fatal
            # ``diagnostic`` is secret-free by construction (built without
            # ``env``/``headers``), so it's safe to log (story 32). Unwrap
            # ExceptionGroups so the message names the real cause, not the
            # anyio TaskGroup wrapper.
            detail = f" ({self._diagnostic})" if self._diagnostic else ""
            logger.warning(
                "MCP server %r failed to start%s: %s",
                self.__name__,
                detail,
                _flatten_exc(exc),
            )
            # Unwind whatever context was entered before the failure.
            try:
                await cm.__aexit__(type(exc), exc, exc.__traceback__)
            except Exception as close_exc:  # noqa: BLE001 — best-effort cleanup
                logger.debug(
                    "MCP server %r cleanup after failed start: %s",
                    self.__name__,
                    close_exc,
                )
            self._cm = None
            self._session = None
            return []
        self._cm = cm
        label = self.__name__.removeprefix("mcp:")  # → tool-name prefix (ADR-0006)
        return [
            MCPClientTool(
                self,
                label,
                remote.name,
                remote.description or "",
                remote.inputSchema,
            )
            for remote in listed.tools
        ]

    async def aclose(self) -> None:
        """Close the session + transport. Idempotent; safe if never started."""
        if self._cm is None:
            return
        try:
            await self._cm.__aexit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 — teardown must not raise
            logger.debug("MCP server %r close error: %s", self.__name__, exc)
        finally:
            self._cm = None
            self._session = None


def _make_mcp_server(
    label: str,
    *,
    open_transport: Callable[[], Any],
    diagnostic: str,
    source: str | None,
) -> MCPServer:
    """Label guard + ``mcp:`` handle prefix for stdio/http builders (ADR-0006)."""
    where = f" in {source}" if source else ""
    if not label:
        msg = f"MCP server label is empty{where}; set a non-empty 'name:'"
        raise ValueError(msg)
    if ":" in label:
        msg = f"MCP server label {label!r} contains ':'{where}"
        raise ValueError(msg)
    server = MCPServer(
        name=f"mcp:{label}", open_transport=open_transport, diagnostic=diagnostic
    )
    server._source = source
    return server


def _build_mcp_stdio_server(spec: dict[str, Any], source: str | None) -> MCPServer:
    """Build an ``MCPServer`` from a ``type: mcp.stdio`` YAML mapping.

    ``command`` (required) is the server executable; ``args`` its CLI
    arguments; ``env`` the subprocess environment (secrets — never logged,
    story 32). The handle name is ``mcp:`` + ``name`` (or the file stem) —
    prefixed so it can't collide with a real tool name. Does NOT connect —
    that's deferred to Agent startup (ADR-0005). Raises ``ValueError`` on a
    malformed declaration, naming the field + source.
    """
    from mcp import StdioServerParameters

    _check_unknown_keys(spec, _MCP_STDIO_KEYS, source, what="MCP stdio tool")
    command = str(_require(spec, "command", source, what="MCP stdio tool"))
    where = f" in {source}" if source else ""
    raw_args = spec.get("args") or []
    if not isinstance(raw_args, list):
        msg = f"MCP stdio tool: 'args' must be a list{where}"
        raise ValueError(msg)
    args = [str(a) for a in raw_args]
    raw_env = spec.get("env") or {}
    if not isinstance(raw_env, dict):
        msg = f"MCP stdio tool: 'env' must be a mapping{where}"
        raise ValueError(msg)
    env: dict[str, str] = {}
    for k, v in raw_env.items():
        if not isinstance(v, str):
            msg = (
                f"MCP stdio tool: 'env.{k}' must be a string{where}, "
                f"got {type(v).__name__}"
            )
            raise ValueError(msg)
        env[str(k)] = v
    label = str(spec.get("name") or (Path(source).stem if source else "mcp"))
    # cothis: warn-don't-skip — server may launch via full path; connect-failure degrades (ADR-0005).
    if _resolve_executable(command) is None:
        logger.warning(
            "MCP stdio server %r: command %r not on PATH%s; "
            "will attempt to launch at run time",
            f"mcp:{label}",
            command,
            where,
        )
    params = StdioServerParameters(command=command, args=args, env=env or None)

    @asynccontextmanager
    async def open_transport() -> AsyncIterator[Any]:
        # Lazy import: the stdio client pulls the full SDK transport stack;
        # only a real stdio server started at Agent runtime pays for it.
        from mcp.client.stdio import stdio_client

        async with stdio_client(params) as (read, write):
            yield (read, write)

    # ``env`` is deliberately excluded from ``diagnostic`` — secrets (story 32).
    return _make_mcp_server(
        label,
        open_transport=open_transport,
        diagnostic=f"command={command!r} args={args!r}",
        source=source,
    )


def _build_mcp_http_server(spec: dict[str, Any], source: str | None) -> MCPServer:
    """Build an ``MCPServer`` from a ``type: mcp.http`` YAML mapping.

    ``url`` (required) is the remote server endpoint; ``headers`` an optional
    mapping sent on every request (secrets like ``Authorization`` — never
    logged, story 32). The handle name is ``mcp:`` + ``name`` (or the file
    stem). Does NOT connect — deferred to Agent startup (ADR-0005). Reuses the
    stdio path's session lifecycle, discovery, dispatch, and normalization;
    only the transport (``streamablehttp_client``) differs. Raises
    ``ValueError`` on a malformed declaration, naming the field + source.
    """
    _check_unknown_keys(spec, _MCP_HTTP_KEYS, source, what="MCP HTTP tool")
    url = str(_require(spec, "url", source, what="MCP HTTP tool"))
    where = f" in {source}" if source else ""
    raw_headers = spec.get("headers") or {}
    if not isinstance(raw_headers, dict):
        msg = f"MCP HTTP tool: 'headers' must be a mapping{where}"
        raise ValueError(msg)
    headers = {str(k): str(v) for k, v in raw_headers.items()}
    label = str(spec.get("name") or (Path(source).stem if source else "mcp"))

    @asynccontextmanager
    async def open_transport() -> AsyncIterator[Any]:
        # Lazy import (see the stdio builder). ``streamablehttp_client`` yields
        # a 3-tuple ``(read, write, get_session_id)``; the session-id callback
        # isn't needed here, so drop it and yield the same ``(read, write)``
        # pair the stdio transport does — keeping ``_default_connect`` uniform.
        # cothis: ceiling — dropping ``get_session_id`` forecloses HTTP session
        # resumption/reconnect after a dropped connection. Each ``Agent.run``
        # opens a fresh HTTP transport; there's no way to resume a previous
        # server-assigned session id. Upgrade path: thread ``get_session_id``
        # through ``_default_connect`` and reconnect with it if the transport
        # drops mid-session.
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(url, headers=headers or None) as (
            read,
            write,
            _get_session_id,
        ):
            yield (read, write)

    # ``headers`` is excluded from ``diagnostic`` (secrets). The url itself
    # is scrubbed — userinfo (``token@host``) and query (``?key=…``) carry
    # credentials that must never reach a log (story 32).
    return _make_mcp_server(
        label,
        open_transport=open_transport,
        diagnostic=f"url={_scrub_url(url)!r}",
        source=source,
    )
