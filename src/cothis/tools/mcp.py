"""MCP (Model Context Protocol) server + client-tool subsystem.

Built on the MCP SDK's ``ClientSessionGroup``: the SDK manages connections,
tool aggregation, name prefixing, and teardown; cothis adds YAML config
parsing, secret-free diagnostics, and dispatchable-tool wrapping (so MCP
tools carry lifecycle hooks like every other tool).

A YAML declaration ``type: mcp.stdio`` / ``type: mcp.http`` parses into an
``MCPServer`` (transport params + diagnostic label). ``MCPServer`` is not a
dispatchable tool — it satisfies the ``Tool`` protocol structurally so it
rides the discovery pipeline, but its ``__call__`` raises. At Agent startup
the ``ClientSessionGroup`` consumes each server's params via
``connect_into``, lists remote tools, and aggregates them under prefixed
names (``{label}.{remote}`` via ``component_name_hook``). Each aggregated
tool is wrapped in an ``MCPClientTool`` so it inherits ``_HookableTool`` for
lifecycle hooks. See ADR-0005 (deferred connect) and ADR-0006 (prefix).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from cothis.tools.core import (
    _check_unknown_keys,
    _HookableTool,
    _require,
    _resolve_executable,
    logger,
)

if TYPE_CHECKING:
    from typing import NoReturn

    from mcp.client.session_group import ClientSessionGroup
    from mcp.types import CallToolResult
    from mcp.types import Tool as McpTool


_MCP_STDIO_KEYS = {"type", "name", "description", "command", "args", "env"}
_MCP_HTTP_KEYS = {"type", "name", "description", "url", "headers"}


def _normalize_mcp_result(result: CallToolResult) -> str:
    """Flatten an MCP ``CallToolResult`` into a single string for the LLM.

    - Join every content block's ``.text`` with newlines.
    - Empty content list → ``"(no output)"`` (the tool ran but said nothing).
    - ``isError`` true → prefix ``"Error: "`` so the model sees it as a
      failure it can act on.

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
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    body = "\n".join(parts) if parts else "(no output)"
    # ``isError`` is camelCase on the MCP pydantic model (verified against
    # mcp 1.28.1 — ``CallToolResult`` fields: content/structuredContent/isError).
    if result.isError:
        return f"Error: {body}"
    return body


class MCPClientTool(_HookableTool):
    """A single remote MCP tool, dispatched over a shared ``ClientSessionGroup``.

    Produced by ``MCPServer.connect_into`` — one instance per remote tool the
    server exposes. Inherits ``_HookableTool`` so ``_execute`` runs its hook
    chains uniformly with every other tool (CONTEXT.md "no per-source
    branching in ``_execute``"). Carries a pre-built ``__cothis_schema__``
    from the server's ``inputSchema`` (already OpenAI-compatible JSON Schema).

    ``__name__`` is the prefixed name (``{label}.{remote}``, assigned by the
    SDK's ``component_name_hook``); ``_remote_name`` is the same prefixed
    name sent to ``group.call_tool`` (the group routes by prefixed name).
    """

    __name__: str
    __doc__: str
    __cothis_schema__: dict[str, Any]

    def __init__(self, group: ClientSessionGroup, mcp_tool: McpTool) -> None:
        super().__init__()
        self._group = group
        self.__name__ = mcp_tool.name
        self.__doc__ = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
        self._remote_name = mcp_tool.name
        # cothis: ceiling — the server's ``inputSchema`` is passed through
        # verbatim as the OpenAI ``parameters`` field. The MCP spec defines
        # ``inputSchema`` as a JSON Schema, which is structurally close to
        # OpenAI's ``parameters`` — but a non-conformant server may ship a
        # schema missing ``type: "object"``, carrying ``$ref``/``$defs``, or
        # with provider-specific quirks, and those leak straight to the model.
        # cothis does no normalisation today. Upgrade path: run the schema
        # through a normaliser (drop ``$defs`` by inlining, default missing
        # ``type`` to ``object``, validate it's a function-shaped schema) so
        # an odd server can't corrupt the tool-call contract.
        self.__cothis_schema__ = {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": self.__doc__,
                "parameters": mcp_tool.inputSchema
                or {"type": "object", "properties": {}},
            },
        }

    async def __call__(self, **kwargs: Any) -> str:
        result = await self._group.call_tool(self._remote_name, kwargs)
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
    # Keep the netloc string rather than rebuilding from ``parts.hostname`` —
    # the latter drops IPv6 brackets (``::1`` for ``[::1]:8000``).
    netloc = parts.netloc.rsplit("@", 1)[-1] if "@" in parts.netloc else parts.netloc
    return urlunsplit(parts._replace(netloc=netloc, query="", fragment=""))


class MCPServer(_HookableTool):
    """A declared MCP server — transport params + diagnostic label.

    Not a callable tool itself — it's a *producer* of tools. Flows through
    discovery (``discover_tools``) as an opaque item (it satisfies the
    ``Tool`` protocol minimally: ``__name__`` + ``__call__``), then the Agent
    resolves it at startup: a ``ClientSessionGroup`` consumes its params via
    ``connect_into``, lists remote tools, and each becomes an
    ``MCPClientTool``. ``__name__`` is a diagnostic label (``mcp:`` + ``name:``
    or the file stem), prefixed so it can never collide with the names of the
    tools it produces — or with any other dispatchable tool in the registry.

    Session lifecycle is owned by the ``ClientSessionGroup`` the Agent holds:
    one ``async with group`` covers every server's connection + teardown.
    See ADR-0005.

    The *transport* is the only thing that differs between MCP kinds, so it's
    the only injected piece: ``params`` is the SDK's ``StdioServerParameters``
    or ``StreamableHttpParameters``. ``diagnostic`` is a secret-free string
    (command + args, or scrubbed url — never ``env``/``headers``) logged if
    the server fails to connect.
    """

    __name__: str

    def __init__(
        self,
        *,
        name: str,
        params: Any,
        diagnostic: str = "",
    ) -> None:
        super().__init__()
        self.__name__ = name
        self.params = params
        self._diagnostic = diagnostic

    @property
    def _label(self) -> str:
        """Raw YAML ``name:`` label, without the ``mcp:`` handle prefix.

        ``__name__`` is the discovery handle (``mcp:{label}``), prefixed so it
        can't collide with a real tool name in the registry. The tool-name
        prefix uses the bare label — what the user wrote in YAML ``name:``,
        stripped of the handle decoration. Used as the fallback when the server
        reports an empty ``Implementation.name`` (ADR-0006).
        """
        return self.__name__[4:] if self.__name__.startswith("mcp:") else self.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError(
            f"MCP server {self.__name__!r} is a server declaration, not a callable tool"
        )

    async def connect_into(self, group: ClientSessionGroup) -> list[MCPClientTool]:
        """Connect this server via ``group``, return its tools as ``MCPClientTool``.

        The group handles transport, session, and tool-name prefixing (via its
        ``component_name_hook`` — ``{server_reported_name}.{tool}``). On
        failure (command not found, subprocess crash, connection refused,
        protocol error) logs at ``WARNING`` naming the server + its
        ``diagnostic`` — never ``env``/``headers`` secrets (story 32) — and
        returns ``[]`` so the rest of the agent's tools still load (story 30).

        cothis: ceiling — this method reaches into SDK internals: it
        snapshots ``group.tools`` before/after ``connect_to_server`` and
        ``model_copy``s each new entry to inject the prefixed name
        ``MCPClientTool`` will see. These are private attributes on the SDK's
        ``ClientSessionGroup``; if the SDK reshapes its tool store or stops
        keying by the prefixed name, this breaks silently (tools registered
        under the wrong name, or not at all). Upgrade path: SDK exposes an
        official "connect one server, return its prefixed tools" API
        (``connect_to_server`` returning the tool list would suffice); adopt
        it and drop the snapshot diff.
        """
        # Snapshot the group's tools before connecting so we can identify
        # which tools this server contributed (prefix is the server's
        # *self-reported* name, which we can't predict from cothis's YAML
        # ``name:`` field — they may differ).
        before = set(group.tools)
        try:
            await group.connect_to_server(self.params)
        except Exception as exc:  # noqa: BLE001 — any startup failure is non-fatal
            detail = f" ({self._diagnostic})" if self._diagnostic else ""
            logger.warning(
                "MCP server %r failed to start%s: %s",
                self.__name__,
                detail,
                _flatten_exc(exc),
            )
            return []
        # The group stores ``Tool.name`` bare but keys its dict by the prefixed
        # name; copy each new tool with its prefixed key so ``MCPClientTool``
        # sees the name the LLM will call it by.
        new_tools = [
            tool.model_copy(update={"name": name})
            for name, tool in group.tools.items()
            if name not in before
        ]
        return [MCPClientTool(group, tool) for tool in new_tools]


def _make_mcp_server(
    label: str,
    *,
    params: Any,
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
    server = MCPServer(name=f"mcp:{label}", params=params, diagnostic=diagnostic)
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
    if _resolve_executable(command) is None:
        logger.warning(
            "MCP stdio server %r: command %r not on PATH%s; "
            "will attempt to launch at run time",
            f"mcp:{label}",
            command,
            where,
        )
    params = StdioServerParameters(command=command, args=args, env=env or None)

    return _make_mcp_server(
        label,
        params=params,
        diagnostic=f"command={command!r} args={args!r}",
        source=source,
    )


def _build_mcp_http_server(spec: dict[str, Any], source: str | None) -> MCPServer:
    """Build an ``MCPServer`` from a ``type: mcp.http`` YAML mapping.

    ``url`` (required) is the remote server endpoint; ``headers`` an optional
    mapping sent on every request (secrets like ``Authorization`` — never
    logged, story 32). The handle name is ``mcp:`` + ``name`` (or the file
    stem). Does NOT connect — deferred to Agent startup (ADR-0005). Reuses
    the stdio path's session lifecycle, discovery, dispatch, and
    normalization; only the transport (``StreamableHttpParameters``) differs.
    Raises ``ValueError`` on a malformed declaration, naming the field + source.
    """
    from mcp.client.session_group import StreamableHttpParameters

    _check_unknown_keys(spec, _MCP_HTTP_KEYS, source, what="MCP HTTP tool")
    url = str(_require(spec, "url", source, what="MCP HTTP tool"))
    where = f" in {source}" if source else ""
    raw_headers = spec.get("headers") or {}
    if not isinstance(raw_headers, dict):
        msg = f"MCP HTTP tool: 'headers' must be a mapping{where}"
        raise ValueError(msg)
    headers = {str(k): str(v) for k, v in raw_headers.items()}
    label = str(spec.get("name") or (Path(source).stem if source else "mcp"))
    params = StreamableHttpParameters(url=url, headers=headers or None)

    return _make_mcp_server(
        label,
        params=params,
        diagnostic=f"url={_scrub_url(url)!r}",
        source=source,
    )
