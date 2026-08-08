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
lifecycle hooks.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cothis.tools.core import (
    ResourceHandle,
    _check_unknown_keys,
    _HookableTool,
    _require,
    logger,
)
from cothis.tools.fs._hygiene import _MAX_BYTES

if TYPE_CHECKING:
    from typing import NoReturn

    from mcp.client.session_group import ClientSessionGroup
    from mcp.types import CallToolResult
    from mcp.types import Tool as McpTool


_MCP_STDIO_KEYS = {"type", "name", "description", "command", "args", "env", "keepalive"}
_MCP_HTTP_KEYS = {"type", "name", "description", "url", "headers", "keepalive"}


def _normalize_mcp_result(result: CallToolResult) -> str:
    """Flatten an MCP ``CallToolResult`` into a single string for the LLM.

    - Join every content block's ``.text`` with newlines.
    - Non-text blocks (image, embedded resource) get a placeholder
      describing their shape — the agent loop is text-only today, but
      the model still needs to know *that* content was returned.
    - Empty content list → ``"(no output)"`` (the tool ran but said nothing).
    - ``isError`` true → prefix ``"Error: "`` so the model sees it as a
      failure it can act on.
    """
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
            continue
        # cothis: non-text placeholder (#92). The agent loop carries
        # strings today; image/resource blocks would otherwise vanish
        # and the model would see "(no output)" for a tool that
        # returned a 70-byte PNG.
        btype = getattr(block, "type", "unknown")
        if btype == "image":
            mime = getattr(block, "mimeType", "unknown")
            size = len(getattr(block, "data", "") or "")
            parts.append(f"[image: mime={mime}, {size} bytes base64]")
        elif btype == "resource":
            # cothis: placeholder carries only the URI + mime type — not
            # the body. ``resource!r`` would serialise the full pydantic
            # model (uri + text/blob), leaking file contents of an
            # EmbeddedResource wrapping a local file.
            resource = getattr(block, "resource", None)
            uri = getattr(resource, "uri", "unknown")
            mime = getattr(resource, "mimeType", "unknown")
            parts.append(
                f"[embedded resource: uri={uri!s}, mime={mime}]"
            )
        else:
            parts.append(f"[non-text block: type={btype}]")
    body = "\n".join(parts) if parts else "(no output)"
    # Cap the result at ``_MAX_BYTES`` bytes — MCP servers are external and
    # can return unbounded output (search dumps, log tails, file echoes) that
    # lands verbatim in the model context and poisons every subsequent turn
    # (#421). Byte-based (not character-based) to match ``fs.create``'s
    # ``len(content.encode("utf-8"))`` convention and #421's stated fix; a
    # character cap would let 1–4 MiB of multibyte text through and silently
    # drift the prompt budget. Slicing the encoded form + ``decode(...,
    # "ignore")`` is codepoint-safe — it drops at most one partial trailing
    # multibyte sequence. The body came from a remote server, so the marker
    # points at narrowing the query (there is no local path to ``fs.read``).
    encoded = body.encode("utf-8")
    if len(encoded) > _MAX_BYTES:
        body = encoded[:_MAX_BYTES].decode("utf-8", "ignore") + (
            f"\n... [truncated: MCP result exceeded {_MAX_BYTES} bytes]"
        )
    # ``isError`` is camelCase on the MCP pydantic model (verified against
    # mcp 1.28.1 — ``CallToolResult`` fields: content/structuredContent/isError).
    if result.isError:
        return f"Error: {body}"
    return body


# Resource caps for the input-schema normaliser. Sibling-in-spirit to
# ``_MAX_BYTES`` in ``fs/_hygiene``: schema-specific (only one consumer
# today, ``_normalize_input_schema``), so they live here rather than in the
# shared caps module. Module-const (not config) to match the ``_MAX_BYTES``
# precedent — operator tuning via env is out of scope for this hardening pass.
# 64 KiB — bounds a pathological server schema's serialised size before it
# reaches the model context. A focused tool schema is a few hundred bytes;
# anything this large is almost always a server bug.
_MAX_SCHEMA_BYTES = 64 * 1024
# 32 — bounds nesting / self-referential ``$ref`` cycles. Real tool schemas
# rarely exceed 4-5 levels; this is a generous safety margin that keeps a
# cyclical ``$defs`` entry from looping forever.
_MAX_SCHEMA_DEPTH = 32


class _SchemaTooDeep(Exception):
    """Raised when an ``inputSchema``'s nesting exceeds ``_MAX_SCHEMA_DEPTH``.

    Caught at the ``_normalize_input_schema`` entry point so the recursion
    terminates with a single minimal-schema fallback rather than propagating
    ``RecursionError`` (a cyclical ``$defs`` entry would otherwise loop).
    """


def _normalize_input_schema(schema: dict[str, Any], *, tool_name: str) -> dict[str, Any]:
    """Normalise an MCP server's ``inputSchema`` into a clean Anthropic ``input_schema``.

    The MCP spec lets any server ship a JSON Schema as ``inputSchema``; that
    dict flows verbatim into ``__cothis_schema__["input_schema"]`` and out to
    every provider (Anthropic via ``schema_for``; OpenAI/Google via
    ``anthropic_tools_to_openai``/``anthropic_tools_to_google`` which pass
    ``input_schema`` straight through as ``parameters``). A non-conformant
    server — missing top-level ``type:object``, carrying ``$ref``/``$defs``
    the Anthropic tool shape rejects, malformed ``required``, oversized or
    self-referential — would corrupt the tool-call contract for every model.
    This is the single chokepoint that stops that.

    Contract (run inside ``MCPClientTool.__init__`` before assigning
    ``__cothis_schema__``):

    1. ``deepcopy`` the input so the server's dict (held by the SDK) is never
       mutated — there is a dedicated deepcopy-contract test.
    2. Default top-level ``type`` to ``"object"`` (synthesise
       ``{type:object, properties:{}}`` when missing/empty/non-object).
    3. Inline local ``$ref`` against ``$defs``/``definitions`` (recursing,
       bounded by ``_MAX_SCHEMA_DEPTH``), then strip
       ``$defs``/``definitions``/``$ref``/``$schema``/``$id`` (keys the
       Anthropic tool shape rejects). Remote/http refs are NEVER resolved —
       only ``#/...`` local fragments; anything else is dropped with a
       WARNING (security: no network ref resolution).
    4. Replace-with-``{}`` any unresolvable ``$ref`` and log at ``WARNING``
       naming the tool + the offending ref.
    5. Coerce ``required`` to a unique list of strings, dropping non-string
       entries with a ``WARNING``.
    6. If the serialised result exceeds ``_MAX_SCHEMA_BYTES`` or nesting
       exceeds ``_MAX_SCHEMA_DEPTH``, replace with ``{"type":"object",
       "properties":{}}`` + ``WARNING``.

    Pure: stdlib-only (``json`` + ``copy.deepcopy`` + the module ``logger``).
    """
    # Deepcopy first so the server's dict is never mutated in place — defense
    # against future changes and a documented contract (deepcopy-contract test).
    work = copy.deepcopy(schema)

    # Harvest ``$defs``/``definitions`` from the root once — all local
    # ``$ref`` fragments resolve against this map (the JSON Schema document
    # model: refs are root-relative).
    defs: dict[str, Any] = {}
    for defkey in ("$defs", "definitions"):
        block = work.get(defkey)
        if isinstance(block, dict):
            for name, sub in block.items():
                defs[str(name)] = sub

    def resolve_ref(ref: str, depth: int) -> Any:
        if not ref.startswith("#/"):
            logger.warning(
                "MCP tool %r: inputSchema $ref %r is not a local fragment; "
                "dropped (remote refs are never resolved)",
                tool_name,
                ref,
            )
            return {}
        # ``#/$defs/X`` or ``#/definitions/X`` — take the trailing segment
        # as the def key (matches the harvest loop above).
        name = ref[2:].split("/", 1)[-1]
        target = defs.get(name)
        if target is None:
            logger.warning(
                "MCP tool %r: inputSchema $ref %r not found in "
                "$defs/definitions; replaced with {}",
                tool_name,
                ref,
            )
            return {}
        # ``walk`` (and thus the ``_MAX_SCHEMA_DEPTH`` counter) is the sole
        # bound for transitive / self-referential refs — a cyclical def
        # terminates there instead of looping forever.
        return walk(target, depth + 1)

    def coerce_required(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            logger.warning(
                "MCP tool %r: inputSchema 'required' is %s, expected a list "
                "of strings; dropped",
                tool_name,
                type(value).__name__,
            )
            return []
        seen: set[str] = set()
        out: list[str] = []
        dropped = False
        for item in value:
            # ``isinstance(True, str)`` is False — bools are dropped here
            # alongside ints/None/dicts (only real strings survive).
            if isinstance(item, str):
                if item not in seen:
                    seen.add(item)
                    out.append(item)
            else:
                dropped = True
        if dropped:
            logger.warning(
                "MCP tool %r: inputSchema 'required' had non-string entries; "
                "dropped",
                tool_name,
            )
        return out

    def walk(node: Any, depth: int) -> Any:
        # Explicit depth counter (never raw recursion) — bounds transitive
        # / self-referential refs and satisfies ty. The raised exception is
        # caught once at the entry point.
        if depth > _MAX_SCHEMA_DEPTH:
            raise _SchemaTooDeep()
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                # A ``$ref`` node replaces the entire node (JSON Schema
                # semantics: sibling keys are ignored under draft-07 ref).
                return resolve_ref(ref, depth)
            out: dict[str, Any] = {}
            for key, value in node.items():
                if key in ("$defs", "definitions", "$ref", "$schema", "$id"):
                    # Provider-rejecting / inline-definition keys: dropped
                    # after their refs were harvested at the root.
                    continue
                if key == "required":
                    coerced = coerce_required(value)
                    if coerced:
                        out[key] = coerced
                    continue
                out[key] = walk(value, depth + 1)
            return out
        if isinstance(node, list):
            return [walk(item, depth + 1) for item in node]
        return node

    try:
        normalised = walk(work, 0)
    except _SchemaTooDeep:
        logger.warning(
            "MCP tool %r: inputSchema nesting exceeded depth ceiling %d; "
            "replaced with a minimal object schema",
            tool_name,
            _MAX_SCHEMA_DEPTH,
        )
        return {"type": "object", "properties": {}}
    if not isinstance(normalised, dict):
        # ``$ref`` at the root resolved to a non-object leaf; default to an
        # empty object so the top-level is always a function-shaped schema.
        normalised = {"type": "object", "properties": {}}
    if not normalised.get("type"):
        normalised["type"] = "object"
    if normalised.get("type") != "object":
        # The root must be an object schema; force it so providers don't
        # reject the tool definition outright.
        normalised = {"type": "object", "properties": {}}
    normalised.setdefault("properties", {})
    # Final byte-size ceiling on the serialised form.
    try:
        encoded = json.dumps(normalised).encode("utf-8")
    except (TypeError, ValueError):
        logger.warning(
            "MCP tool %r: inputSchema is not JSON-serialisable; replaced "
            "with a minimal object schema",
            tool_name,
        )
        return {"type": "object", "properties": {}}
    if len(encoded) > _MAX_SCHEMA_BYTES:
        logger.warning(
            "MCP tool %r: inputSchema serialised to %d bytes (ceiling %d); "
            "replaced with a minimal object schema",
            tool_name,
            len(encoded),
            _MAX_SCHEMA_BYTES,
        )
        return {"type": "object", "properties": {}}
    return normalised


class MCPSessionHandle(ResourceHandle):
    """A ``ResourceHandle`` backed by one MCP server's session.

    ``acquire`` connects the server into the shared ``ClientSessionGroup``
    (listing its tools as a side effect); ``release`` disconnects that one
    session. The HandleManager owns one *dynamically generated subclass* per
    server (so each server is one pool entry keyed by its own class), with
    ``_group`` and ``_params`` set as class attributes. ``keepalive`` / ``pin``
    come from the YAML declaration, so MCP sessions follow the same
    keepalive + LRU lifecycle as any other handle.
    """

    # Set on the dynamic subclass generated per server in ``_ensure_mcp``.
    _group: ClientSessionGroup
    _params: Any
    # Fallback-label cell shared with the group's ``component_name_hook``.
    # The hook fires inside ``connect_to_server`` with nothing identifying
    # *which* cothis server is connecting, so ``acquire`` publishes its own
    # label first — an empty-name server keeps its own prefix across
    # re-acquires instead of inheriting the last-connected server's
    # prefix.
    _fallback: dict[str, str]
    _fallback_label: str
    _session: Any = None

    async def acquire(self) -> None:
        self._fallback["label"] = self._fallback_label
        self._session = await self._group.connect_to_server(self._params)

    async def release(self) -> None:
        if self._session is not None:
            try:
                await self._group.disconnect_from_server(self._session)
            finally:
                self._session = None


class MCPClientTool(_HookableTool):
    """A single remote MCP tool, dispatched over a shared ``ClientSessionGroup``.

    Produced by ``MCPServer.connect_into`` — one instance per remote tool the
    server exposes. Inherits ``_HookableTool`` so ``_execute`` runs its hook
    chains uniformly with every other tool (CONTEXT.md "no per-source
    branching in ``_execute``"). Carries a pre-built ``__cothis_schema__``
    from the server's ``inputSchema`` (Anthropic tool shape's ``input_schema``,
    a JSON Schema passed through verbatim).

    ``__name__`` is the prefixed name (``{label}.{remote}``, assigned by the
    SDK's ``component_name_hook``); ``_remote_name`` is the same prefixed
    name sent to ``group.call_tool`` (the group routes by prefixed name).
    """

    __name__: str
    __doc__: str
    __cothis_schema__: dict[str, Any]
    # Set by ``_ensure_mcp`` to the per-server ``MCPSessionHandle`` subclass
    # so ``ensure_handle_ready`` / ``mark_inflight`` manage the session.
    _handle_cls: Any = None

    def __init__(self, group: ClientSessionGroup, mcp_tool: McpTool) -> None:
        super().__init__()
        self._group = group
        self.__name__ = mcp_tool.name
        self.__doc__ = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
        self._remote_name = mcp_tool.name
        # cothis: the server's ``inputSchema`` is a JSON Schema, which is
        # structurally what Anthropic's ``input_schema`` expects — but the
        # MCP spec lets any server ship one, and a non-conformant schema
        # (missing ``type:object``, carrying ``$ref``/``$defs``, malformed
        # ``required``, oversized or self-referential) would leak straight to
        # every provider and corrupt the tool-call contract. Normalise once
        # here at the single chokepoint; see ``_normalize_input_schema`` for
        # the contract (deepcopy → default type → inline ``$ref`` → strip
        # provider-rejecting keys → coerce ``required`` → size/depth caps).
        # The raw server form is kept on ``_raw_input_schema`` for diagnostic
        # (so a WARNING can name what diverged) and is NEVER sent to the model.
        self._raw_input_schema = mcp_tool.inputSchema
        self.__cothis_schema__ = {
            "name": mcp_tool.name,
            "description": self.__doc__,
            "input_schema": _normalize_input_schema(
                mcp_tool.inputSchema or {"type": "object", "properties": {}},
                tool_name=mcp_tool.name,
            ),
        }

    async def __call__(self, **kwargs: Any) -> str:
        result = await self._group.call_tool(self._remote_name, kwargs)
        return _normalize_mcp_result(result)


_URL_RE = re.compile(r"https?://[^\s'\"<>]+")


def _flatten_exc(exc: BaseException) -> str:
    """Describe an exception, unwrapping ``ExceptionGroup``s to the real cause.

    anyio runs the MCP transport inside a task group, so a connection/protocol
    failure surfaces as an ``ExceptionGroup`` whose own message — ``"unhandled
    errors in a TaskGroup (1 sub-exception)"`` — hides what actually went
    wrong. Recurse into ``.exceptions`` and join the leaf messages so the
    startup warning names something the operator (and the model) can act on.

    URLs in leaf messages are scrubbed: httpx exceptions embed the full
    request URL, and httpx masks userinfo passwords but NOT query strings —
    an ``?api_key=…`` in the ``url:`` field would otherwise reach the log in
    cleartext (story 32).
    """
    subs = getattr(exc, "exceptions", None)
    if subs:
        return "; ".join(_flatten_exc(s) for s in subs)
    return _URL_RE.sub(lambda m: _scrub_url(m.group(0)), f"{type(exc).__name__}: {exc}")


def _scrub_url(url: str) -> str:
    """Strip userinfo and query from a url for safe logging.

    A url may carry credentials in the userinfo (``https://token@host``) or
    in the query string (``?api_key=secret``); both are dropped so the
    diagnostic keeps only ``scheme://host:port/path`` (story 32). Applied to
    every url-derived string that reaches a log: the ``diagnostic``, and
    exception messages via ``_flatten_exc``.
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
        keepalive: float = 600.0,
    ) -> None:
        super().__init__()
        self.__name__ = name
        self.params = params
        self._diagnostic = diagnostic
        self.keepalive = keepalive

    @property
    def _label(self) -> str:
        """Raw YAML ``name:`` label, without the ``mcp:`` handle prefix.

        ``__name__`` is the discovery handle (``mcp:{label}``), prefixed so it
        can't collide with a real tool name in the registry. The tool-name
        prefix uses the bare label — what the user wrote in YAML ``name:``,
        stripped of the handle decoration. Used as the fallback when the server
        reports an empty ``Implementation.name``.
        """
        return self.__name__[4:] if self.__name__.startswith("mcp:") else self.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError(
            f"MCP server {self.__name__!r} is a server declaration, not a callable tool"
        )

    async def connect_into(
        self, group: ClientSessionGroup
    ) -> tuple[list[MCPClientTool], Any]:
        """Connect this server via ``group``; return ``(tools, session)``.

        The session is returned so the caller (``Agent._ensure_mcp``) can adopt
        it as the server's ``MCPSessionHandle`` first acquire — the startup
        connection that lists tools is not wasted. On failure logs at
        ``WARNING`` naming the server + its ``diagnostic`` — never
        ``env``/``headers`` secrets (story 32) — and returns ``([], None)`` so
        the rest of the agent's tools still load (story 30).

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
        # cothis: shape guard (#63). ``group.tools`` is a private SDK
        # attribute; an SDK upgrade that reshapes it (list, renamed,
        # missing) would silently break tool discovery. Fail loud at
        # first connect with a diagnostic naming the divergence.
        tools_attr = getattr(group, "tools", None)
        if not isinstance(tools_attr, dict):
            raise RuntimeError(
                f"MCP SDK shape changed: ClientSessionGroup.tools is "
                f"{type(tools_attr).__name__}, expected dict "
                f"(prefixed-name → Tool). connect_into's snapshot "
                f"diff needs updating; see issue #63."
            )
        before = set(tools_attr)
        try:
            session = await group.connect_to_server(self.params)
        except asyncio.CancelledError:
            # cothis: a sibling task in the MCP SDK's task group died
            # (typically the post_writer task on a flaky remote HTTP
            # transport), anyio cancelled the group, and the handshake
            # await surfaced CancelledError. Since 3.8 CancelledError
            # inherits from BaseException (not Exception), the broad
            # ``Exception`` clause below does NOT catch it — without this
            # branch it escapes connect_into → _ensure_mcp and cancels the
            # whole agent turn, blocking even unrelated local tools
            # (fs.read / fs.delete). Same treatment as
            # HandleManager._release (core.py, the #185 fix). The server
            # contributes no tools; the rest of the agent's tools still
            # load (#370).
            detail = f" ({self._diagnostic})" if self._diagnostic else ""
            logger.warning(
                "MCP server %r handshake cancelled%s (likely a flaky remote "
                "transport); contributing no tools.",
                self.__name__,
                detail,
            )
            return [], None
        except Exception as exc:  # noqa: BLE001 — any startup failure is non-fatal
            detail = f" ({self._diagnostic})" if self._diagnostic else ""
            logger.warning(
                "MCP server %r failed to start%s: %s",
                self.__name__,
                detail,
                _flatten_exc(exc),
            )
            return [], None
        # The group stores ``Tool.name`` bare but keys its dict by the prefixed
        # name; copy each new tool with its prefixed key so ``MCPClientTool``
        # sees the name the LLM will call it by.
        new_tools = [
            tool.model_copy(update={"name": name})
            for name, tool in group.tools.items()
            if name not in before
        ]
        return [MCPClientTool(group, tool) for tool in new_tools], session


def _make_mcp_server(
    label: str,
    *,
    params: Any,
    diagnostic: str,
    source: str | None,
    keepalive: float = 600.0,
) -> MCPServer:
    """Label guard + ``mcp:`` handle prefix for stdio/http builders."""
    where = f" in {source}" if source else ""
    if not label:
        msg = f"MCP server label is empty{where}; set a non-empty 'name:'"
        raise ValueError(msg)
    if ":" in label:
        msg = f"MCP server label {label!r} contains ':'{where}"
        raise ValueError(msg)
    server = MCPServer(
        name=f"mcp:{label}",
        params=params,
        diagnostic=diagnostic,
        keepalive=keepalive,
    )
    server._source = source
    return server


def _parse_keepalive(spec: dict[str, Any], source: str | None) -> float:
    """Parse the optional ``keepalive:`` seconds field with validation."""
    raw = spec.get("keepalive")
    if raw is None:
        return 600.0
    where = f" in {source}" if source else ""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        msg = f"MCP server: 'keepalive' must be a number (seconds){where}"
        raise ValueError(msg) from None
    if value <= 0:
        msg = f"MCP server: 'keepalive' must be > 0{where}"
        raise ValueError(msg)
    return value


def _build_mcp_stdio_server(spec: dict[str, Any], source: str | None) -> MCPServer:
    """Build an ``MCPServer`` from a ``type: mcp.stdio`` YAML mapping.

    ``command`` (required) is the server executable; ``args`` its CLI
    arguments; ``env`` the subprocess environment (secrets — never logged,
    story 32). The handle name is ``mcp:`` + ``name`` (or the file stem) —
    prefixed so it can't collide with a real tool name. Does NOT connect —
    that's deferred to Agent startup. Raises ``ValueError`` on a
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
    if shutil.which(command) is None:
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
        keepalive=_parse_keepalive(spec, source),
    )


def _build_mcp_http_server(spec: dict[str, Any], source: str | None) -> MCPServer:
    """Build an ``MCPServer`` from a ``type: mcp.http`` YAML mapping.

    ``url`` (required) is the remote server endpoint; ``headers`` an optional
    mapping sent on every request (secrets like ``Authorization`` — never
    logged, story 32). The handle name is ``mcp:`` + ``name`` (or the file
    stem). Does NOT connect — deferred to Agent startup. Reuses
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
        keepalive=_parse_keepalive(spec, source),
    )
