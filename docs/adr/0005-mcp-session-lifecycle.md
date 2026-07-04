# MCP session lifecycle: lazy connect, Agent-held `ClientSessionGroup`

An MCP server is declared via `type: mcp.stdio` or `type: mcp.http` YAML.
It is **not** a tool but a *producer* of tools: one server declaration
expands into many tools, discovered at runtime via the MCP `tools/list` call
and dispatched via `tools/call`. The MCP Python SDK is async-only.

The tension: discovery (`load_tools_from_layer`) is **synchronous** and runs
before the Agent exists, but a session must be **connected asynchronously**
(needs a running event loop) and **persist** across many `run` / `run_stream`
calls in one `chat` session.

## Decision

**Discovery stores server params; the Agent connects lazily via a
`ClientSessionGroup` and holds it for its lifetime; teardown exits the
group's context.**

- **Discovery** (`_build_mcp_stdio_server` / `_build_mcp_http_server`):
  parse the YAML, build the SDK's transport params
  (`StdioServerParameters` / `StreamableHttpParameters`) and a secret-free
  `diagnostic` string, return one `MCPServer` handle. No connection — the
  loader is sync and has no event loop to own a persistent session. The
  **transport** is the only thing that differs between the two kinds;
  everything downstream is shared.
- **Resolution** (`Agent._ensure_mcp`, first `run`): create one
  `ClientSessionGroup(component_name_hook=...)`, enter it
  (`__aenter__`), then `connect_into(group)` each `MCPServer`. The group
  handles transport, session, `tools/list`, and name prefixing. Each new
  tool is wrapped in an `MCPClientTool` and registered in `_tool_map`.
  Runs at most once (`_mcp_started` guard) — the group is persistent.
- **Dispatch** (`MCPClientTool.__call__`): `await group.call_tool(...)`
  on the shared group. No per-call reconnect.
- **Teardown** (`Agent.aclose`): `group.__aexit__(None, None, None)` — one
  call tears down every server's session and transport (subprocess or HTTP
  connection). `ask` calls this after its single `run`; `chat` calls it
  when the session ends.

**Why `ClientSessionGroup`?** The SDK's `ClientSessionGroup` (added in
mcp 1.x) solves exactly the session-spanning problem: it uses an internal
`AsyncExitStack` to hold every server's transport+session open across
calls, aggregates tools from all servers, prefixes tool names via
`component_name_hook`, and tears everything down on `__aexit__`. Before
this class existed (or was discovered), cothis hand-rolled the same
machinery in `MCPServer` (`_default_connect`, manual `_cm` / `_session`
management, per-server `start`/`aclose`). The SDK class subsumes all of
that; `MCPServer` is now a thin config container that hands its params to
the group via `connect_into`.

**The Agent separates `MCPServer` handles from callable tools at
construction.** `model_post_init` splits `self.tools` into `_mcp_servers`
(handles) and `_tool_map` (dispatchable tools). This is startup-time
branching, not per-dispatch branching — `_execute` still treats every
registered tool uniformly (`MCPClientTool` inherits `_HookableTool` like
every other tool), preserving CONTEXT.md's "no per-source branching in
`_execute`".

## Considered alternatives

- **Hand-rolled per-server context management (prior approach).** Each
  `MCPServer` held its own `_cm` / `_session`, entered in `start()` and
  exited in `aclose()`. Rejected in favour of the SDK's
  `ClientSessionGroup`: it provides the same capability (cross-method
  session persistence via `AsyncExitStack`), plus multi-server tool
  aggregation and name prefixing, all maintained upstream. The hand-rolled
  version was ~200 LOC of reimplementation.

- **Connect during discovery** (sync, via a throwaway `asyncio.run`).
  Rejected: it spawns the subprocess twice (once to list tools at
  discovery, again to hold a persistent session at run), and
  `asyncio.run` inside discovery cannot produce a session that survives
  into the Agent's own event loop — the loop that created it closes,
  taking the session with it.

- **Reconnect per call** (open/`initialize`/`tools/call`/close each
  dispatch). Rejected: a stdio server pays subprocess spawn + MCP
  handshake latency on every tool call, and stateful servers (a browser
  session, an open file handle) lose their state between calls. The
  confirmed decision is persistent sessions.

## Consequences

- **Both `_ensure_mcp` and `aclose` must run in the same task.** The MCP
  SDK uses anyio task groups / cancel scopes internally; exiting a cancel
  scope in a different task than entered raises. The Agent awaits both
  from its own event loop (never spawning them as separate tasks), so this
  holds.

- **A failed `connect_into` is non-fatal.** Bad command, connection
  refused, or a protocol error logs at `WARNING` (naming the server + its
  `diagnostic` — command/args for stdio, scrubbed url for http; never
  `env`/`headers` — secrets), unwraps `ExceptionGroup`s to the real
  cause, and returns `[]`. The server contributes no tools; the rest of
  the agent's tools still load.

- **MCP tool names are prefixed by `component_name_hook`.** The SDK
  assigns `{server_self_reported_name}.{remote_name}` (e.g.
  `test-server.add`). The prefix comes from the server's `Implementation`
  name (its `FastMCP(name=…)` argument at init), NOT from cothis's YAML
  `name:` field. Two servers reporting the same implementation name
  collide at the SDK level (it raises on duplicate tool keys); the
  Agent's first-write-wins dedup catches duplicates within a single
  server's contribution. The server *handle* itself cannot collide: its
  `__name__` is prefixed `mcp:`, which is not a valid dotted tool name.

- **`aclose` is a full reset, not just a close.** It exits the
  `ClientSessionGroup` (closing every server), drops the resolved
  `MCPClientTool` entries from `_tool_map` (tracked by name in
  `_mcp_tool_names`), and clears the `_mcp_started` guard. Reusing the
  same Agent after `aclose` therefore reconnects with fresh sessions on
  the next `run` instead of dispatching against closed ones.

- **The test seam is `connect_with_session` on the group.** Tests use
  the SDK's in-memory `create_connected_server_and_client_session`
  transport + `group.connect_with_session` so the adapter code under
  test is exactly production — only the transport differs. No subprocess,
  no network, deterministic.
