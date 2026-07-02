# MCP session lifecycle: lazy connect, Agent-held, manual context

An MCP server is declared via `type: mcp.stdio` or `type: mcp.http` YAML.
It is **not** a tool but a *producer* of tools: one server declaration
expands into many tools, discovered at runtime via the MCP `tools/list` call
and dispatched via `tools/call`. The MCP Python SDK is async-only, and its
`ClientSession` + transport clients (`stdio_client`,
`streamablehttp_client`) are `async with` context managers that close on
exit.

The tension: discovery (`load_tools_from_layer`) is **synchronous** and runs
before the Agent exists, but a session must be **connected asynchronously**
(needs a running event loop) and **persist** across many `run` / `run_stream`
calls in one `chat` session.

## Decision

**Discovery stores server params; the Agent connects lazily and holds the
session; teardown closes it.**

- **Discovery** (`_build_mcp_stdio_server` / `_build_mcp_http_server`):
  parse the YAML, build an `open_transport` factory (async CM yielding a
  `(read, write)` stream pair) and a secret-free `diagnostic` string, return
  one `_MCPServer` handle. No connection — the loader is sync and has no
  event loop to own a persistent session. The **transport** is the only
  thing that differs between the two kinds; everything downstream
  (`_default_connect` → `ClientSession` → `tools/list`) is shared.
- **Resolution** (`Agent._ensure_mcp`, first `run`): for each `_MCPServer`,
  `start()` the session (connect, `initialize`, `tools/list`), wrap each
  remote tool in an `_MCPClientTool`, and register those in `_tool_map`.
  Runs at most once (`_mcp_started` guard) — the session is persistent.
- **Dispatch** (`_MCPClientTool.__call__`): `await session.call_tool(...)`
  on the shared session held by the `_MCPServer`. No per-call reconnect.
- **Teardown** (`Agent.aclose`): `aclose()` every server, closing its
  session and transport (subprocess or HTTP connection). `ask` calls this
  after its single `run`; `chat` calls it when the session ends.

**The session is held open with manual `__aenter__` / `__aexit__`, not
`async with`.** The connection context is entered in `start()` and exited in
`aclose()` — two different methods. `async with` sugar can only hold a
context open within one lexical block, so the lifecycle is driven by hand:
`start` stores the live context manager on `self._cm`, `aclose` exits it.

**The Agent separates `_MCPServer` handles from callable tools at
construction.** `model_post_init` splits `self.tools` into `_mcp_servers`
(handles) and `_tool_map` (dispatchable tools). This is startup-time
branching, not per-dispatch branching — `_execute` still treats every
registered tool uniformly (`_MCPClientTool` inherits `_HookableTool` like
every other tool), preserving CONTEXT.md's "no per-source branching in
`_execute`".

## Considered alternatives

- **Connect during discovery** (sync, via a throwaway `asyncio.run`).
  Rejected: it spawns the subprocess twice (once to list tools at discovery,
  again to hold a persistent session at run), and `asyncio.run` inside
  discovery cannot produce a session that survives into the Agent's own
  event loop — the loop that created it closes, taking the session with it.

- **Reconnect per call** (open/`initialize`/`tools/call`/close each
  dispatch). Rejected: a stdio server pays subprocess spawn + MCP handshake
  latency on every tool call, and stateful servers (a browser session, an
  open file handle) lose their state between calls. The confirmed decision
  is persistent sessions.

- **`async with` inside `_ensure_mcp`, holding the loop there.** Rejected:
  the session must outlive `_ensure_mcp` and span every subsequent `run`
  call in a `chat`. A context manager entered in `_ensure_mcp` and exited at
  the end of `_ensure_mcp` would close the session before the first tool
  call. The lifecycle genuinely spans methods.

- **`contextlib.AsyncExitStack` on the Agent.** A reasonable variant — but
  the per-server `_MCPServer` already owns exactly one connection context, so
  a single `self._cm` slot with manual enter/exit is simpler than a stack.
  The stack buys nothing when there is one context per server.

## Consequences

- **Both `start` and `aclose` must run in the same task.** The MCP SDK uses
  anyio task groups / cancel scopes internally; exiting a cancel scope in a
  different task than entered raises. The Agent awaits both from its own
  event loop (never spawning them as separate tasks), so this holds. Tests
  await `start` and `aclose` in one coroutine — same task. This is a real
  constraint: a future refactor that moves `aclose` onto a different task
  (e.g. a background cleanup coroutine) would break it.

- **A failed `start` is non-fatal.** Bad command, connection refused, or a
  protocol error logs at `WARNING` (naming the server + its `diagnostic` —
  command/args for stdio, url for http; never `env`/`headers` — secrets),
  unwinds any partial context, and returns `[]`. The server contributes no
  tools; the rest of the agent's tools still load.

- **MCP tool names are prefixed and de-duplicated at registration.** Each
  produced tool's `__name__` is `{label}.{remote_name}` (ADR-0006), so it
  normally can't collide with a builtin or user tool (which use bare names).
  Any residual clash (e.g. a user tool named exactly `{label}.{remote}`) is
  caught at registration in `_ensure_mcp`: first-write-wins keeps the
  already-registered tool, logs the clash at ERROR. The server *handle*
  itself cannot collide: its `__name__` is prefixed `mcp:`, which is not a
  valid dotted tool name, so a server label can neither shadow nor be
  shadowed by a real tool.

- **`aclose` is a full reset, not just a close.** It closes every server,
  drops the resolved `_MCPClientTool` entries from `_tool_map`, and clears
  the `_mcp_started` guard. Reusing the same Agent after `aclose` therefore
  reconnects with fresh sessions on the next `run` instead of dispatching
  against closed ones. `ask` (one run, then discard) and `chat` (close at
  session end) don't exercise reuse, but the reset makes it safe by
  construction rather than by caller discipline.

- **The transport seam (`open_transport`) and the test seam (`connect`)
  both exist for testability.** `open_transport` is injected per kind by the
  builder so the transport choice is isolated; `connect` bypasses
  `open_transport` entirely, letting tests pass the SDK's in-memory
  `create_connected_server_and_client_session` transport so the adapter
  code under test is exactly production — only the transport differs. No
  subprocess, no network, deterministic.

- **The HTTP transport cannot resume a dropped server session.**
  `streamablehttp_client` yields a 3-tuple `(read, write, get_session_id)`;
  the session-id callback is dropped to match stdio's 2-tuple contract (see
  the `cothis:` ceiling comment in `_build_mcp_http_server`). If the HTTP
  connection drops mid-session, cothis has no way to reconnect with the
  server-assigned session id; each `run` opens a fresh transport. Threading
  `get_session_id` through `_default_connect` is the upgrade path.
