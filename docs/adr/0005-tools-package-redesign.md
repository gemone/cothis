# Tools package redesign

The tools module was standardised in one PR: the `tools.py` monolith
(~2200 LOC doing five jobs) was split into a `tools/` package, MCP was
rebuilt on the SDK's `ClientSessionGroup`, MCP tool names gained a
server prefix, and a resource-handle subsystem gave every tool a
managed external-resource lifecycle. These four decisions are
interdependent — the package split shaped where the MCP and handle code
live; the handle subsystem redefines MCP's session lifecycle; the
prefix rule depends on the deferred-connect decision — so they're
recorded together rather than as four ADRs that cross-reference and
supersede each other.

## 1. Package split

The package layout is `core.py` / `yaml.py` / `mcp.py` / `builtins.py` /
`format.py`, named by the concern each module owns (not the PRD's
`schema.py` / `shell.py` / `python_ext.py` / `registry.py` taxonomy):

- `core.py` — the shared foundation: `Tool` protocol, `_HookableTool`
  (lifecycle hooks), `@tool`/`ToolDef` (Python-tool API), schema
  helpers, shared validators, layer loading, `discover_tools` (layer
  merge + shadow + load hooks), and the `ResourceHandle` /
  `HandleManager` subsystem (§3).
- `yaml.py` — the YAML shell-tool pipeline (`load_yaml_tools` →
  `CommandBlock` → `_compile` → `_ShellTool` → `preview`).
- `mcp.py` — `MCPServer`, `MCPClientTool`, built on the SDK's
  `ClientSessionGroup`.
- `builtins.py` — `fs.read` / `fs.dir` / `fs.write` + the `TOOLS`
  registry (the builtin layer).
- `format.py` — `format_tool_output` (json/csv/tsv/yaml serialisation).

A new submodule is justified when a concern is clearly distinct AND its
extraction leaves the source file more focused — not to mirror an
external taxonomy. The test for adding one is whether the file it came
from becomes easier to reason about.

**Python extensions are a peer source, not a `shell()` wrapper.** The
PRD asked Python extension files to be "a thin wrapper over the
shell-tool template." The shipped design treats `@tool`-decorated Python
functions as first-class tools (the same API the built-in `fs.*` use);
`shell()` is available for Python tools that need shell glue, but
optional. Forcing every Python extension through `shell()` would give
up the schema-fidelity win (`Annotated[T, Field(...)]` constraints,
rich per-arg descriptions) that `@tool` provides — the core motivation
of the PRD itself. The loader auto-scans each imported module for
module-level `@tool`-decorated `ToolDef` instances; no `TOOLS` export
contract.

### Considered

- **The PRD's module names.** `schema.py` implies a pydantic base class
  (dropped); `python_ext.py` / `registry.py` would each be small files
  doing one job `core.py` already absorbs. Rejected.
- **Keep `tools.py` as one file.** At ~2200 LOC it was too large to
  reason about. Rejected.

### Consequences

- `from cothis.tools import TOOLS` is not supported (the legacy
  back-compat import). The public aggregator is `discover_tools`;
  `TOOLS` stays internal in `builtins.py` as a layer input.
- Submodule import paths (`cothis.tools.core`, `.yaml`, `.mcp`,
  `.builtins`, `.format`) are part of the test surface — tests import
  from submodules directly for white-box checks.

## 2. MCP session lifecycle: deferred connect, group-held sessions

An MCP server is declared via `type: mcp.stdio` / `type: mcp.http` YAML.
It is a *producer* of tools, not a tool itself: one declaration expands
into many tools discovered at runtime via `tools/list`, dispatched via
`tools/call`. The MCP Python SDK is async-only.

The tension: discovery (`load_tools_from_layer`) is **synchronous** and
runs before the Agent exists, but a session must be **connected
asynchronously** (needs a running event loop) and persist across many
`run` / `run_stream` calls in one `chat` session.

### Decision

**Discovery stores transport params; the Agent connects lazily on first
`run` into a single `ClientSessionGroup` it holds, and tears the group
down at `aclose`.**

- **Discovery** (`_build_mcp_stdio_server` / `_build_mcp_http_server`):
  parse the YAML, build the SDK's transport params
  (`StdioServerParameters` / `StreamableHttpParameters`) and a
  secret-free `diagnostic` string, return one `MCPServer` handle. No
  connection — the loader is sync and has no event loop to own a
  persistent session. Transport is the only thing that differs between
  the two kinds; everything downstream is shared.
- **Resolution** (`Agent._ensure_mcp`, first `run`): create one
  `ClientSessionGroup(component_name_hook=…)`, enter it (`__aenter__`),
  then `connect_to_server` each `MCPServer`'s params. Each new tool is
  wrapped in an `MCPClientTool` and registered in `_tool_map`. Runs at
  most once (`_mcp_started` guard).
- **Dispatch** (`MCPClientTool.__call__`): `await group.call_tool(...)`
  on the shared group.
- **Teardown** (`Agent.aclose`): the group's sessions are released
  first (see §3 — handles disconnect their own sessions), then
  `group.__aexit__(None, None, None)` tears down whatever remains.

**Why `ClientSessionGroup`?** The SDK class (added in mcp 1.x) solves
exactly the session-spanning problem: an internal `AsyncExitStack` holds
every server's transport+session open across calls, aggregates tools
from all servers, prefixes tool names via `component_name_hook`, and
tears everything down on `__aexit__`. It subsumes ~200 LOC of
hand-rolled per-server context management that predated it.

### Considered

- **Hand-rolled per-server context management.** Each `MCPServer` held
  its own `_cm` / `_session`. Rejected: `ClientSessionGroup` provides
  the same capability plus multi-server aggregation and name prefixing,
  maintained upstream.
- **Connect during discovery** (sync, via a throwaway `asyncio.run`).
  Rejected: spawns the subprocess twice, and `asyncio.run` cannot
  produce a session that survives into the Agent's event loop.
- **Reconnect per call.** Rejected: a stdio server pays subprocess
  spawn + MCP handshake latency on every call, and stateful servers lose
  their state between calls.

### Consequences

- **`_ensure_mcp` and `aclose` run in the same task.** The MCP SDK uses
  anyio task groups / cancel scopes internally; exiting a cancel scope
  in a different task than entered raises. The Agent awaits both from
  its own event loop.
- **A failed `connect_to_server` is non-fatal.** Bad command, connection
  refused, or protocol error logs at `WARNING` (naming the server + its
  `diagnostic` — never `env`/`headers` secrets), unwraps
  `ExceptionGroup`s to the real cause, and returns no tools; the rest
  of the agent's tools still load.
- **HTTP session resumption is delegated to the SDK.** cothis holds no
  session id, no reconnect policy, no transport state. If the SDK later
  exposes resumption control, cothis will thread it through
  `MCPServer.connect_into`.

## 3. Resource handles: managed tool resources (keepalive + LRU)

cothis's MCP tools were eager and permanent — every declared server
connected at Agent startup and stayed open until `aclose`, regardless of
whether the LLM ever called its tools. There was also no general way for
a third-party Python tool to declare "I hold an external resource (DB
connection, HTTP session) that should be kept warm or reclaimed when
idle."

### Decision

**A `ResourceHandle` — an external resource declared independently
(`@resource`), bound to one or more tools (`@tool(handle=…)`), managed
by the Agent with keepalive + LRU. MCP sessions are one instance of
this mechanism: each server's startup connection (§2) is adopted as the
handle's first acquire, then follows keepalive + LRU.**

- **Schema stays fresh, connections come and go.** Each MCP server
  still connects once at startup to `list_tools` (schema is observed,
  not disk-cached — see "rejected: disk-cached schema"). That startup
  connection is **adopted** as the handle's first acquire (not wasted
  and reconnected): the session is seeded onto a per-server
  `MCPSessionHandle` instance and registered live in the pool. After
  that, the handle follows the standard keepalive + LRU lifecycle.
- **Handle is a parallel subsystem to Tool lifecycle, not a sixth
  hook.** Lifecycle hooks are stateless per-call interceptors; a handle
  is stateful and spans calls. Both are invoked in `_execute` (handle
  ensure after `pre_execute`, before the tool body), but they are
  distinct concepts.
- **No per-source branching.** `_execute` calls duck-typed
  `ensure_handle_ready(tool)`; tools without a bound handle no-op.
- **`__call__` is self-healing.** A handle reclaimed while idle is
  transparently re-acquired on the next call; the LLM never sees a
  "tool temporarily unavailable."
- **Management unit is the handle instance, not the tool.** A handle
  shared by several tools has one `last_used` and is reclaimed /
  re-acquired as one unit. For MCP, each server is one dynamically
  generated `MCPSessionHandle` subclass (so each server is one pool
  entry keyed by its own class).
- **In-flight protection.** A handle is never reclaimed while a tool
  call is in progress. `_execute` brackets the body with
  `mark_inflight` / `handle_call_done`; the reaper and LRU eviction
  skip handles with a positive in-flight count. `call_done` also
  refreshes `last_used` at call *end*, so a long call gets a full
  keepalive window after it finishes.
- **Three lifecycle knobs on `@resource`:**
  - `keepalive` (default 600s) — idle seconds before reclamation.
  - `eager` (default false) — acquire on the Agent's first run via
    `HandleManager.start_eager()`.
  - `pin` (default false) — keep the resource alive until `aclose`:
    exempt from keepalive reclamation, LRU eviction, and the
    `max_handles` budget. Implies `eager`.
- **MCP YAML exposes `keepalive` and `pin`.** `eager` is not exposed
  (MCP always adopts at startup). `pin: true` opts a server back into
  the permanent-session behaviour (the pre-handle status quo).

### Considered

- **Disk-cached tool schema.** Rejected: without a live connection
  there is no way to detect server-side drift (new tool, changed
  signature). cothis has no proxy tool to fall back on, so stale schema
  would cause silent call failures. Schema is observed fresh each
  startup.
- **Handle as a sixth lifecycle hook stage.** Rejected: lifecycle hooks
  are stateless per-call; a handle is stateful across calls. Folding
  them into one table gives one row with different semantics.
- **Handle injected as a tool function parameter.** Rejected: it would
  pollute the LLM schema (`_build_schema` reads the signature). The
  handle is a tool-instance attribute (`query.handle`) accessed by the
  function body.

### Consequences

- **MCP tools change from eager/permanent to managed.** Startup
  connects each server once and adopts that connection; sessions are
  reclaimed when idle past `keepalive`, re-acquired on the next call
  via `connect_to_server`, and released via `disconnect_from_server`.
  `pin: true` opts out.
- **In-flight calls are protected from the reaper.** A handle with a
  positive in-flight count is never reclaimed or evicted. `call_done`
  refreshes `last_used` at call end.
- **A background reclamation pass runs between turns** (not inside
  `_execute`): handles past keepalive are released; under LRU pressure
  the coldest evictable handle is evicted. Pinned and in-flight handles
  are skipped.
- **Pinned handles are exempt from the `max_handles` budget.** They
  neither count against the limit nor get evicted, so a pool full of
  pinned handles still admits a new non-pinned one.
- **`aclose` releases handles before exiting the group.** MCP-session
  handles disconnect their own sessions, then the group context closes.
- **Third-party tools gain a managed-resource path** without writing
  their own connection lifecycle — `@resource` + bind.

## 4. MCP tool-name prefix

MCP servers are producers: one declaration expands into many remote
tools whose names come from the server's `tools/list` response — names
cothis doesn't author and can't constrain. A remote `search` and a
YAML-declared `search` would share one `__name__`, and the MCP tool
(registered later, at runtime in `_ensure_mcp`) would silently overwrite
the YAML tool in `_tool_map`.

### Decision

**Each MCP tool's `__name__` is prefixed with its producing server's
name, assigned by the SDK's `component_name_hook`.** A server that
self-reports as `test-server` returning a remote `add` registers as
`test-server.add`. The prefixed name is identical in `__name__`,
`_tool_map` key, and schema `function.name` — one name, one source of
truth.

**Prefix uniqueness follows from server-name uniqueness.** The prefix
comes from the server's self-reported `Implementation.name` (its
`FastMCP(name=…)` argument), NOT from cothis's YAML `name:` field. Two
servers reporting the same implementation name collide at the SDK level;
cothis's own server-handle dedup (`mcp:{label}`) is a load-time guard.
MCP-vs-non-MCP collisions are made highly unlikely, and any residual
clash is caught at registration with an ERROR log + first-write-wins.

**Fallback to the YAML label when the server reports an empty name.**
The hook falls back to the cothis-side YAML `name:` label (stripped of
its `mcp:` prefix) so the tool still gets a meaningful prefix. The hook
fires inside `connect_to_server` with nothing identifying *which* cothis
server is connecting, so the label travels through a mutable cell
updated immediately before each connect — by the startup loop in
`_ensure_mcp`, and by `MCPSessionHandle.acquire` on every re-acquire —
keeping each empty-name server's prefix stable across the keepalive/LRU
session lifecycle. Connects never run concurrently (startup,
`start_eager`, and `ensure_acquired` all await connects inline), so the
temporal handoff is race-free.

**This is a prefix, not a namespace system.** Builtins (`fs.read`) and
user tools keep their existing names. `_tool_map` stays a flat dict; no
hierarchy, no namespace registry.

### Considered

- **Runtime duplicate-detection on MCP tool names.** Reconstruct layer
  identity at `_ensure_mcp` time, run same-layer raise / cross-layer
  shadow. Rejected: layer identity has collapsed by runtime (§2
  deliberately defers connection past discovery), and runtime raise has
  poor UX (a conflict surfaces only on the first `run`). The prefix
  removes the problem by construction.
- **Two-layer `dict[ns, dict[tool, Tool]]` or trie.** Rejected (YAGNI):
  the only subtree need (`aclose`) is already one line; no current name
  exceeds two segments.

### Consequences

- **The prefix is visible to the model.** The LLM sees and calls
  `context7.query-docs`; cothis routes it back to the server with the
  bare `query-docs`. A deliberate, small token cost for collision safety.
- **One residual case is handled by first-write-wins.** If a single
  server's `tools/list` returns two tools of the same remote name, the
  first is registered, the second skipped with an ERROR log.
