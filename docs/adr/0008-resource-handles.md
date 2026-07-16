# Resource handles: managed tool resources (keepalive + LRU)

cothis's MCP tools were eager and permanent — every declared server
connected at Agent startup and stayed open until `aclose`, regardless of
whether the LLM ever called its tools. There was also no general way for
a third-party Python tool to declare "I hold an external resource (DB
connection, HTTP session) that should be kept warm or reclaimed when
idle." Studying `pi-mcp-adapter`'s MCP management surfaced both gaps.

## Decision

**Introduce a `Resource handle` — an external resource declared
independently (`@resource`), bound to one or more tools
(`@tool(handle=…)`), managed by the Agent with keepalive + LRU.**

- **Schema stays fresh, connections come and go.** Each MCP server still
  connects once at startup to `list_tools` (schema must be observed, not
  guessed — see "rejected: disk-cached schema"). After listing, the
  connection is released; it is re-acquired on the first `call_tool` and
  reclaimed when idle.
- **Handle is a parallel subsystem to Tool lifecycle, not a sixth hook.**
  Lifecycle hooks (`pre_execute`, etc.) are stateless per-call
  interceptors; a handle is stateful and spans calls. Both are invoked in
  `_execute` (handle ensure after `pre_execute`, before the tool body),
  but they are distinct concepts — conflating them would give the
  lifecycle table one row with different semantics.
- **No per-source branching.** `_execute` calls a duck-typed
  `ensure_handle_ready(tool)`; tools without a bound handle no-op, the
  same pattern as `run_hooks_safe` (CONTEXT.md's "no per-source branching
  in `_execute`" holds).
- **`__call__` is self-healing.** A handle reclaimed while idle is
  transparently re-acquired on the next call; the LLM never sees a
  "tool temporarily unavailable."
- **Management unit is the handle instance, not the tool.** A handle
  shared by several tools has one `last_used` (any calling tool refreshes
  it) and is reclaimed/re-acquired as one unit.
- **Defaults: keepalive 600s, max_handles 8.** Both configurable.

## Considered alternatives

- **Disk-cached tool schema (pi-mcp-adapter route).** Rejected: without a
  live connection there is no way to detect that a server changed its
  tools (new tool, changed signature) while its declaration stayed
  identical. A declaration fingerprint only catches declaration-level
  changes, not server-side drift. cothis has no proxy tool to fall back
  on, so stale schema would cause silent call failures. Schema is
  observed fresh each startup instead.

- **Handle as a sixth lifecycle hook stage.** Rejected: lifecycle hooks
  are stateless per-call; a handle is stateful across calls. Folding them
  into one table gives one row with different semantics and confuses
  readers. Handle is a parallel subsystem.

- **Handle injected as a tool function parameter.** Rejected: it would
  pollute the LLM schema (`_build_schema` reads the signature) — the LLM
  would see a parameter it cannot supply. The handle is a tool-instance
  attribute (`query.handle`) accessed by the function body; the signature
  stays equal to the LLM schema.

- **Handle declared after the tool (attach back).** Rejected on semantic
  grounds: the resource is prior to and independent of the tool — the
  tool depends on the resource, not the reverse. Declare `@resource`
  first, bind via `@tool(handle=…)`.

## Consequences

- **MCP tools change from eager/permanent to lazy/reclaimable.** Startup
  connects each server once to list, then releases. ADR-0005's "one
  `ClientSessionGroup` held for the Agent's lifetime" still holds for
  the group, but individual sessions are now added/removed by the handle
  manager via `connect_to_server` / `disconnect_from_server`.
- **A background reclamation pass runs between turns** (not inside
  `_execute`): handles past keepalive are released; under LRU pressure
  the coldest are evicted. This is the one new async surface in the
  Agent.
- **Third-party tools gain a managed-resource path** without writing
  their own connection lifecycle — `@resource` + bind.
