# MCP tool-name prefix: label-distinguish MCP tools, not duplicate-detect

MCP servers are producers: one `type: mcp.*` declaration expands into many
remote tools at Agent startup. The remote tool names come from the server's
`tools/list` response — names cothis doesn't author and can't constrain. A
remote `search` from an MCP server and a YAML-declared `search` tool would
share one `__name__`, and the MCP tool (registered later, at runtime in
`_ensure_mcp`) would silently overwrite the YAML tool in `_tool_map`.

Issue #18 originally asked for this to be solved by extending the existing
conflict machinery: same-layer duplicate detection and cross-layer shadow,
covering MCP-produced tool names. The tension: those names don't exist at
discovery (the server isn't connected yet — ADR-0005), and layer identity
has already collapsed by the time `_ensure_mcp` runs. Honoring the AC would
mean reconstructing layer info at runtime to decide raise-vs-shadow for
names that only appear there.

## Decision

**Each MCP tool's `__name__` is prefixed with its producing server's
name, assigned by the SDK's `component_name_hook`.** A server that
self-reports as `test-server` returning a remote `add` registers as
`test-server.add`, not `add`. The prefix is assigned by the
`ClientSessionGroup`'s `component_name_hook` callback (which receives
the tool name and the server's `Implementation` info), not by cothis
hand-appending a label. The prefixed name is identical in `__name__`,
`_tool_map` key, and schema `function.name` — one name, one source of
truth.

**Prefix uniqueness follows from server-name uniqueness.** The prefix
comes from the server's self-reported `Implementation.name` (its
`FastMCP(name=…)` argument at init), NOT from cothis's YAML `name:`
field. Two servers reporting the same implementation name collide at
the SDK level (the `ClientSessionGroup` raises on duplicate tool
keys); cothis's own server-handle dedup (`mcp:{label}`) is a
load-time guard on the cothis side. MCP-vs-non-MCP collisions are made
highly unlikely (prefixed name vs. bare/builtin name), and any residual
clash (a user tool named exactly `{server_name}.{remote}`) is caught at
registration with an ERROR log + first-write-wins, not silently
overwritten.

**Fallback to the YAML label when the server reports an empty name.**
The MCP spec requires `Implementation.name` to be non-empty, but a
non-conformant server may send an empty string. The prefix hook falls
back to the cothis-side YAML `name:` label (stripped of its `mcp:`
handle prefix) so the tool still gets a meaningful, unique prefix
rather than a bare remote name that could collide. Servers connect
sequentially inside `_ensure_mcp`, so the hook reads the current
server's label from a mutable cell updated before each `connect_into`.
The YAML label is always non-empty (validated at load time by
`_make_mcp_server`), so the fallback is always a valid prefix.

**This is a prefix, not a namespace system.** Builtins (`fs.read`) and user
tools (a bare `my_tool`) keep their existing names. Only MCP tools carry a
source-distinguishing prefix. cothis does not introduce a hierarchy, a
namespace registry, or subtree operations; `_tool_map` stays a flat dict,
and "all tools of one MCP server" is found by string-prefix filter, not by
data-structure traversal.

**One residual case is handled by first-write-wins.** If a single server's
`tools/list` returns two tools of the same remote name (server bug or
intentional), the first is registered, the second is skipped with an `ERROR`
log naming the server and the duplicate. This mirrors the general principle
that MCP failures degrade rather than crash the run (ADR-0005).

## Considered alternatives

- **Runtime duplicate-detection on MCP tool names (the original #18 AC).**
  Reconstruct layer identity at `_ensure_mcp` time, run same-layer raise /
  cross-layer shadow on the freshly-listed names. Rejected: layer identity
  has collapsed by runtime (ADR-0005 deliberately defers connection past
  discovery), so re-threading it is a real cost for a problem the prefix
  removes by construction. Runtime raise also has poor UX — a conflict
  surfaces only on the first `run`, after the user has already been
  prompted in `chat`.

- **Two-layer `dict[ns, dict[tool, Tool]]` or trie.** Group tools by
  namespace for subtree operations (e.g. drop all of one server's tools in
  `aclose` without `isinstance` checks). Rejected (YAGNI): the only current
  subtree need (`aclose`) is already one line via
  `isinstance(_MCPClientTool)`, a namespace structure would force a decision
  about single-segment user tool names, and no current name exceeds two
  segments. `_tool_map` stays a flat dict.

- **Forbid bare names; require every tool to declare a namespace.** Rejected:
  breaks the existing `@tool def my_tool()` Python-tool API and the
  "no-`type:` YAML keeps working" backward-compat requirement (#18 AC).

## Consequences

- **The prefix is visible to the model.** The LLM sees `context7.query-docs`
  and calls it by that name; cothis routes it back to the server with the
  bare `query-docs`. A deliberate, small token cost for collision safety.

- **The original #18 ACs are satisfied at the server-handle layer, not the
  tool-name layer.** Two MCP servers with colliding tool names but different
  labels never collide; same-label collisions are caught at the handle layer.
