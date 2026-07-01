# Async tool dispatch: `isawaitable` bridge

Tool dispatch in `_execute` is `async def`. All tools are called as
`result = tool(**args); if inspect.isawaitable(result): result = await result`.
Sync tools (ToolDef, `_ShellTool`, bare callables) return non-coroutine
values — the `isawaitable` check skips the await, behavior unchanged. Async
tools (MCP `_MCPClientTool`) return coroutines — the await activates.

## Decision

**Dispatch path goes full async; user tool functions stay sync-compatible.**

`_execute` and `_execute_tool_calls` are `async def`. `run` / `run_stream`
`await` the dispatch call site. Inside `_execute`, after calling the tool,
the `isawaitable` bridge decides whether to await.

User-authored `@tool` functions remain synchronous `def`. They return plain
values (strings, dicts, lists), not coroutines. The bridge sees a
non-awaitable and skips the await — zero behavior change for existing tools.

MCP tools (issue #16) will be `async def` returning coroutines. The bridge
awaits them. This is the only async tool source today; future async tool
sources follow the same pattern automatically.

## Considered alternatives

- **Keep dispatch sync, run MCP tools via `asyncio.run` per call.** Rejected:
  `_execute` runs inside an existing event loop (`run` is `async def`,
  called via `asyncio.run` in `cli.py`). Calling `asyncio.run` inside a
  running loop raises `RuntimeError`. Bridging via `run_in_executor` or
  thread-pool adds complexity for no benefit.

- **Force all tools to be `async def`.** Rejected: every builtin
  (`fs.read`, `fs.write`, `_ShellTool`) and every user `@tool` function
  would need rewriting. The burden on users (must understand async to write
  a tool) outweighs the purity gain. The `isawaitable` bridge gives the same
  dispatch uniformity without the migration cost.

- **Per-source branching: `if isinstance(tool, _MCPClientTool): await ...`.**
  Rejected: contradicts CONTEXT.md's "no per-source branching in `_execute`"
  principle. The `isawaitable` check is structural (does the return value
  need awaiting?), not source-based (what type of tool is this?).

## Consequences

- Every tool call passes through one `inspect.isawaitable` check. The cost
  is one C-level function call per dispatch — negligible.
- Sync tools are unaffected: their return values are `str` / `dict` / `list`
  / `int`, none of which are awaitable.
- `_execute` and `_execute_tool_calls` are now `async def`; all callers
  (`run`, `run_stream`) `await` them. Tests calling `_execute` directly must
  be `async def` + `@pytest.mark.asyncio`.
- Adding a new async tool source (beyond MCP) requires no `_execute` change
  — just return a coroutine from the tool's `__call__`.
