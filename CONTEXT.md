# cothis

A complete coding agent built on `any-llm`. The agent loop is a
ReAct cycle; its capability is extended through tools discovered at
startup — from Python callables, YAML declarations, and MCP servers
(issue #1).

## Language

**Tool**:
Any callable object the `Agent` registers by name and dispatches by
calling with keyword arguments. The single dispatch protocol — there is
no per-source branching in `_execute`. Every concrete tool class
(`ToolDef`, `_ShellTool`, `MCPClientTool`) inherits `_HookableTool`, so
hooks run uniformly for all sources (YAML tools' chains are empty no-ops).
A tool may be sync or async: `_execute` awaits the return value only if it
is awaitable (ADR-0004), so MCP's async tools and sync Python/shell tools
share one dispatch path.
_Avoid_: function (too generic), handler, action.

**`@tool` decorator**:
The single Python-tool definition API (replaces the dropped pydantic
base class, issue #1 stories 1–10). Three forms: `@tool`, `@tool("name")`,
`@tool(name=…, description=…)`. Reads a Google-style docstring via `griffe`
(summary → tool description, `Args:` → per-arg descriptions) and
`inspect.signature` (types + required/optional), pre-builds an OpenAI
schema on `__cothis_schema__`. `fs.read`, `fs.write`, `fs.dir` all use it.
_Avoid_: tool factory, tool wrapper, tool class.

**Tool output format**:
How `_execute` serialises a tool's result for the tool message. Controlled
by `COTHIS_TOOL_OUTPUT_FORMAT` (default `json`). Only `dict`/`list` results
are formatted; `str` results bypass (text is text). Formats: `json`
(model-native), `csv` / `tsv` (tabular; nested dicts flattened with dotted
paths; bare-list-of-scalars falls back to json), `yaml` (native nesting).
_Avoid_: tool response, tool payload.

**Hook chain**:
An ordered list of callbacks registered on one lifecycle stage of a tool
(`pre_load` / `after_load` / `pre_execute` / `after_execute` / `on_error`).
Callbacks run in registration order. For data-carrying stages
(`pre_execute`, `after_execute`) the chain is a **pipeline**: each callback's
output feeds the next. For predicate stages (`pre_load`) the chain is
**short-circuit AND**: any `False` return skips the tool. An exception in
any callback **short-circuits** the chain (remaining callbacks don't run);
the `on_error` stage is the escape hatch for audit/logging when this
happens.
_Avoid_: plugin, filter, interceptor (overloaded).

**Tool lifecycle**:
The five stages a `ToolDef` passes through, from discovery to dispatch.
Each stage has a hook chain (see Hook chain). The two load stages fire
**after cross-layer shadow resolution** (see Layer) — on the winning tool
only. A shadowed tool's load hooks never fire (it is dropped before the
load phase runs); there is no fallback to the lower layer if the winner's
`pre_load` returns `False` (the slot goes empty).

Each stage is **observable**: every load/dispatch decision (shadow
override, `pre_load=False`, hook exception, gating miss, `on_error`
observer failure) is logged at `WARNING`; HTTP/transport noise and
per-call I/O stay at `DEBUG` (visible under `-v`). No startup decision
is silent.

| Stage | When | Input | Return | Chain semantics | Exception → |
|---|---|---|---|---|---|
| `pre_load` | after cross-layer merge, on the winner | none | `False` = skip (no fallback) | short-circuit AND | skip tool |
| `after_load` | same point as `pre_load`, after it passes | none | unused (side effect) | all run, no short-circuit | skip tool |
| `pre_execute` | `_execute`, before tool body | `args: dict` | `dict` (modified) | pipeline (A's output → B's input) | short-circuit → on_error → error to LLM |
| `after_execute` | `_execute`, after tool body, before formatting | `result, args` | `result` (modified) | pipeline | use original result |
| `on_error` | any prior stage raised | `exc, phase, args, result` (missing = None) | `None` (side effect only — cannot recover) | short-circuit on its own exception | swallowed to `logger.debug` |

`on_error` is pure side-effect: it observes failures (audit, telemetry,
alerts) but **cannot recover** them. Its own exceptions are swallowed
(chain terminates, `logger.debug` records, main flow's original error
proceeds). `phase` is one of `"pre_load"` / `"after_load"` /
`"pre_execute"` / `"tool"` / `"after_execute"`, naming which stage raised.
_Avoid_: pipeline stage, middleware (both too generic).

**Tool source**:
A path that yields `Tool` objects. **Currently implemented**: Python
(`@tool`-decorated functions), YAML (declarative shell template), and MCP
(external tool server, `type: mcp.stdio` — issue #16 — or `type: mcp.http`
— issue #17). **Planned, not yet implemented** (issue #1): dynamic
discovery of user-authored Python files. The pydantic schema base class
(stories 1–10) was **dropped** — `@tool` is the single Python-tool
definition API. Tool source is the **format** axis only (Python / YAML /
MCP) — it is **never** a precedence axis (see Layer).
_Avoid_: tool type (collides with `YAMLTool`), tool kind, backend,
layer (different axis).

**MCP server**:
An external tool server declared by a `type: mcp.stdio` or `type: mcp.http`
YAML file — the `MCPServer` handle. Not a `Tool` itself but a *producer*
of tools: one server declaration expands into many tools, discovered at
runtime via the MCP `tools/list` call. Its `__name__` is a diagnostic label
(`mcp:` + `name:` or the file stem), prefixed so it can never collide with a
dispatchable tool's name in the discovery registry. The stdio and http
variants differ only in their **transport** (SDK `StdioServerParameters` vs
`StreamableHttpParameters`); everything downstream (session, discovery,
dispatch, normalisation) is shared. The session it opens is
**persistent** — connected once at Agent startup via a
`ClientSessionGroup`, held across every dispatch, closed at teardown
(see ADR-0005).
_Avoid_: MCP tool (that's the produced `MCPClientTool`), MCP client,
plugin.

**Transport**:
The wire an `MCPServer` speaks over — a stdio subprocess (`type: mcp.stdio`,
SDK `StdioServerParameters`) or an HTTP connection (`type: mcp.http`,
SDK `StreamableHttpParameters`). It is the **only** thing that differs
between MCP kinds, so it is the only injected piece: each builder supplies
the matching `params` object, and the `ClientSessionGroup` consumes it
uniformly. A secret-free `diagnostic` string (url scrubbed of
userinfo/query, or command — never `env`/`headers`)
travels alongside for failure logs.
_Avoid_: protocol (that's MCP itself), connection (too vague), channel.

**MCP tool**:
A single remote tool produced by an MCP server — the `MCPClientTool`. Its
`__name__` is **prefixed** with its server's self-reported name (the SDK's
`component_name_hook` assigns `{server_name}.{remote}`): a server reporting
`test-server` with a remote `query-docs` registers as
`test-server.query-docs`. Dispatch is async: `__call__` awaits
`group.call_tool` on the shared `ClientSessionGroup` and normalises the
result to a string (text blocks joined; empty → `"(no output)"`; errors →
`"Error: "` prefix). See ADR-0006 for the prefix scheme's collision
properties.
_Avoid_: MCP server (that's the `MCPServer` producer), remote function,
namespace (implies a hierarchy cothis doesn't have).

**YAMLTool**:
A tool produced from a YAML declaration under `.agents/tools/`. The
YAML-source `Tool` — carries `__name__`, `__doc__`, `__signature__`, and
a pre-built `__cothis_schema__` (so per-arg descriptions reach the LLM
without going through `any-llm`'s lossy `callable_to_tool`). `_compile`
produces a `CommandBlock` from the YAML; `load_yaml_tools` gates the
executable and wraps it in a `_ShellTool` instance.
_Avoid_: shell tool (overloaded with execution mode), YAML function.

**CommandBlock**:
The validated, platform-selected form of a YAMLTool — what `_compile`
produces and both `load_yaml_tools` (gate + wrap) and `preview` (render)
consume. Carries `name`, `description`, `command`, `shell`, `arg_specs`;
exposes `gate_target` (the name Gating resolves) and `render(**kwargs)`
(Placeholder substitution via `str.format_map`). Does NOT carry a resolved
executable path — Gating is `load_yaml_tools`'s concern so preview can
render any branch regardless of host PATH.

Distinct from `_CommandBlock`, the transient per-level parse triple
(command/shell/args, no identity) that `_parse_command_block` produces and
`_select_platform` consumes before `_compile` promotes the selected triple
to a full `CommandBlock`.
_Avoid_: compiled tool (collides with Tool), resolved tool, tool spec
(collides with the raw YAML).

**Execution mode**:
How a YAMLTool's `command:` runs, determined by its YAML type. **argv
mode** — `command:` is a list, passed to `execve` with `shell=False`;
each element is one argv item, spaces/special chars are safe by default.
**shell mode** — `command:` is a string, passed to a declared `shell:`
interpreter with `shell=True`; supports pipes / `&&` / redirection. The
`shell:` field is required for shell mode and names the gated interpreter.
_Avoid_: command type (collides with arg type), run style.

**Platform**:
One of `linux`, `macos`, `unix` (= linux+macOS), `windows` — the keys of
the `platforms:` map in a YAML tool. The current platform is detected
from `sys.platform` at load time; the matching platform entry (exact
key, then `unix` fallback for linux/macos) overrides the top-level
`command:` / `shell:` / `args:`. A platform entry may omit `command:` to
inherit the top-level default.
_Avoid_: OS (too generic), target, environment.

**Gating**:
The load-time check that the executable a tool needs (argv[0] in argv
mode, the `shell:` interpreter in shell mode) is on PATH. If not found,
the tool is not registered — the model never sees a tool it cannot
dispatch on this host. The skip is observable: a `WARNING` is logged
naming the tool and the missing executable (every load/dispatch decision
is visible by default — see Tool lifecycle). Resolution is via
`shutil.which`. **MCP stdio servers** get the same WARNING but are *not*
skipped — they may still launch (full path, runtime PATH) and the
connect-failure path degrades gracefully.
_Avoid_: filter, guard, condition.

**Discovery path**:
A directory scanned for YAML tool declarations at startup. **Project
tools** live under `.agents/tools/` (relative to cwd); **user tools** live
under `$COTHIS_HOME/tools/` (default `~/.cothis/tools/`, overridable via
the `COTHIS_HOME` environment variable). Both are optional; absence is not
an error. Each discovery path is exactly one **Layer** (project-local or
user-global); builtins are a third layer with no directory. Cross-layer
conflicts shadow (project-local > user-global > builtins); same-layer
conflicts raise (see Layer).
_Avoid_: config dir, registry root, tools folder.

**Layer**:
A precedence tier in tool discovery. Three layers, highest precedence
first: **project-local** (`.agents/tools/`), **user-global**
(`$COTHIS_HOME/tools/`, default `~/.cothis/tools/`), **builtins**
(compiled into `TOOLS`). Cross-layer name conflicts are resolved by
**shadowing** (higher precedence wins); same-layer conflicts are an
author error (raise `ValueError`). Layer is independent of tool source:
a project-local YAML tool and a project-local Python tool claiming the
same name are **same-layer** (raise), not cross-source shadow. Format
(Python/YAML/MCP) is never a precedence axis.
_Avoid_: source (collides with Tool source, the format axis), level,
tier (too generic).

**Placeholder**:
A `{arg_name}` token in a `command:` template, substituted with the
arg's value before dispatch. Rendered by `str.format_map` (standard Python
format-string semantics — NOT `string.Template`, whose `$name` delimiter
collides with shell variables). `{{` escapes to a literal `{`, format specs
(`{n:03d}`) and conversions (`{p!r}`) work as in Python. Undeclared
placeholders (name not in `args:`) raise `ValueError` at compile time
(in `_compile`, shared by both `load_yaml_tools` and `preview`).

**Namespace**:
The dotted prefix of a tool's name (`fs` in `fs.read`, `date` in
`date.current`). Tools are organised by namespace and discovered
scattered across the directory tree (`.agents/tools/date/current.yaml`
→ `date.current`), not as a flat list. A namespace is derived from the
tool's declared `name:` field, not its file path.
_Avoid_: group, category, module (overloaded with Python module).

**`if:` expression** *(removed)*:
A previous design used GitHub-Actions-style `if:` expressions on command
branches, with `has_shell()` / `has_exe()` predicates. Removed in the
platform-map refactor — see ADR-0001. `platforms:` map keys (`linux` /
`macos` / `unix` / `windows`) replace per-branch `if:` predicates; argv[0]
/ `shell:` gating replaces `has_shell()` / `has_exe()`.
_Avoid_: (term retired; do not reintroduce without revisiting ADR-0001).
