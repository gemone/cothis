# cothis

A complete coding agent built on `any-llm`. The agent loop is a
ReAct cycle; its capability is extended through tools discovered at
startup — from Python callables, YAML declarations, and MCP servers
(issue #1).

## Language

**@model_metadata `max_tokens` resolution**:
`Agent` passes an output-token cap to every `amessages` call. The cap is
resolved by `cothis.model_metadata.resolve_max_tokens(model, provider,
override)` against a bundled copy of litellm's
`model_prices_and_context_window.json` (`src/cothis/data/`, read via
`importlib.resources`). Match order: explicit `override` >
exact `model` key > `{provider}/{model}` key > fallback `8192`.
Per-entry field order: `max_output_tokens` > legacy `max_tokens` > 8192.
The bundled JSON is refreshed weekly by the `update-model-prices`
workflow (ADR-0007). Override: `--max-tokens` / `COTHIS_MAX_TOKENS`.
_Avoid_: model context window, max context (the cap is on *output* tokens only).

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
`inspect.signature` (types + required/optional), pre-builds an
Anthropic-shape tool schema (`{name, description, input_schema}`) that
`Agent` passes straight to `any_llm.amessages`. `fs.read`, `fs.write`,
`fs.list`, `fs.search` all use it.
_Avoid_: tool factory, tool wrapper, tool class.

**Tool output format**:
How `_execute` serialises a tool's result for the tool message. Controlled
by `COTHIS_TOOL_OUTPUT_FORMAT` (default `json`). Only `dict`/`list` results
are formatted; `str` results bypass (text is text). Formats: `json`
(model-native), `csv` / `tsv` (tabular; nested dicts flattened with dotted
paths; bare-list-of-scalars falls back to json), `yaml` (native nesting).
_Avoid_: tool response, tool payload.

**Tool runtime context**:
The Agent-owned state every `Tool` runs against — `cwd` (the directory
path inputs resolve against), set by the Agent at construction (CLI
`cwd=Path.cwd()` by default; `ask` uses the caller's). `cwd` is
**never** tool-schema-supplied: a model-controlled `cwd` would defeat
the security boundary (the model could escape via `..`). Every `fs.*`
tool resolves user-supplied paths against `cwd` via `_resolve_under`:
absolute paths and `../` escapes that leave `cwd` are rejected with
`"Error: path outside cwd boundary: …"`. In-cwd symlinks are followed
(so a symlink to elsewhere inside `cwd` works); out-of-cwd symlink
targets are rejected by the same `relative_to(cwd)` check on the
resolved path. The context travels via a `ContextVar` (`WORKDIR`),
not a parameter — tools read `WORKDIR.get()` once at entry.
_Avoid_: working directory (too OS-flavoured), project root (a `cwd`
need not be a project), session cwd (that's `Session.cwd`, the
persisted record of the `cwd` the session was started in).

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

**Resource handle**:
An external resource a `Tool` depends on **between** calls — a server
session, a subprocess, a database connection. A parallel concern to Tool
lifecycle (not a sixth hook stage): lifecycle hooks are stateless per-call
interceptors; a handle is stateful and spans many calls. A handle is
declared **independently of** tools (`@resource`), then bound to one or
more tools (`@tool(handle=…)`); a tool **may** bind one or **none**
(`fs.read` holds nothing). The Agent manages handles — not per-tool:
keepalive reclaims a handle idle past a window (default 600s); LRU evicts
under capacity pressure (default 8). A handle shared by several tools has
one `last_used` (any of its tools calling refreshes it) and is reclaimed
once, then transparently re-acquired on the next call to any of its
tools. The re-acquire is inserted in `_execute` after `pre_execute`,
before the tool body, via the same duck-typed no-op pattern as
`run_hooks_safe` (no per-source branching). Tools without a bound handle
keep the existing path untouched. Three lifecycle knobs: `keepalive`
(default 600s), `eager` (acquire on first run, default false), and `pin`
(exempt from reclamation/eviction and the `max_handles` budget until
`aclose`; implies `eager`, default false). MCP sessions are one instance
of this mechanism: each server's startup connection is adopted as the
handle's first acquire, then follows keepalive + LRU. In-flight calls
are protected: the reaper never reclaims a handle whose tool body hasn't
returned.
_Avoid_: connection (too narrow — misses subprocess/file handles),
resource (too generic), pool (implies many; a tool may hold one).

**Tool source**:
A path that yields `Tool` objects. **Implemented**: Python
(`@tool`-decorated functions, auto-scanned from `.py` files in a
discovery path), YAML (declarative shell template), and MCP (external
tool server, `type: mcp.stdio` — issue #16 — or `type: mcp.http` —
issue #17). The pydantic schema base class (stories 1–10) was **dropped**
— `@tool` is the single Python-tool definition API. **Deviation from
PRD story 34**: the PRD asked Python extension files to export a `TOOLS`
list; the shipped loader instead auto-scans for `@tool`-decorated
`ToolDef` instances at module level (no export contract — the author
just decorates). **Deviation from PRD story 38**: the PRD asked Python
extensions to be a "thin wrapper over the shell-tool template" (i.e.
Python files call the `shell()` helper); the shipped design treats
Python extensions as a peer source — `@tool` functions are first-class,
the `shell()` helper is available but optional. See ADR-0005. Tool
source is the **format** axis only (Python / YAML / MCP) — it is
**never** a precedence axis (see Layer).
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
dispatch, normalisation) is shared. The session it opens is **managed by
the resource-handle subsystem** (ADR-0005): connected once at Agent
startup via a `ClientSessionGroup` (that connection is adopted as the
handle's first acquire), then reclaimed when idle past `keepalive`
(default 600s) and re-acquired on the next call. `pin: true` opts a
server back into permanent-session behaviour (ADR-0005).
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
`test-server.query-docs`. When the server reports an empty name, the prefix
falls back to the YAML `name:` label (ADR-0005). Dispatch is async: `__call__`
awaits `group.call_tool` on the shared `ClientSessionGroup` and normalises the
result to a string (text blocks joined; empty → `"(no output)"`; errors →
`"Error: "` prefix). See ADR-0005 for the prefix scheme's collision
properties.
_Avoid_: MCP server (that's the `MCPServer` producer), remote function,
namespace (implies a hierarchy cothis doesn't have).

**YAMLTool**:
A tool produced from a YAML declaration under `.agents/tools/`. The
YAML-source `Tool` — carries `__name__`, `__doc__`, `__signature__`, and
a pre-built Anthropic-shape tool schema (`{name, description, input_schema}`,
so per-arg descriptions reach the LLM without going through `any-llm`'s
lossy `callable_to_tool`). `_compile` produces a `CommandBlock` from the
YAML; `load_yaml_tools` gates the executable and wraps it in a
`_ShellTool` instance.
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
**shell mode** — `command:` is a string, passed to a `shell:`
interpreter with `shell=True`; supports pipes / `&&` / redirection. The
`shell:` field names the gated interpreter; if omitted, cothis auto-selects
the OS default (`sh` on POSIX, `cmd` on Windows — story 16). Shell-mode
arg values are quoted per interpreter (`shlex.quote` for POSIX,
`subprocess.list2cmdline` for `cmd`). On POSIX this fully closes
injection (story 22); on `cmd.exe` it is partial — whitespace-bearing
values are double-quoted, but values like `foo&bar` pass through
unquoted and `%VAR%` expansion is undefended (see the `cothis:` ceiling
on `_shell_quote`). Argv mode (`command:` as a list) is fully safe on
all platforms — prefer it for untrusted input.
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

**System prompt assembly**:
How `Agent` builds the `system` parameter for `amessages`. When
`system` is a `str` (the persona), `_assemble_system` produces
`[persona_block, agents_md_block?, catalog_slot?]`, each block
carrying `cache_control: {type: ephemeral}`. A pre-built
`list[dict]` is passed through unchanged; `None` omits the
parameter entirely. The AGENTS.md block is sourced from
`_load_agents_md` (see Context file); the catalog slot is reserved
for #30 (no-op until skills land). Implemented in #33.
_Avoid_: system prompt (collides with the raw persona text).

**Context file (`AGENTS.md`)**:
A Markdown file injected into the system prompt to give the model
project-specific instructions. Read from three layers in
`COTHIS_AGENTS_ORDER` (default `user-agents,user-cothis,project`):
`~/.agents/` (user-agents), `~/.cothis/` (user-cothis), and `./`
(project). Each layer matches the first filename in
`COTHIS_AGENTS_PATTERN` (comma-separated, default `AGENTS.md`).
User-global layers are toggled off by setting
`COTHIS_AGENTS_USER_GLOBAL` to `0` / `false` / `no` / `off`. Layers
are concatenated as one block, XML-tagged
(`<agents_md type="layer-name">…</agents_md>`); the block is omitted
entirely when no files are found. Empty files (after stripping) are
skipped. Env/CLI only — no config file.
_Avoid_: project file, rules file, instructions (too generic).

**`if:` expression** *(removed)*:
A previous design used GitHub-Actions-style `if:` expressions on command
branches, with `has_shell()` / `has_exe()` predicates. Removed in the
platform-map refactor — see ADR-0001. `platforms:` map keys (`linux` /
`macos` / `unix` / `windows`) replace per-branch `if:` predicates; argv[0]
/ `shell:` gating replaces `has_shell()` / `has_exe()`.
_Avoid_: (term retired; do not reintroduce without revisiting ADR-0001).

**Session**:
The durable unit of conversation history — one `cothis chat` invocation's
accumulated turns, persisted to SQLite so it survives process exit (ADR-0006,
 implemented in #34). A `Session` owns an in-memory `messages: list[dict]`
that is isomorphic to `Agent._messages` (Anthropic block shape) — the
Agent reads from this mirror after load; SQLite is its durable backing,
never read from after the single load-time SELECT. Writes flow through an
in-process queue to a single background consumer thread (or, in test mode,
inline via `flush_sync=True`); `user` is enqueued on input, `tool_result`
per-execution, `assistant` as one atomic multi-block write at MessageStop.
`cothis chat` constructs a Session lazily (session_id always in memory;
lock eager, `sessions` row + title + `.gitignore` lazy on first drain);
`cothis ask` constructs none (ephemeral). One `Session` per process;
a second process reaching a live session is refused via a cross-process
file lock (filelock; `fcntl.flock` on POSIX, `msvcrt`+`kernel32` on Windows)
on `<cache_dir>/<id>.lock` — decoupled from the db location because locks
are flock carriers, not durable state.
**Db resolution (three modes)**: `COTHIS_SESSIONS_TYPE=project` →
`<cwd>/.agents/sessions/session.db` (split layout, per-project);
`COTHIS_SESSIONS_DIR=<path>` → `<path>/session.db` (split layout, custom
location); neither set → `$COTHIS_HOME/agents.db` (default single-file,
unified entry for all sessions and eventually config/audit tables).
Locks live under `$XDG_CACHE_HOME/cothis/` (default `~/.cache/cothis/`).
_Avoid_: conversation (too generic), history (the message list, not the
container), dialogue.

**Block**:
A single content element of an Anthropic message — exactly the Anthropic
content-block dict shape (`text` / `thinking` / `tool_use` /
`tool_result` / `image`). cothis does **not** define a separate Python
type for blocks: the in-memory representation is the Anthropic dict, and
SQLite stores one row per block with per-type columns (`content` /
`signature` / `tool_id` / `tool_name` / `tool_input` / `tool_use_id` /
`tool_output` / `image_source`) so inject/query/prune/compress are native
SQL. A *Message* is `{role, content: list[Block]}`; Anthropic requires
strict user/assistant alternation, so consecutive same-role appends
(per-execution `tool_result`) merge into one Message. Atomic multi-block
writes (the assistant at MessageStop) share one SQLite transaction;
reload drops any trailing partial turn that would leave an orphan
`tool_use` (no matching `tool_result`).
_Avoid_: chunk, fragment, entry (all too generic); node (collides with
the fork tree's session node, #35).

**Branch (session fork)**:
A session created by forking another at a chosen point — git-branch
semantics, no merge. Each session carries `parent_id` and `parent_seq`
on its `sessions` row (NULL on roots). The fork tree is stored as flat
rows; an in-memory `SessionGraph` (stdlib `dict` + functions: `roots`,
`ancestors`, `subtree`, `children_of`, `is_leaf`) layers tree operations
on top — measured 3-10x faster than a recursive CTE on a 10k-session
fixture. Forked sessions number `seq`/`msg_idx`/`block_idx` from 0
(independent numbering); `Session.load` walks the ancestor chain
(root → parent), loads each ancestor's blocks through that link's
`parent_seq` cap (inclusive), and prepends them to the fork's own
messages so the Agent reads one flat conversation. Forks do NOT see the
parent's post-fork blocks. `cothis delete` is leaf-only: a node with
living children is refused with `SessionHasChildrenError` (no orphans).
_Avoid_: thread (collides with OS thread); copy (too generic); clone.

**Hot/cold archive**:
The two-tier session-storage layout introduced by #36 (ADR-0013). The
*hot* DB is the one `Storage` opens at `~/.cothis/agents.db` (or the
project split); sessions idle past a threshold (default 90 days) move
to *cold* — monthly SQLite files under `<db_path parent>/archive/
YYYY-MM.db`. Each move is one atomic cross-DB transaction
(`ATTACH 'archive/YYYY-MM.db' AS arch; INSERT INTO arch.* SELECT * FROM
main.*; DELETE FROM main.*; VACUUM; DETACH`). `archive/index.json` maps
`session_id → {archive_db, archived_at}` so cold lookup doesn't scan
every monthly file. `run_archival_pass` runs once per 24h per process,
wired into `Session.new` / `Session.load` startup (ADR-0011). Cold
sessions are read in place via a separate sqlite3 connection on
`Session.load`; the first new write promotes them back to hot with
`updated_at = now` and drops the index entry (so a freshly-touched
session isn't immediately re-archived). `cothis delete` spans both DBs
— leaf-only check applies across hot and cold (ADR-0012). The CLI
exposes the layer: `cothis archive` (run the pass), `cothis archive
<id>` (archive one), `cothis archive restore <id>` (promote-back),
`cothis archive compress <file>` (gzip a cold DB for transport).
_Avoid_: backup (implies offline copies); tier (too generic); freezer.

**Skill**:
An on-disk capability package the agent can activate on demand (#30,
ADR-0014). One directory under a skills layer (`.agents/skills/`,
`$COTHIS_HOME/skills/`, or `~/.agents/skills/`) containing a
`SKILL.md` with YAML frontmatter (`name`, `description`, optional
`deactivation`) plus a Markdown body, plus optional resource files
referenced by the body. Cross-layer name conflicts resolve by
shadowing (higher-precedence layer wins; a WARNING names both).
_Avoid_: plugin (implies a runtime extension API); module (collides
with Python module); package (too generic).

**Progressive disclosure**:
The two-stage loading model for skills (#30, ADR-0014 §1). The agent
is told *about* every discovered skill up front via the
`<available_skills>` catalog block in the system prompt (name +
one-line description), and loads the full body of a specific skill
only when `load_skill(name)` is invoked. Keeps the system prompt
bounded regardless of how many skills are installed.
_Avoid_: lazy loading (too generic); on-demand fetch.

**Catalog (`<available_skills>`)**:
The system-prompt block that lists every discovered skill as
`- name: description` (#68, ADR-0014 §1). Rendered by
`cothis.skills.format_catalog` as a pure function of the discovered
list; `None` when the list is empty (the block is omitted entirely).
Appended to the system prompt with `cache_control: {type: ephemeral}`
so the catalog is cache-constant per agent run.
_Avoid_: skill list; directory (collides with the on-disk directory).

**Activation**:
The runtime state where a skill's body has been injected into the
tool-result content (via `load_skill`) and its name added to
`Session.active_skills` (#158, ADR-0014 §1). The active set is
runtime-only (not persisted); `Session.load` rebuilds it by replaying
the `load_skill` / `deactivate_skill` tool_use sequence per skill
(#71, ADR-0014 §4). While any skill is active, every turn's latest
user-typed message carries an `<active_skills>` footer naming them
(#72).
_Avoid_: enabled (too generic); loaded (collides with file loading).

**Deactivation (Delete strategy)**:
The mechanism that retires an active skill by archiving its tagged
blocks (#167-#170, ADR-0014 §4). Four-part: Half A marks future
writes for the skill `state='archived'` at enqueue time; Half B
queue-updates historical and in-flight rows; an in-memory walk stamps
the current `messages` mirror; the projection layer
(`_request_messages`) filters blocks with `_cothis_state='archived'`
so the model never sees them on later turns. The skill stays on disk
and in the catalog; only its tagged blocks are hidden. Re-activation
via a fresh `load_skill` produces a new visible epoch — the archived
epoch stays archived. A skill declaring `deactivation: summarize`
falls back to Delete with a logged WARNING (Summarize strategy
deferred).
_Avoid_: removal (implies deletion); unloading (collides with file
unloading); suspension (too generic).

**Session handler (`.session`)**:
The tool-protocol extension that lets a tool mutate session state on
execution (#157, ADR-0014 §3). Declared on `ToolDef` via two flags:
`inject_session=True` causes `Agent._execute_tool` to pass the live
`Session` as a `_session` kwarg (also stripped from the LLM-facing
schema); `skill_marker=True` opts the tool into persist-time
`_cothis_skill` tagging on its `tool_use` / `tool_result` blocks.
Handlers decide via session state (catalog membership, set
membership), never by parsing the result text — the latter would
allow a malicious skill body to forge state mutations.
`load_skill` and `deactivate_skill` both declare both flags.
_Avoid_: tool hook (collides with the per-tool lifecycle hooks);
callback (too generic).
