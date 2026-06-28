# cothis

A basic coding agent built on `any-llm`. The agent loop is a small
ReAct cycle; its capability is extended through tools discovered at
startup — today from Python callables and YAML declarations, with MCP
and dynamic Python extensions planned (issue #1).

## Language

**Tool**:
Any callable object the `Agent` registers by name and dispatches by
calling with keyword arguments. The single dispatch protocol — there is
no per-source branching in `_execute`.
_Avoid_: function (too generic), handler, action.

**Tool source**:
A path that yields `Tool` objects. **Currently implemented**: Python
(hand-written `def` or class) and YAML (declarative shell template).
**Planned, not yet implemented** (issue #1): MCP (external tool server,
stories 25–32) and Python dynamic extensions (stories 33–38). The
pydantic schema base class (stories 1–10) is also planned as the
universal layer every source compiles down to.
_Avoid_: tool type (collides with `YAMLTool`), tool kind, backend.

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
the tool is silently not registered — the model never sees a tool it
cannot dispatch on this host. Resolution is via `shutil.which`.
_Avoid_: filter, guard, condition.

**Discovery path**:
A directory scanned for YAML tool declarations at startup. **Project
tools** live under `.agents/tools/` (relative to cwd); **user tools** live
under `~/.config/cothis/tools/` (global across all projects). Both are
optional; absence is not an error. User-global loads first, project-local
loads second, and project-local shadows user-global on name conflict.
Today only the project-local path is wired into `cli.py`; the
user-global path is planned (issue #1, stories 20 and 33).
_Avoid_: config dir, registry root, tools folder.

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
