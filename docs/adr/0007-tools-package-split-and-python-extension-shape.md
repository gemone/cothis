# Tools package split and Python-extension shape

Issue #1 (the tools-module PRD) proposed a `tools/` package of six modules:
`schema.py` (pydantic `BaseModel` tool base), `shell.py` (YAML), `mcp.py`,
`python_ext.py`, `registry.py`, and `__init__.py`. PRD explicitly flagged
this as a file-layout exception to AGENTS.md's "fewest files possible"
principle and recommended recording it as an ADR. This ADR records the
split that shipped and two related deviations from the PRD's user stories.

## Decision

**The package layout is `core.py` / `yaml.py` / `mcp.py` / `builtins.py` /
`format.py`, not the PRD's `schema.py` / `shell.py` / `mcp.py` /
`python_ext.py` / `registry.py`.** Naming follows the concern each
module owns, not the tool-source taxonomy:

- `core.py` — the shared foundation: `Tool` protocol, `_HookableTool`
  (lifecycle hooks), `@tool`/`ToolDef` (Python-tool API), schema
  helpers, shared validators, layer loading, `discover_tools` (layer
  merge + shadow + load hooks).
- `yaml.py` — the YAML shell-tool pipeline (`load_yaml_tools` →
  `CommandBlock` → `_compile` → `_ShellTool` → `preview`).
- `mcp.py` — `MCPServer`, `MCPClientTool`, built on the SDK's
  `ClientSessionGroup`.
- `builtins.py` — `fs.read` / `fs.dir` / `fs.write` + the `TOOLS`
  registry (the builtin layer).
- `format.py` — `format_tool_output` (json/csv/tsv/yaml serialisation
  of structured tool results).

A new submodule is justified when a concern is clearly distinct AND
its extraction leaves the source file more focused — not to mirror an
external taxonomy. The test for adding one is the same as the test
that justified the original split: does the file it came from become
easier to reason about?

**Deviation from PRD story 34 — Python extensions have no `TOOLS`
export contract.** The PRD asked each Python extension file to export a
`TOOLS` list the loader would register. The shipped loader auto-scans
each imported module for module-level `@tool`-decorated `ToolDef`
instances (`isinstance(obj, ToolDef)`). Authors just decorate; no
export boilerplate. `TOOLS` survives only in `builtins.py` as the
*builtin* registry — a layer input to `discover_tools`, not a
public extension contract.

**Deviation from PRD story 38 — Python extensions are a peer source,
not a thin wrapper over the shell template.** The PRD asked Python
extensions to be implemented "as a thin wrapper over the shell-tool
template, so that there is one extension concept, not two" — i.e.
Python extension files call the `shell()` helper. The shipped design
treats `@tool`-decorated Python functions as first-class tools (the
same API the built-in `fs.read` / `fs.write` use); the `shell()`
helper is available for Python tools that need shell glue, but
optional. Reason: forcing every Python extension through `shell()`
would give up the schema fidelity win (`Annotated[T, Field(...)]`
constraints, rich per-arg descriptions) that `@tool` provides — the
core motivation of the PRD itself (stories 1–10).

## Considered alternatives

- **The PRD's exact module names.** Rejected: `schema.py` implies a
  pydantic base class (stories 1–10, dropped — see CONTEXT.md "Tool
  source"), `python_ext.py` and `registry.py` would each be small
  files doing one job that `core.py` already absorbs cleanly. The
  shipped names match the actual concerns after the pydantic-base
  decision was reversed.

- **Keep `tools.py` as one file.** Rejected: at ~2200 LOC it was doing
  five jobs (protocol + hooks, Python tool API, YAML pipeline, MCP
  subsystem, output formatting, plus the builtin fs tools). The split
  was the original justification for the file-layout exception the PRD
  asked for; not splitting would have left a file too large to reason
  about.

- **Honour story 38 as written (Python extensions = `shell()` calls
  only).** Rejected: it would re-introduce the lossy-schema problem
  the PRD set out to solve, and the `@tool` API is strictly more
  expressive (rich types, hooks) without being harder to use.

## Consequences

- **`from cothis.tools import TOOLS` is not supported.** Story 39
  (backward-compat for the legacy import path) is dropped: the only
  in-tree consumer (`cli.py`) was migrated to `discover_tools` in the
  same PR, and re-exporting `TOOLS` would invite new consumers to
  depend on a partial view (builtins only). The public aggregator is
  `discover_tools`. `TOOLS` stays in `builtins.py` as an internal
  layer input.

- **Submodule import paths (`cothis.tools.core`, `.yaml`, `.mcp`,
  `.builtins`, `.format`) are part of the test surface.** Tests import
  from submodules directly (e.g. `from cothis.tools.yaml import
  preview, shell`). The package `__init__.py` re-exports only the
  author-facing API; tests reach past it for white-box checks.

- **Adding a sixth submodule requires justifying it the same way the
  original split was justified.** A new concern that's clearly
  distinct AND whose extraction leaves its source file more focused
  is the bar — not mirroring an external taxonomy or matching the
  PRD's original naming.

- **Story 21 (multiple shell tools per YAML via `tools:` list) is
  deferred.** Tracked in a follow-up issue; this ADR doesn't speak to
  it beyond noting that the current `_TOOL_KEYS` set excludes `tools:`
  and `_compile` parses one tool per file.
