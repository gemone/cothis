# Tool lifecycle hooks: five-stage chain with `on_error`

Tools carry a five-stage lifecycle hook chain (`pre_load` / `after_load` /
`pre_execute` / `after_execute` / `on_error`). Every tool inherits from
`_HookableTool`, which owns hook storage and the four invocation methods.
`_execute` runs hooks uniformly — no per-source branching.

## Decision

**Five stages, not four.** The original design (issues #6, #7) named four
stages. `on_error` was added during grilling because hook chains
short-circuit on exception (the observer needs an escape hatch for
audit/telemetry when any prior stage raises).

**`on_error` is pure side-effect.** It observes failures but cannot
recover them. Its own exceptions are swallowed (chain terminates,
`logger.debug` records). This avoids the phase-dependent type ambiguity
of "recovery" (recovery from `pre_execute` would mean args; from
`after_execute` would mean result — no single return type works).

**Chain semantics per stage:**

| Stage | Chain type | Exception → |
|---|---|---|
| `pre_load` | short-circuit AND | skip tool, on_error |
| `after_load` | all run (no short-circuit) | skip tool, on_error |
| `pre_execute` | pipeline (A's output → B's input) | short-circuit, on_error, error to LLM |
| `after_execute` | pipeline | short-circuit, on_error, use original result |
| `on_error` | short-circuit on its own exception | swallowed to `logger.debug` |

**`_HookableTool` base class, not mixin or protocol.** Both `ToolDef`
(Python tools) and `_ShellTool` (YAML tools) inherit it. YAML tools'
hook chains are empty no-ops; they don't register callbacks today. If
YAML hook support is needed later, it'll be a loader concern (e.g. load
a same-name `.py` file), not an `_execute` change.

**`_execute` uses duck-typing (`_run_hooks_safe`), not `isinstance`.**
This keeps the `Tool` Protocol minimal (`__name__` + `__call__` only)
and handles bare callables (lambdas, legacy `def`s) that don't inherit
`_HookableTool`.

## Considered alternatives

- **`isinstance(tool, ToolDef)` gate in `_execute`** — rejected: it
  introduces per-source branching, contradicting CONTEXT.md's "no
  per-source branching in `_execute`" principle.
- **`on_error` can recover (return a value)** — rejected: the return
  type is phase-dependent (args for pre_execute, result for
  after_execute), making the signature ambiguous.
- **Mixin instead of base class** — rejected: single base is clearer
  than Python's MRO-prone mixin pattern.
- **Hook logic as module-level functions** — rejected: each class would
  need forwarding boilerplate.

## Consequences

- Every tool (Python + YAML) passes through the hook pipeline in
  `_execute`, even if the chains are empty. The per-call overhead is one
  `getattr` + empty-list iteration — negligible.
- The `Tool` Protocol stays minimal (`__name__` + `__call__`). Hook
  methods are a runtime duck-typing surface, not a protocol requirement.
- Adding a sixth stage (e.g. `on_success`) would extend `_HOOK_STAGES`
  and add a method to `_HookableTool` — no `_execute` change needed
  unless the stage has new invocation semantics.
