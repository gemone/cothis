# Layer precedence: three tiers, winner-takes-all, no fallback

Tools are discovered from three **layers** in ascending precedence:
**builtins** (`TOOLS`, compiled in) < **user-global**
(`~/.config/cothis/tools/`) < **project-local** (`.agents/tools/`). A
name conflict across layers is resolved by **shadowing** — the higher
precedence layer wins, the lower is dropped. The shadowed tool's load
hooks never fire; the winner's load hooks fire after merge; if the
winner's `pre_load` returns `False` the slot goes **empty** (no fallback
to the shadowed tool).

## Decision

**Format is never a layer.** Tool source (Python / YAML / MCP — see
CONTEXT.md "Tool source") is the *format* axis only. Only the discovery
tier (builtin / user-global / project-local) determines precedence. Two
tools of any format combination in the *same* layer claiming one name
are an author error (raise `ValueError`); cross-layer conflicts shadow.

**Load hooks run on the merged winner only, after shadow resolution.**
`pre_load` / `after_load` fire once per name, on whichever tool won the
cross-layer merge. A shadowed tool's load hooks never fire — discovery
collects candidates from all layers, shadow resolution collapses to one
winner per name, and only then do load hooks run.

**`pre_load=False` empties the slot; no fallback.** When the winner's
`pre_load` returns `False` (or raises), the tool is dropped and the
slot stays empty. The lower-layer tool that was shadowed is *not*
restored. Shadowing is a replacement, not a try.

## Considered alternatives

- **Format as a layer** (e.g. Python shadows YAML within the same
  directory). Rejected: the issues' "same-source raise, cross-source
  shadow" language cannot classify a YAML file and a Python file in the
  same `.agents/tools/` claiming one name — both "same-layer" (raise)
  and "cross-source" (shadow) apply. Picking format-as-layer would make
  the directory's behaviour depend on file ordering and authoring
  language, which is silent breakage.
- **Load hooks fire per-discovered-tool** (the pre-Q3 behaviour, where
  `_run_load_hooks` ran inside `load_python_tools_from_dir`). Rejected:
  a shadowed tool's `after_load` audit/metric callbacks would fire even
  though the tool is never dispatched — side effects for a tool the
  model never calls. Moving load hooks to post-merge ensures hooks
  observe only tools that will actually be registered.
- **`pre_load=False` falls back to the shadowed tool.** Rejected:
  fallback turns shadow into a multi-valued operation (the shadowed
  tool must be retained as a candidate, not dropped), contradicting the
  winner-takes-all merge. It is also a silent surprise: an author
  declares a project-local override expecting it to mask the
  user-global tool, but a `pre_load` condition they may not have
  noticed secretly re-enables the user-global tool. No fallback is
  honest — "you shadowed with a tool whose own pre_load says don't
  load; the name is gone here."

## Consequences

- **Cross-layer merge is a dict overwrite in ascending precedence
  order.** Builtins load first, user-global overwrites by name,
  project-local overwrites by name. Each overwrite is one
  `logger.warning` naming both layers and source paths (see
  CONTEXT.md "Tool lifecycle" — every load/dispatch decision is
  observable at `WARNING`).
- **Same-layer duplicate detection happens *inside* the per-layer
  loader** (`load_tools_from_layer`), where YAML and Python candidates
  share one `seen` dict. A YAML file and a Python file in the same
  `.agents/tools/` claiming one name raise — they are same-layer, not
  cross-source. The per-loader `seen` dicts in `load_tools_from_dir` /
  `load_python_tools_from_dir` are folded into one shared `seen` per
  directory.
- **Audit/telemetry authors must put load hooks on the winning tool.**
  A user-global tool's `after_load` callback will not fire if a
  project-local tool shadows it. This is correct (the user-global tool
  is not dispatched) but means cross-cutting concerns (e.g. "log every
  fs.read that *could* have run") can't be expressed as load hooks on
  lower layers — they belong in `pre_execute` on the winner, or in a
  discovery-side observability hook (not yet built).
- **`pre_load` cannot express "try this, else fall back."** Authors
  who want fallback must use a different mechanism (e.g. `pre_load`
  that succeeds and delegates internally), not shadowing. This is a
  deliberate ceiling.
- **Discovery order is fixed:** builtin → user-global → project-local.
  `_all_tools(project_dir, user_dir)` takes both paths explicitly
  (pure function of inputs); cli.py supplies the literal paths. No
  upward directory search for a project root — discovery is
  cwd-relative (`.agents/tools/`) plus the fixed user-global path
  (`~/.config/cothis/tools/`).
