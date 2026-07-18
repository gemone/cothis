## Project rules

You are building a *complete* coding agent. Engineering discipline keeps it lean: fewer files, fewer dependencies, no speculative abstraction — but never at the cost of missing capability.

Before writing any code, stop at the first rung that holds:

1. Does this need to be built at all? (YAGNI — does the ReAct loop already cover it?)
2. Does Python's stdlib already do this? Use it.
3. Does `any-llm` already expose this (provider switching, tool calling, response shape)? Use it.
4. Does an already-installed dependency (`pydantic`, `typer`) solve it? Use it.
5. Can this be one function? Make it one function.
6. Only then: write the minimum code that works.

Rules:

- No abstraction that wasn't requested. The package layout (`agent.py`, `cli.py`, `tools/`, `__init__.py`) is the intended shape. `tools/` is a package because the original `tools.py` outgrew one file (it was doing five+ jobs); its submodules are the result of that split, each owning one concern: `core.py` (Tool protocol, `_HookableTool`, `@tool`/`ToolDef`, schema helpers, shared validators, `discover_tools` — layer merge + shadow + load hooks), `yaml.py` (YAML shell-tool pipeline: `load_yaml_tools` → `CommandBlock` → `_compile` → `_ShellTool` → `preview`), `builtins.py` (`fs.read`/`fs.dir`/`fs.write` + `TOOLS` registry), `mcp.py` (`MCPServer`, `MCPClientTool`, built on the SDK's `ClientSessionGroup`), `format.py` (`format_tool_output` — json/csv/tsv/yaml). A new submodule is justified when a concern is clearly distinct AND its extraction leaves the source file more focused — not to mirror an external taxonomy. The test for adding one is the same as the test that justified the original split: does the file it came from become easier to reason about?
- No new dependency if it can be avoided. `pyproject.toml` is lean on purpose.
- No boilerplate nobody asked for. No config layers, no plugin systems, no settings pydantic-model wrapping what env vars already do.
- Deletion over addition. Boring over clever. Fewest files possible.
- Question complex requests: "Does cothis actually need X, or does the existing loop + `any-llm` cover it?"
- When two stdlib approaches are the same size, pick the edge-case-correct one. Lean means less code, not the flimsier algorithm.
- Mark intentional simplifications with a `cothis:` comment. If the shortcut has a known ceiling (single-turn tool call, no streaming, no tool-parallelism, hardcoded prompt), the comment names the ceiling and the upgrade path.

Never compromise on: prompt correctness (the LLM only sees what you send — drift here is silent breakage), tool I/O safety (the agent writes files via `fs.write` — confirm paths are scoped as intended), error messages that the LLM can act on (vague errors break the loop), API key handling (read once, never log), CLI ergonomics (flags and env vars behave as documented in the README), external API signatures and intent (verify by inspecting the real signature — `inspect.signature`, `help()`, source — and a one-liner repro, not from memory; passing `.status(transient=True)` when `rich.Status.__init__` already hardcodes `Live(transient=True)` is a `TypeError`, and relying on a default you assumed rather than confirmed is silent breakage when the assumption is wrong). Code without its check is unfinished: non-trivial logic leaves ONE runnable check behind, the smallest thing that fails if the logic breaks. cothis uses **pytest** (configured in `pyproject.toml`, tests under `tests/`) — reach for it for anything beyond a one-liner; for trivial checks a `python -c "assert ..."` is fine. Fixtures like `tmp_path` / `monkeypatch` / `caplog` are welcome when they make the test honest; what's not welcome is a test so buried in mocks that it tests the mock, not the logic.

## Agent skills

### Issue tracker

GitHub issues for `gemone/cothis`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
