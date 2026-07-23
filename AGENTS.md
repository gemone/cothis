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

- No abstraction that wasn't requested. The package layout (`agent.py`, `cli.py`, `tools/`, `__init__.py`) is the intended shape. `tools/` is a package because the original `tools.py` outgrew one file (it was doing five+ jobs); its submodules are the result of that split, each owning one concern: `core.py` (Tool protocol, `_HookableTool`, `@tool`/`ToolDef`, schema helpers, shared validators, `discover_tools` — layer merge + shadow + load hooks), `yaml.py` (YAML shell-tool pipeline: `load_yaml_tools` → `CommandBlock` → `_compile` → `_ShellTool` → `preview`), `builtins.py` (re-export shim — `TOOLS` registry + `from cothis.tools.fs.{read,list,search,write} import …`), `tools/fs/` (`_hygiene.py` for WORKDIR + boundary, `patch.py` for codex apply_patch, `read.py`/`write.py`/`list.py`/`search.py` — `fs.read`/`fs.write`/`fs.list`/`fs.search`), `mcp.py` (`MCPServer`, `MCPClientTool`, built on the SDK's `ClientSessionGroup`), `format.py` (`format_tool_output` — json/csv/tsv/yaml). A new submodule is justified when a concern is clearly distinct AND its extraction leaves the source file more focused — not to mirror an external taxonomy. The test for adding one is the same as the test that justified the original split: does the file it came from become easier to reason about?
- No new dependency if it can be avoided. `pyproject.toml` is lean on purpose.
- No boilerplate nobody asked for. No config layers, no plugin systems, no settings pydantic-model wrapping what env vars already do.
- Deletion over addition. Boring over clever. Fewest files possible.
- Question complex requests: "Does cothis actually need X, or does the existing loop + `any-llm` cover it?"
- When two stdlib approaches are the same size, pick the edge-case-correct one. Lean means less code, not the flimsier algorithm.
- Mark intentional simplifications with a `cothis:` comment. If the shortcut has a known ceiling (single-turn tool call, no streaming, no tool-parallelism, hardcoded prompt), the comment names the ceiling and the upgrade path.

Never compromise on: prompt correctness (the LLM only sees what you send — drift here is silent breakage), tool I/O safety (the agent writes files via `fs.write` — confirm paths are scoped as intended), error messages that the LLM can act on (vague errors break the loop), API key handling (read once, never log), CLI ergonomics (flags and env vars behave as documented in the README), external API signatures and intent (verify by inspecting the real signature — `inspect.signature`, `help()`, source — and a one-liner repro, not from memory; passing `.status(transient=True)` when `rich.Status.__init__` already hardcodes `Live(transient=True)` is a `TypeError`, and relying on a default you assumed rather than confirmed is silent breakage when the assumption is wrong). Code without its check is unfinished: non-trivial logic leaves ONE runnable check behind, the smallest thing that fails if the logic breaks. cothis uses **pytest** (configured in `pyproject.toml`, tests under `tests/`) — reach for it for anything beyond a one-liner; for trivial checks a `python -c "assert ..."` is fine. Fixtures like `tmp_path` / `monkeypatch` / `caplog` are welcome when they make the test honest; what's not welcome is a test so buried in mocks that it tests the mock, not the logic.

## Tool description standard

Every model-facing tool description (the `description=` kwarg on `@tool(...)`, **distinct from the Python docstring**) must carry four things, in the same order the prior-art PRs converged on:

1. **Return-format signal** — name the shape. Dict fields (`[{name, type}]`), list shape, string sentinel (`"fs.create: created hello.txt (1 lines)"`). One phrase is enough; the model only needs to recognise the result on first read.
2. **Field semantics** — for each named field, one phrase on what it carries + any cross-reference. `line` is 1-based and matches `fs.read` numbering; `type` is `"file"` or `"dir"`; `content` may be multi-line. If the field references another tool's output, say so.
3. **Concrete example** — one canonical invocation with a return preview. **English only** — non-ASCII examples inflate token cost (~1.5× under BPE per the #108/#124 audit) and every prior-art PR used English. Form is an RST literal block:
4. **Boundary notes** — caps, exclusion rules, idempotency hints. Past 500 entries the shape changes to `{truncated: true}`; sensitive files (`*.env`, private keys) are excluded regardless of `glob`; repeated calls on an already-active skill return a short notice. Anything non-obvious the model would otherwise learn by failing.

Example (from `fs.read`, after PR #197):

```python
_READ_DESCRIPTION = """Read UTF-8 text files with 1-based line numbers (tab-separated).

Pass a single path or a list. Each line in the output is prefixed
with its line number and a tab so you can reference exact lines in
``start_line`` / ``end_line`` on follow-up calls.

Single path — returns one numbered block::

    fs.read(path='config.py')
    → 1\tdebug = True
      2\tport = 8080

In a multi-path call, one missing file produces an
``Error: file not found: <path>`` block for that file; the others
return normally (the call does not abort).
"""
```

Points hit: (1) "1-based line numbers (tab-separated)" + `→ 1\tdebug = True` preview; (2) "Each line...prefixed with its line number and a tab"; (3) `Example::` block with `fs.read(path=...)` invocation; (4) "one missing file produces an Error...the call does not abort."

The Python docstring on the function is for humans reading the source and is unchanged by this standard.

### Why this test shape

`tests/test_tool_description_audit.py` enforces the standard as a CI floor via string-presence checks (description non-empty, length ≥ 120, contains `toolname(`, contains `Example` or `::`, contains `→` or `Returns`). Three alternatives were considered:

- **AST / structural parser.** Rejected: descriptions are free-form prose with embedded RST literal blocks; parsing them into a strict schema either rejects valid prose variants or forces a template that kills readability. The cost of a stricter parser isn't worth the marginal regression it catches beyond the string floor.
- **LLM judge over a fixture set.** Rejected: non-deterministic in CI, adds latency + cost per run, and the regression mode (one-line description with no structure) is already caught by the string floor.
- **No test, human review only.** Rejected: this is the pre-#191 status quo. Seven issues (#190, #192, #194, #196, #199, #201, #203) were opened on the same regression before the standard was written; human review without a check kept missing it.

The floor is deliberately permissive: it fails the obvious regression (one-line description) and passes anything a reasonable reviewer would accept. Tightening belongs in code review, not in the test.

### Prior art

The PRs that established the standard (issue → PR):

- #190 → #191 (`fs.write`, since removed)
- #192 → #193 (`deactivate_skill`)
- #194 → #195 (`load_skill`)
- #196 → #197 (`fs.read`)
- #199 → #200 (`fs.search`)
- #201 → #202 (`fs.list`)
- #203 → #204 (`code.lines`)

## Agent skills

### Issue tracker

GitHub issues for `gemone/cothis`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
