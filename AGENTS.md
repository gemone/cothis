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

- No abstraction that wasn't requested. The package layout (`agent.py`, `cli.py`, `tools/`, `__init__.py`) is the intended shape. `tools/` is a package because the original `tools.py` outgrew one file (it was doing five+ jobs); its submodules are the result of that split, each owning one concern: `core.py` (Tool protocol, `_HookableTool`, `@tool`/`ToolDef`, schema helpers, shared validators, `discover_tools` — layer merge + shadow + load hooks), `yaml.py` (YAML shell-tool pipeline: `load_yaml_tools` → `CommandBlock` → `_compile` → `_ShellTool` → `preview`), `builtins.py` (re-export shim — `TOOLS` registry + `from cothis.tools.fs.{read,list,search,create,modify,delete} import …`), `tools/fs/` (`_hygiene.py` for WORKDIR + boundary, `read.py`/`list.py`/`search.py`/`create.py`/`modify.py`/`delete.py` — `fs.read`/`fs.list`/`fs.search`/`fs.create`/`fs.modify`/`fs.delete`), `mcp.py` (`MCPServer`, `MCPClientTool`, built on the SDK's `ClientSessionGroup`), `format.py` (`format_tool_output` — json/csv/tsv/yaml). A new submodule is justified when a concern is clearly distinct AND its extraction leaves the source file more focused — not to mirror an external taxonomy. The test for adding one is the same as the test that justified the original split: does the file it came from become easier to reason about?
- No new dependency if it can be avoided. `pyproject.toml` is lean on purpose.
- No boilerplate nobody asked for. No config layers, no plugin systems, no settings pydantic-model wrapping what env vars already do.
- Deletion over addition. Boring over clever. Fewest files possible.
- Question complex requests: "Does cothis actually need X, or does the existing loop + `any-llm` cover it?"
- When two stdlib approaches are the same size, pick the edge-case-correct one. Lean means less code, not the flimsier algorithm.
- Mark intentional simplifications with a `cothis:` comment. If the shortcut has a known ceiling (single-turn tool call, no streaming, no tool-parallelism, hardcoded prompt), the comment names the ceiling and the upgrade path.

Never compromise on: prompt correctness (the LLM only sees what you send — drift here is silent breakage), tool I/O safety (the agent writes files via `fs.create` / `fs.modify` / `fs.delete` — confirm paths are scoped as intended), error messages that the LLM can act on (vague errors break the loop), API key handling (read once, never log), CLI ergonomics (flags and env vars behave as documented in the README), external API signatures and intent (verify by inspecting the real signature — `inspect.signature`, `help()`, source — and a one-liner repro, not from memory; passing `.status(transient=True)` when `rich.Status.__init__` already hardcodes `Live(transient=True)` is a `TypeError`, and relying on a default you assumed rather than confirmed is silent breakage when the assumption is wrong). Code without its check is unfinished: non-trivial logic leaves ONE runnable check behind, the smallest thing that fails if the logic breaks. cothis uses **pytest** (configured in `pyproject.toml`, tests under `tests/`) — reach for it for anything beyond a one-liner; for trivial checks a `python -c "assert ..."` is fine. Fixtures like `tmp_path` / `monkeypatch` / `caplog` are welcome when they make the test honest; what's not welcome is a test so buried in mocks that it tests the mock, not the logic.

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

## Startup latency budget

`cothis --help` is the first thing every user runs. It must feel instant. This section pins the budget and the import rule that keeps it honest.

### Baselines + ceilings (2026-07-24)

Measured per the methodology in `tests/test_startup_latency.py` — 3-run median wall time of `python -c "import cothis.cli; cothis.cli.app(['--help'])"` minus `python -c "pass"` on the same host:

| Platform | Baseline | Ceiling | Rationale |
|----------|---------:|--------:|-----------|
| Linux    | ~390 ms  | 600 ms  | 1.5× baseline — headroom for CI variance |
| macOS    | ~1010 ms | 1200 ms | bumped from 750ms after CI measured 1010ms; tighten after stable data |
| Windows  | ~840 ms  | 1300 ms | #45 post-deferral data point × 1.5 |

**Target:** the issue asks for `baseline + 50 ms` per platform. Ceilings here are conservative (1.5× baseline) because CI runners vary by ~50–100 ms run-to-run; tighten toward the 50 ms target once we accumulate stable data. A gross regression (a new top-level `import tensorflow`) gets caught today; a subtle one (a new 30 ms import) may not.

### Lazy-import rule

Every third-party top-level import in the startup path (`cothis/__init__.py`, `cothis/cli.py`, `cothis/agent.py`) must **either**:

1. **Carry an inline `# cost: ~Nms` comment** naming its measured cost — so a reviewer sees the hit in the diff. Example: `from pydantic import BaseModel  # cost: ~5ms`.
2. **Be deferred** under `if TYPE_CHECKING:` or inside the function that first uses it. The patterns from #45 (anthropic SDK), #81 (griffe), #118 (follow-ups) are the template.

Stdlib imports (`pathlib`, `typing`, `os`, `sys`, `asyncio`, …) are exempt — they're cheap and the audit ignores them. The third-party set is `any_llm`, `anthropic`, `click`, `filelock`, `griffe`, `mcp`, `pathspec`, `prompt_toolkit`, `pydantic`, `rich`, `typer`, `yaml` — derived from `pyproject.toml`'s `[project.dependencies]` plus transitive module names.

Enforced by `tests/test_startup_latency.py::test_no_unjustified_third_party_imports` (AST-based, not string match — handles `if TYPE_CHECKING` blocks correctly).

### Profile a new import

To measure a new third-party import's cost before adding it to the startup path:

```
python -X importtime -c "import <package>" 2>&1 | tail -1
```

The `importtime` flag prints `self_us | cumulative_us | module` to stderr. Use the `self_us` value (first column) — that's the cost of the module itself, excluding transitive deps already loaded.

### Out of scope

- Runtime latency (turn time, tool dispatch) — separate concern.
- Cothis SDK import time for embedders — focus is the CLI.

## Renaming

When a PR renames a tool, concept, CLI flag, or public class/function/module, the docs drift silently until someone reads them — by which point multiple files carry the old name. This has shipped two bugs already (#150 AGENTS.md/CONTEXT.md drift on `fs.dir` → `fs.list`, #153 README drift on the same rename).

### When the checklist applies

A rename is any of these in a PR:

- Tool rename (`fs.dir` → `fs.list`).
- Concept rename (`active_skills` → `active set`).
- CLI flag rename (`--skill` → `--activate`).
- Public class / function / module rename (`ToolDef` → `Tool`).

Internal-only renames (a private `_helper` function, a local variable) are exempt — the surface area is bounded by the file.

### Post-rename scan

Run this grep from the repo root before opening the PR:

```
grep -rn '<old-name>' README.md AGENTS.md CONTEXT.md docs/ tests/
```

If `CONTEXT-MAP.md` exists (multi-context repo), include the sub-context `CONTEXT.md` files in the scan. The scan must return nothing (or only legitimate historical references — e.g. an ADR explaining why the rename happened).

### Scope (6 locations)

1. **`README.md`** — user-facing docs; first place drift is read.
2. **`AGENTS.md`** — project rules + conventions; drift here teaches contributors the wrong name.
3. **`CONTEXT.md`** (+ any sub-context per `CONTEXT-MAP.md`) — domain glossary; drift here breaks the ubiquitous-language invariant.
4. **`docs/adr/*.md`** — ADRs reference tool/concept names in their decision context. Historical references inside an ADR explaining the rename are legitimate; references that describe the *current* state as the old name are drift.
5. **`tests/*.py`** — tests reference names in assertions (`assert tool.name == "fs.write"`) and fixtures. These break loudly (test failure) on rename, which is the desired signal — but the test file itself should be updated as part of the rename PR, not as a follow-up.
6. **The PR's own description** — so the squash-merge commit message uses the new name. A rename PR titled `feat(tools): add fs.dir` (when the new name is `fs.list`) leaves a permanent misnamed commit.

### PR-description convention

If the PR renames anything in scope, the description includes a one-line confirmation: *"Post-rename docs scan ran (AGENTS.md § Renaming); returned no drift outside legitimate historical references."* No template file is added — the project enforces conventions via review, not via a PR-template scaffold.

### Motivating bugs

- **#150** — `fs.dir` → `fs.list` rename (PR #101) left AGENTS.md + CONTEXT.md referring to `fs.dir`; caught by reading docs, fixed in #152.
- **#153** — same rename left README.md referring to `fs.dir`; caught separately, fixed in #162 alongside unrelated work.

Each was a separate fix PR; neither was caught by the rename PR itself. The checklist makes the rename PR responsible for the scan.
## External boundary fail-loud

Two classes of bug share a shape: code depends on something **external** — a third-party SDK's private attribute, or a prior registration it's about to overwrite — without a diagnostic when the assumption breaks. The user sees missing tools or shadowed commands with no log line. This rule makes the dependency loud.

Scoped to **boundary sites only** (init, registration, first-call setup) — not hot paths. Runtime cost matters inside loops; at boundaries it doesn't.

### Rule 1 — Third-party non-public attribute read

Every read of an attribute the upstream package hasn't documented as public must be preceded by a shape check (`isinstance` / `hasattr` / type assertion) that raises `RuntimeError` or logs `WARNING` naming: the expected shape, the actual shape, the upstream package, the ADR/issue reference.

**Signals that an attribute is non-public:** leading underscore (`_private`), absent from upstream's `__all__`, documented as "internal" in upstream docs.

**In-scope examples:**

- `group.tools` on `mcp.ClientSessionGroup` — guarded in `cothis.tools.mcp.connect_into` (#63, ADR-0005). The SDK may reshape its tool store or re-key entries; the guard fails loud at first connect.
- `client._raw` on a third-party client.

**Out of scope:**

- Python introspection builtins (`obj.__class__`, `obj.__dict__`, `type(obj)`, `hasattr(obj, "x")` itself).
- stdlib attributes.
- Attributes on our own internal classes (`session._lock`, `tool._handle_cls` — first-party).
- namedtuple `_replace` / `_fields` — "private" by convention but part of the public API.

### Rule 2 — Silent overwrite / shadow

Every code path that overwrites a prior registration (dict key, command name, tool name, env-derived path) at a boundary site must emit a `logger.warning` naming the key, the prior source, and the new source — unless the overwrite is the explicit documented contract.

**Overwrite is the contract (no warning needed):**

- `os.environ[...] = ...` — env vars are write-often by design.
- `self._cache[key] = ...` — internal cache, no prior source to name.

**Overwrite is NOT the contract (warning required):**

- `self._tool_map[key] = tool` — two different tools sanitising to the same wire name shadow silently (#112 pattern). Guarded in `Agent.model_post_init`.
- `_entries[name] = Entry(...)` in `slash.register` — a plugin registering `/skills` after the framework has registered `/skills` silently shadows (#112). Guarded in `cothis.slash.register`.

### Enforcement

`tests/test_boundary_fail_loud_audit.py` is a **regression guard**, not a new-bug finder. Each known boundary site is parametrised; the test verifies the guard is still present in the source. If a refactor moves or renames a site, the test turns red and the registry gets updated with the new location. New boundary sites are added to the registry when they're introduced — the rule is this section; the test is the backstop.

A general static scan for `obj._foo` attribute reads was considered and rejected: distinguishing third-party objects from first-party ones requires cross-module type inference that's noisy in practice. The registry approach is explicit + cheap + catches the actual regression mode (guard removed in a refactor).

### Motivating bugs

- **#63** — `MCPServer.connect_into` read `group.tools` without a shape guard. An SDK upgrade would silently break tool discovery. Fixed by the `isinstance` + `RuntimeError` guard that names the divergence + the ADR.
- **#112** — `slash.register` silently overwrote on duplicate name. A plugin registering `/skills` after the framework shadowed the framework command with no signal. Fixed by the collision `logger.warning`.

### Related

- **#80** — silent-except in `_build_schema` — same shape, different mechanism (exception swallow vs. attribute read). Out of scope here; the pattern is covered by the project's "no silent failure" stance in the Project rules section.
## Text boundary guard

Three rules for every line that decodes bytes, emits text, or mutates file content. Each covers a distinct boundary; each was paid for by a shipped bug.

1. **Decode bytes strict by default.** `open(..., encoding='utf-8', errors='strict')` — or just `encoding='utf-8'`, since strict is the default — for every path whose output reaches the model or the user. Locale fallback (UTF-8 → cp1252 → latin-1) is opt-in and must be a visible two-tier helper (#166), not a silent `errors='replace'` that injects U+FFFD into the prompt.
2. **Emit Unicode-native by default.** `json.dumps(..., ensure_ascii=False)` and `yaml.dump(..., allow_unicode=True)` are the project defaults. `ensure_ascii=True` is opt-in: it escapes every non-Latin codepoint to `\uXXXX`, inflating token cost ~1.5× under BPE (#108). If a downstream consumer genuinely requires ASCII (e.g. SMTP headers), add an inline `# text-boundary: allow` marker naming the consumer.
3. **Don't hardcode `\n` on replacement lines.** Code that mutates file content reads via `splitlines(keepends=True)` and joins via `"".join(...)`; the original terminators survive. Appending `+ "\n"` to a replacement line (the #96 / #215 pattern) produces mixed endings on CRLF files and spurious trailing newlines on no-trailing-newline files. Forbidden in `tools/fs/*`; outside that directory, prefer the same pattern but a literal `\n` is not flagged.

Each rule is enforced by `tests/test_text_boundary_audit.py` as a source-level scan over `src/cothis/**/*.py`. Violations can be suppressed with a same-line `# text-boundary: allow` comment when the call is genuinely justified (binary-safe regex search, ASCII-only wire format, etc.) — the marker keeps the rationale next to the call site, not in a separate allowlist file.

### Motivating bugs

- **#108** — `format_tool_output`'s JSON path defaulted to `ensure_ascii=True`, escaping CJK/emoji to `\uXXXX` (1.5× token cost).
- **#166** — `_parse_skill_md` used `errors='replace'`, silently substituting U+FFFD for undecodable bytes → garbled bytes flowed into `Skill.body` → `<skill_content>` → the system prompt (silent prompt-injection vector).
- **#96** — `_apply_hunk` hardcoded `\n` on replacement lines; CRLF files got mixed endings, no-trailing-newline files gained spurious trailing newlines. Anti-pattern now forbidden; motivating code deleted in #213.

## Agent skills

### Issue tracker

GitHub issues for `gemone/cothis`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
