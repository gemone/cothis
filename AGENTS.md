## Project rules

You are building a *basic* coding agent. Basic means small on purpose, not flimsy. The best feature is the one never shipped.

Before writing any code, stop at the first rung that holds:

1. Does this need to be built at all? (YAGNI — does the ReAct loop already cover it?)
2. Does Python's stdlib already do this? Use it.
3. Does `any-llm` already expose this (provider switching, tool calling, response shape)? Use it.
4. Does an already-installed dependency (`pydantic`, `typer`) solve it? Use it.
5. Can this be one function? Make it one function.
6. Only then: write the minimum code that works.

Rules:

- No abstraction that wasn't requested. The four-file layout (`agent.py`, `cli.py`, `tools.py`, `__init__.py`) is the intended shape — resist splitting it further unless a file is clearly doing two jobs.
- No new dependency if it can be avoided. `pyproject.toml` is lean on purpose.
- No boilerplate nobody asked for. No config layers, no plugin systems, no settings pydantic-model wrapping what env vars already do.
- Deletion over addition. Boring over clever. Fewest files possible.
- Question complex requests: "Does cothis actually need X, or does the existing loop + `any-llm` cover it?"
- When two stdlib approaches are the same size, pick the edge-case-correct one. Basic means less code, not the flimsier algorithm.
- Mark intentional simplifications with a `cothis:` comment. If the shortcut has a known ceiling (single-turn tool call, no streaming, no tool-parallelism, hardcoded prompt), the comment names the ceiling and the upgrade path.

Not basic about: prompt correctness (the LLM only sees what you send — drift here is silent breakage), tool I/O safety (the agent writes files via `fs.write` — confirm paths are scoped as intended), error messages that the LLM can act on (vague errors break the loop), API key handling (read once, never log), CLI ergonomics (flags and env vars behave as documented in the README). Basic code without its check is unfinished: non-trivial logic leaves ONE runnable check behind, the smallest thing that fails if the logic breaks (a `python -c "assert ..."` one-liner or one small script under `src/cothis/`; no test framework, no fixtures). Trivial one-liners need no test.

(This file also applies to agents working on the cothis repo itself. Especially to them.)

## Agent skills

### Issue tracker

GitHub issues for `gemone/cothis`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
