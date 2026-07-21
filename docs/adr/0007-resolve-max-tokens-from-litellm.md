# Resolve `max_tokens` from bundled litellm model metadata

Date: 2026-07-19

Status: Accepted. Implements PR1 decision recorded under "PR1 follow-on
decisions" in ADR-0006 (item 5: "max_tokens hardcoded this slice"). The
hardcoded `_DEFAULT_MAX_TOKENS = 8192` constant landed in #31 is removed.

## Context

`Agent` calls `any_llm.amessages(..., max_tokens=...)` on every turn.
#31 hardcoded `8192` for every model. Wrong-sized caps silently hurt:

- Too small → a generation can be cut off mid-tool-call. The PR #37
  security review already had to fix the loop so a partial `tool_use`
  block with `stop_reason == "max_tokens"` still gets a paired
  `tool_result` (otherwise the session 400s next turn). Right-sizing
  the cap from the start removes most of those cutoffs.
- Too large → the provider rejects the request with a 400.

The right cap is the model's own `max_output_tokens`, which litellm
publishes in `model_prices_and_context_window.json`. PR1 #29 asks for
this resolved per-model, overridable, with a workflow keeping the
metadata current.

## Decision

**Bundle litellm's JSON in the wheel; resolve `max_output_tokens` by
model id at the first `amessages` call; let an explicit override win.**

### 1. Bundle, not runtime-fetch

`src/cothis/data/model_prices.json` ships in the wheel (read via
`importlib.resources`). The JSON is ~1.6 MB; bundle-once beats a network
call per process (no offline breakage, no cold-start latency, no
dependency on litellm as a runtime package). A weekly GitHub workflow
(`.github/workflows/update-model-prices.yml`) refreshes the file by PR.

`functools.cache` on `_metadata()` means the JSON is parsed once per
process even across multiple `Agent` constructions (relevant for tests).

### 2. Matching strategy

`resolve_max_tokens(model, provider, override=None)` tries, in order:

1. `override` (if a positive int) — explicit `--max-tokens` /
   `COTHIS_MAX_TOKENS` always wins.
2. Exact `model` key — e.g. `claude-sonnet-4-5`, `gpt-4.1-mini`.
3. `{provider}/{model}` key — the form litellm uses for providers that
   prefix model ids (`openrouter/openai/gpt-oss-120b`,
   `mistral/mistral-small-latest`).
4. Fallback `8192`.

Field precedence on a matched entry: `max_output_tokens` first (modern
field); fall back to legacy `max_tokens` **only when `max_input_tokens`
is also absent**. Per litellm's own `sample_spec.max_tokens` contract,
when `max_input_tokens` is set the legacy `max_tokens` duplicates it
(the *input* cap) — returning that as the output cap would inflate the
`max_tokens` argument and 400 the first `amessages` call. The
conditional rule lands in #64. As of the bundled JSON audited at #64's
fix, **111 chat-reachable entries** (non-embedding, token-costed) hit
the legacy path; **23 are misclassified** (have `max_input_tokens`, so
the legacy field is the input cap and is correctly skipped) including
`perplexity/sonar*`, `openrouter/auto`/`free`/`bodybuilder`,
`azure/mistral-large-*`, `azure/gpt-3.5*-instruct*`, and many
`together_ai/Qwen*` / `together_ai/openai/gpt-oss-20b`. The remaining
**88 genuinely lack `max_input_tokens`** and legitimately use
`max_tokens` as the output cap (e.g. `replicate/anthropic/*`,
`openrouter/gryphe/mythomax-l2-13b`, many `together-ai-*` bucket
proxies).

### 3. Known ceiling — provider-name divergence

litellm's `litellm_provider` field names diverge from any-llm's provider
keys (e.g. `together_ai` vs `together`, `fireworks_ai` vs `fireworks`).
The resolver does **not** fuzzy-match on the provider field — doing so
would require a hand-maintained name map that drifts as either side
renames. A model whose only key in litellm is provider-prefixed under a
divergent name resolves to the `8192` fallback; the user overrides with
`--max-tokens` for that model. Documented in the README.

### 4. Agent wiring

New `Agent.max_tokens: int | None = None` field. `None` → resolved once
on first `amessages` call and cached in `_resolved_max_tokens`
(`PrivateAttr`, `-1` sentinel = unresolved). Explicit int wins and is
never re-resolved. One resolution site (`_effective_max_tokens`) covers
both `run` and `run_stream`. The `_DEFAULT_MAX_TOKENS` constant from #31
is deleted.

### 5. CLI plumbing

`--max-tokens` / `COTHIS_MAX_TOKENS` added to both `ask` and `chat`,
type `int | None`, default `None`. Same precedence as the existing
`--provider` / `COTHIS_PROVIDER` pair (explicit flag > env var > None →
resolver).

## Considered

- **Depend on `litellm` at runtime and call `litellm.get_max_tokens`.**
  Rejected: litellm is a heavy dependency (`pip install litellm` pulls
  ~100 transitive packages) for a single metadata lookup. Bundling the
  one JSON file keeps `pyproject.toml` lean (the project rule — no new
  dependency if stdlib + a data file can do it).
- **Fetch the JSON over HTTP at first call, cache locally.** Rejected:
  adds a network round-trip to cold start, breaks offline, and the file
  changes slowly enough that a weekly PR refresh is fine.
- **Fuzzy match on `litellm_provider`.** Rejected (see §3).

## Consequences

- Every model resolves to its real output cap where litellm knows it;
  unknown models fall back to `8192` (the value #31 hardcoded, so no
  regression).
- The bundled JSON adds ~1.6 MB to the wheel. Acceptable: it's static
  data, gzips well, and ships one metadata file rather than a runtime
  dependency tree.
- The weekly workflow means new models land via a reviewable PR (not
  silently at runtime). No PR on no-diff weeks (idempotent).
- `--max-tokens` / `COTHIS_MAX_TOKENS` is now part of the public CLI
  surface and the documented override path.
