# Anthropic Messages API migration

Date: 2026-07-18

Status: Accepted. Supersedes the tool-schema-shape assumptions in
ADR-0005 (which were OpenAI/Chat-Completions-shaped); ADR-0005 is
otherwise left in force.

## Context

cothis's agent loop was built on `any_llm.acompletion` — the OpenAI
Chat Completions wire format. `system` was a `{role: "system"}` message;
assistant turns carried OpenAI `tool_calls`; tool results were
`{role: "tool", tool_call_id, content}` messages; streamed tool
arguments arrived as JSON-string fragments reassembled by
`_assemble_tool_calls` + best-effort-parsed by `_safe_parse_args`. Tool
schemas were OpenAI-shaped (`{type: "function", function: {name,
description, parameters}}`).

PR1 (#29) needs `system` as a first-class parameter (persona + AGENTS.md
blocks, each with `cache_control`), native content blocks (so a future
session store can persist one row per block), and `thinking` blocks for
turn coherence. None of these fit the Completion shape without ad-hoc
encoding. Meanwhile `max_tokens` is hardcoded rather than matched to the
model.

## Decision

**Migrate the internal wire format to `any_llm.amessages` (the Anthropic
Messages API), end-to-end.** `system` becomes a top-level parameter (a
list of content blocks); `messages` hold only user/assistant turns as
Anthropic block lists; tool schemas are Anthropic-shaped throughout.

This ADR records the decisions landed in #31. Related PR1 decisions
validated in research but implemented in later slices are indexed under
"PR1 follow-on decisions" with their implementing issue.

### 1. `amessages` as the internal wire format

The agent calls `any_llm.amessages(model, messages, max_tokens, *,
system, tools, stream, …)`. `system` is a top-level list of content
blocks (not a `{role: system}` message); each block carries
`cache_control: {type: ephemeral}`. For #31 the list is just the persona
block; #33 adds the AGENTS.md block; #30 reserves a catalog slot.

any-llm's default `_amessages` auto-converts Messages↔Completions for
non-Anthropic providers (validated end-to-end on openrouter). The
Anthropic provider passes `system`/`messages` through verbatim to
`client.messages.create` (`MessagesParams.model_dump(exclude_none=True)`),
so block-level `cache_control` reaches the API unmodified.

`max_tokens` is passed through; #31 hardcodes 8192, #32 resolves it
from the bundled litellm JSON.

**Considered: the Responses API.** Rejected: not provider-portable
(Anthropic-only). The Messages API is portable across providers via
any-llm's converter, and clearer than Completion (`system` first-class,
content blocks native). Completion is deprecated.

### 2. Anthropic tool shape, end-to-end (supersedes ADR-0005 §schema)

`_build_schema` (`tools/core.py`), `_build_tool_schema`
(`tools/yaml.py`), and the `MCPClientTool` constructor (`tools/mcp.py`)
produce `{name, description, input_schema}`. `__cothis_schema__` stores
the Anthropic shape. `schema_for` is structurally unchanged
(`getattr(tool, "__cothis_schema__", tool)`) — shape-agnostic.

ADR-0005's schema mentions assumed OpenAI shape (`function.parameters`,
`__cothis_schema__` as an OpenAI schema); those assumptions are
superseded here. ADR-0005 is not edited.

The round-trip (Anthropic → OpenAI via any-llm's
`_convert_tools_to_openai`) is lossless for every shipped tool;
MCP schemas (with `$schema`/`$defs`/`$ref`/`enum`) pass through
verbatim as `input_schema`.

### 3. Message shape; `_messages` carries metadata, projected at send time

`messages` are user/assistant only. Assistant content is a block list
(`text` / `thinking` / `tool_use`); user content is a block list
(`text` / `tool_result` / `image`). Stored assistant dicts carry
response metadata (`id` / `model` / `stop_reason` / `usage`) for
inspection; `_request_messages` projects each to `{role, content}`
before the next `amessages` call (Anthropic's native API rejects extra
fields on message dicts).

Tool results are `tool_result` blocks inside user messages, with string
`content` (matching `_execute_tool`'s string contract). Errors carry the
Anthropic-native `is_error: true` flag so the model can distinguish a
failed call from a successful one. `_safe_parse_args` and its tests are
deleted — `tool_use.input` arrives already-parsed.

### 4. `run_stream` on `MessageStreamEvent`; turn decision by `stop_reason`

`run_stream` consumes the Anthropic stream-event union directly
(any-llm synthesises these events from OpenAI chunks for non-Anthropic
providers via `messages_compat`). Block state is seeded from
`ContentBlockStartEvent.content_block`; deltas accumulate per block
(`TextDelta`/`ThinkingDelta` append, `SignatureDelta` overwrites the
block's final signature, `InputJSONDelta` accumulates then parses at
block stop). Dispatch narrows the event union by `isinstance` (ty cannot
narrow by string `event.type` comparison).

**Turn decision uses `stop_reason` from `MessageDeltaEvent`**
(`== "tool_use"` → tool turn). Any other `stop_reason` ends the loop and
returns the concatenated `text` blocks (empty → `""`). This removes the
old content-None retry heuristic: `stop_reason` is the authoritative
end-of-turn signal, so there is no "did the provider drop content?"
ambiguity. `MessageStopEvent` is only a stream-termination latch — the
first one seen ends iteration (openrouter emits duplicates).

`thinking` blocks are accumulated passively (#31 does not pass the
`thinking` param, so claude won't emit them and other providers never
do); they are replayed verbatim if they arrive, since stripping them
makes the model re-invoke tools (validated).

### 5. `max_tokens` hardcoded this slice

#31 passes `max_tokens = 8192`. #32 replaces this with a resolver over
the bundled litellm `model_prices_and_context_window.json`, fallback
8192, overridable via `COTHIS_MAX_TOKENS` / `--max-tokens`, with a
weekly GitHub workflow refreshing the JSON.

## PR1 follow-on decisions (validated in #29 research; implemented later)

These are stable decisions recorded here as the PR1 index; each lands
in its own slice and may extend this ADR.

- **Session store — one row per block, WAL + `BEGIN deferred`.**
  Validated: `BEGIN deferred` fairness=1.0 across 4 concurrent writers;
  `BEGIN IMMEDIATE` starves. Hot DB at `$COTHIS_HOME/cothis.db`,
  overridable via `COTHIS_SESSIONS_DIR`. Implemented in #34.
- **Fork tree — in-memory graph over flat rows.** Validated 3-10×
  faster than a recursive CTE; `flat_load` of 10k sessions ≈ 9ms.
  Implemented in #35.
- **Cold/hot archival — `ATTACH` cross-DB atomic transactions.**
  Validated atomic + idempotent. Implemented in #36.
- **System prompt assembly — persona + 3-layer AGENTS.md.** Implemented
  in #33.

## Consequences

- Every provider (openrouter, anthropic, openai, mistral, …), every
  shipped tool (`fs.*`, YAML, MCP), and the streaming `cothis chat`
  experience keep working — validated by the round-trip + the offline
  unit suite (no LLM in tests).
- `_messages` is now homogeneous `list[dict[str, Any]]` (no
  `ChatCompletionMessage` in the union), which is the shape #34's
  session store will serialise one block at a time.
- The highest-risk change was `run_stream` (events synthesised from
  OpenAI chunks on non-Anthropic providers); it gets the most test
  attention (synthetic `MessageStreamEvent` flows via the real
  anthropic SDK `Raw*Event` constructors).
- `run_stream`'s accumulator depends on the anthropic SDK event types
  at runtime (the `isinstance` narrowing). any-llm re-exports these as
  `MessageStreamEvent`, so the dependency is already transitive.
