"""Summariser selection + summarisation prompt builder (compaction slice A).

A pure-data companion to :mod:`cothis.ai.context_budget`. Where the budget
primitive reports *how full* the context window is, this module answers the
two questions compaction asks once the budget says "compact":

1. **Which** model/provider performs the summarisation? —
   :func:`resolve_summary_model`, a selector mirroring
   :func:`cothis.ai._retry.resolve_max_retries` (override arg > env var >
   session pair).
2. **What** do we send it? — :func:`build_summarisation_request`, a pure
   builder that renders a window of older turns into one user text message
   plus one system text block: the exact shape
   :meth:`cothis.ai.base.AIProvider.amessages` consumes.

Slice A deliberately stops here. It *selects* and *shapes*; it never
*executes* an ``amessages`` call and never wires into the live agent loop
(slice C owns the run-loop wiring via ``SessionPhase="compaction"``; slice B
owns the eviction policy that decides which turns constitute the window).

No I/O, no SDK imports, no provider-client construction — this preserves the
providers' lazy-client invariant and matches :mod:`cothis.ai.context_budget`'s
discipline. The environment is read only inside :func:`resolve_summary_model`
and only when the explicit ``override`` argument does not supply a value.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from cothis.ai.context_budget import (
    ContextBudget,
    PressureLevel,
    estimate_input_tokens,
)

# Operator-knob env var overriding the summariser model. Format is
# ``provider/model`` or a bare ``model`` (which inherits the session
# provider); empty / whitespace-only is treated as unset. Mirrors the
# ``COTHIS_MAX_TOKENS`` / ``--max-tokens`` idiom — a typer flag in front of
# this env var is wiring, deferred to slice C.
_SUMMARY_MODEL_ENV = "COTHIS_SUMMARY_MODEL"

# Default cap on the rendered transcript length, in characters. When the
# rendered window exceeds this the OLDEST turns are dropped (with a marker)
# so the most recent — highest-value — turns survive verbatim. 240k chars is
# roughly 60k tokens at the ~4-chars-per-token estimate (see
# :func:`cothis.ai.context_budget.estimate_input_tokens`), comfortably inside
# even a 128k context window once system + output headroom are budgeted.
_DEFAULT_MAX_WINDOW_CHARS = 240_000

# Per-block char caps so a single giant tool result or tool input cannot
# crowd out the rest of the transcript. Applied before the overall window
# cap; deliberately generous so a normal tool payload renders whole.
_TOOL_INPUT_CHAR_CAP = 1000
_TOOL_RESULT_CHAR_CAP = 4000

# Inserted where oldest turns were dropped, so the summariser knows the
# transcript was clipped at the *start* (recent context remains complete).
_TRUNCATION_MARKER = "...[older turns truncated]..."
_TRUNCATION_TAIL = "...[truncated]"

#: Default retention floor counted in TURN-GROUPS (a turn-group is one
#: user-typed message plus the assistant reply + any tool_use/tool_result
#: closure that follows it). The active tail is never compacted past this
#: floor, even under CRITICAL pressure. Four groups is roughly eight
#: messages in the no-tool case — enough recent context for the model to
#: stay grounded after the older turns are summarised.
#:
#: This default is the one open knob the user flagged for confirmation;
#: slice C may expose it via a typer flag in front of the same env-var
#: idiom as ``COTHIS_SUMMARY_MODEL``.
_DEFAULT_MIN_RETAINED_TURNS = 4

#: Post-eviction ratio targets. HIGH pressure reclaims down to the MEDIUM
#: boundary (~0.75 of capacity); CRITICAL is more aggressive (~0.50),
#: trading more summarisation for more headroom. Both are upper bounds on
#: the retained tail's estimated token count; the retention floor always
#: takes precedence over the ratio target.
_DEFAULT_HIGH_TARGET_RATIO = 0.75
_DEFAULT_CRITICAL_FLOOR_RATIO = 0.50

#: Default condensation instruction shipped as the system block. Written as a
#: running-summary directive: the model condenses the rendered transcript into
#: a concise summary the agent re-injects in place of the raw older turns.
SUMMARY_SYSTEM_PROMPT = (
    "You are condensing a longer conversation to free context budget. "
    "Read the transcript below and produce a concise running summary that a "
    "continuing agent can use in place of the original turns.\n\n"
    "Preserve, in order of importance:\n"
    "- the user's goals, requests, and any explicit constraints;\n"
    "- key decisions made and their rationale;\n"
    "- tool calls that were made and their outcomes (success or failure, "
    "what was learned);\n"
    "- open items, TODOs, and unresolved questions;\n"
    "- file paths, code references, identifiers, and other exact tokens the "
    "agent must reuse verbatim.\n\n"
    "Drop pleasantries, restatements, and reasoning that is not load-bearing "
    "for future turns. Do not add information that is not in the transcript. "
    "Write the summary as plain prose (optionally short bullet lists); it "
    "replaces the older turns, so it must stand on its own."
)


@dataclass(frozen=True)
class SummaryTarget:
    """The ``(provider, model)`` pair that performs a summarisation call.

    Resolved by :func:`resolve_summary_model`. A frozen value type: once
    selected it is passed unchanged to
    :func:`cothis.ai.base.get_provider` and
    :func:`cothis.ai.model_metadata.resolve_max_tokens` in slice C.
    """

    model: str
    provider: str


@dataclass(frozen=True)
class SummarisationRequest:
    """The ``amessages`` payload for one summarisation call (no execution).

    Carries the Anthropic-shaped block lists the provider consumes:

    * :attr:`system` — a single ``{"type": "text", ...}`` block carrying the
      condensation instruction. No ``cache_control``: this is a one-shot
      ephemeral call with no stable prefix to anchor a breakpoint on, and the
      Anthropic provider adds a breakpoint idempotently anyway.
    * :attr:`messages` — exactly one ``user`` turn whose single text block is
      the labelled transcript rendered from ``window``.
    * :attr:`max_tokens` — the output cap, carried through unchanged for the
      ``amessages`` call (slice C computes it via
      :func:`~cothis.ai.model_metadata.resolve_max_tokens`).

    Frozen: a built request is a snapshot, never mutated before send.
    """

    system: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    max_tokens: int


def _truncate(text: str, limit: int) -> str:
    """Return ``text`` capped at ``limit`` chars, with a tail marker if cut.

    A no-op when ``text`` already fits. The marker is appended *inside* the
    limit so a truncated value never exceeds ``limit`` characters.
    """
    if len(text) <= limit:
        return text
    keep = max(limit - len(_TRUNCATION_TAIL), 0)
    return text[:keep] + _TRUNCATION_TAIL


def _render_tool_input(input_val: Any) -> str:
    """Render a ``tool_use`` ``input`` payload compactly for the transcript.

    ``input`` is conventionally a dict (the tool's arguments). Rendered as
    deterministic JSON (``ensure_ascii=False`` to count real characters, as
    :func:`~cothis.ai.context_budget.estimate_input_tokens` does); a
    non-serialisable value falls back to ``str()`` rather than raise. Empty
    dict / ``None`` render as the empty string so an arg-less call reads as
    ``name()``. Capped at :data:`_TOOL_INPUT_CHAR_CAP`.
    """
    if input_val is None:
        return ""
    if isinstance(input_val, dict) and not input_val:
        return ""
    try:
        rendered = json.dumps(input_val, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        rendered = str(input_val)
    return _truncate(rendered, _TOOL_INPUT_CHAR_CAP)


def _render_tool_result_content(content: Any) -> str:
    """Render a ``tool_result`` ``content`` field to flat text.

    Anthropic shapes ``tool_result.content`` as either a string or a list of
    blocks (commonly ``text`` blocks); both are flattened to a single string.
    Anything else falls back to ``str()``. ``None`` renders empty. Capped at
    :data:`_TOOL_RESULT_CHAR_CAP` so one verbose tool cannot crowd out the
    rest of the transcript.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return _truncate(content, _TOOL_RESULT_CHAR_CAP)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # ``text`` is the load-bearing block; other block types
                # contribute their ``text`` field if they carry one.
                if block.get("type") == "text" or "text" in block:
                    parts.append(str(block.get("text", "")))
        return _truncate("\n".join(parts), _TOOL_RESULT_CHAR_CAP)
    return _truncate(str(content), _TOOL_RESULT_CHAR_CAP)


def _render_message(message: dict[str, Any]) -> str:
    """Render one Anthropic-shaped message dict to labelled transcript lines.

    Reads only ``role`` + ``content`` so this tolerates BOTH shapes that reach
    a summarisation call: the post-projection ``{role, content}`` form that
    :func:`cothis.agent._request_messages` emits AND the stored-with-metadata
    assistant form (which additionally carries ``id`` / ``model`` /
    ``stop_reason`` / ``usage``). Block rendering rules:

    * ``text`` block            -> ``"User: <text>"`` / ``"Assistant: <text>"``
      (prefix follows the message ``role``).
    * ``tool_use`` block        -> ``"Assistant called <name>(<input>) [id=<id>]"``.
    * ``tool_result`` block     -> ``"Tool result (<tool_use_id>): <content>"``
      (labeled ``Tool result`` regardless of the carrying role, since a
      ``tool_result`` rides in a ``user`` message but is semantically a tool
      reply).
    * ``thinking`` block        -> **skipped**. Internal reasoning is not
      load-bearing for the running summary and bloats the transcript.
    * unknown block types       -> skipped defensively.

    Returns the empty string when the message renders to nothing (e.g. a
    thinking-only message); the caller drops such empties before joining.
    Never raises on a malformed block — a ``KeyError`` here would crash the
    summarisation path, so every field is read tolerantly via ``.get``.
    """
    role = message.get("role", "")
    content = message.get("content", "")
    lines: list[str] = []
    if isinstance(content, str):
        if content:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)
    if not isinstance(content, list):
        return ""
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = str(block.get("text", ""))
            if text:
                prefix = "User" if role == "user" else "Assistant"
                lines.append(f"{prefix}: {text}")
        elif btype == "tool_use":
            name = str(block.get("name", "<unknown>"))
            rendered_input = _render_tool_input(block.get("input", {}))
            block_id = str(block.get("id", ""))
            lines.append(
                f"Assistant called {name}({rendered_input}) [id={block_id}]"
            )
        elif btype == "tool_result":
            tool_use_id = str(block.get("tool_use_id", ""))
            rendered = _render_tool_result_content(block.get("content", ""))
            lines.append(f"Tool result ({tool_use_id}): {rendered}")
        # ``thinking`` and any unknown block type: deliberately skipped.
    return "\n".join(lines)


def _render_window(
    window: list[dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    """Render the full window to one labelled transcript string.

    Joins the per-message renderings with newlines, dropping messages that
    render to nothing (e.g. thinking-only turns). When the joined transcript
    exceeds ``max_chars`` the OLDEST messages are dropped and replaced with
    :data:`_TRUNCATION_MARKER`, so the most recent — highest-value — turns
    survive verbatim. At least the single most recent message is always
    retained, even when one message alone exceeds the cap (a marker-only
    transcript would give the summariser nothing to condense). Returns the
    empty string for an empty / all-skipped window.
    """
    rendered = [r for r in (_render_message(m) for m in window) if r]
    if not rendered:
        return ""
    # +1 per message for the newline that joins it to the next.
    total = sum(len(r) + 1 for r in rendered)
    if total <= max_chars:
        return "\n".join(rendered)
    # Truncate oldest: walk tail-to-head, retaining messages until the next
    # (older) one would overflow. The first iteration always includes the
    # most recent message (``kept`` is empty, so the guard is False).
    kept: list[str] = []
    running = len(_TRUNCATION_MARKER) + 1  # marker + its joining newline
    for r in reversed(rendered):
        if kept and running + len(r) > max_chars:
            break
        kept.append(r)
        running += len(r) + 1
    kept.reverse()
    return _TRUNCATION_MARKER + "\n" + "\n".join(kept)


def resolve_summary_model(
    *,
    session_model: str,
    session_provider: str,
    override: str | None = None,
) -> SummaryTarget:
    """Resolve which ``(provider, model)`` performs a summarisation call.

    Precedence (highest first):

    1. ``override`` argument — explicit, programmatic (e.g. a future typer
       flag from slice C). Empty / whitespace-only is treated as unset.
    2. :data:`_SUMMARY_MODEL_ENV` (``COTHIS_SUMMARY_MODEL``) — the
       operator-knob default, with the same parse rules as ``override``.
    3. The session pair ``(session_provider, session_model)`` — summarise
       with the same model that runs the turn. The safe default: no extra
       credentials, no cold-start latency from a different provider.

    Spec parse (applied to whichever of override / env wins): split on the
    FIRST ``/`` — ``"provider/model"`` yields ``(provider, model)``; a bare
    ``"model"`` (no slash) inherits ``session_provider``. Whitespace is
    trimmed. A spec that parses to an empty model (e.g. ``"/"`` or
    ``"provider/"``) falls through to the next precedence level rather than
    selecting an empty model.

    Pure and deterministic. The env var is read ONLY when ``override`` does
    not supply a usable value: an explicit ``override=""`` or whitespace-only
    override parses to nothing and DOES fall through to the env var, then to
    the session pair if the env var is also unset.
    """
    if override is not None:
        target = _spec_to_target(override, session_provider=session_provider)
        if target is not None:
            return target
    env_spec = os.environ.get(_SUMMARY_MODEL_ENV)
    if env_spec is not None:
        target = _spec_to_target(env_spec, session_provider=session_provider)
        if target is not None:
            return target
    return SummaryTarget(model=session_model, provider=session_provider)


def _spec_to_target(
    spec: str,
    *,
    session_provider: str,
) -> SummaryTarget | None:
    """Parse a model spec into a :class:`SummaryTarget`, or ``None`` if unset.

    ``None`` means "no usable value here — fall through to the next
    precedence level". A spec is unusable when it is empty / whitespace-only,
    or when it parses to an empty model (a stray slash). A bare model
    (no slash) inherits ``session_provider``; ``"provider/model"`` keeps the
    parsed provider even when the session provider differs.
    """
    text = spec.strip()
    if not text:
        return None
    if "/" in text:
        provider_part, _, model_part = text.partition("/")
        provider = provider_part.strip()
        model = model_part.strip()
    else:
        provider = ""
        model = text
    if not model:
        # e.g. "/" or "provider/" — would select an empty model. Skip.
        return None
    return SummaryTarget(
        model=model,
        provider=provider if provider else session_provider,
    )


def build_summarisation_request(
    *,
    window: list[dict[str, Any]],
    max_tokens: int,
    system_text: str | None = None,
    max_window_chars: int = _DEFAULT_MAX_WINDOW_CHARS,
) -> SummarisationRequest:
    """Build the ``amessages`` payload for one summarisation call.

    Inline-renders ``window`` to a single user text message containing a
    labelled transcript, with the condensation instruction as a single system
    text block. The window is the Anthropic ``{role, content}`` shape that
    :func:`cothis.agent._request_messages` produces post-projection; the
    renderer reads only ``role`` + ``content``, so the stored-with-metadata
    assistant shape is tolerated too.

    Design choice — inline-render rather than pass the window through:

    * **No tool schema needed.** Anthropic validates ``tool_use`` blocks
      against the ``tools`` param, so pass-through would force shipping tool
      schemas to the summariser. ``tools`` stays ``None``.
    * **Provider-agnostic.** The same labelled transcript drives an OpenAI /
      Google translation path, not just the native Anthropic one.
    * **Single user turn.** One system block + one user block is the cheapest
      valid request shape.

    Parameters:

    * ``window`` — older turns to condense (post-projection Anthropic shape).
    * ``max_tokens`` — output cap, carried through unchanged for the
      ``amessages`` call. Slice C computes it via
      :func:`~cothis.ai.model_metadata.resolve_max_tokens`.
    * ``system_text`` — replaces the default :data:`SUMMARY_SYSTEM_PROMPT`
      verbatim when set (testability + slice-C customisation). ``None`` keeps
      the default.
    * ``max_window_chars`` — overall transcript cap; oldest turns are
      truncated past it (see :func:`_render_window`).

    Tolerates an empty window: returns a well-shaped request whose transcript
    text is the empty string, rather than raising.
    """
    instruction = system_text if system_text is not None else SUMMARY_SYSTEM_PROMPT
    transcript = _render_window(window, max_chars=max_window_chars)
    system: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": transcript}]}
    ]
    return SummarisationRequest(
        system=system,
        messages=messages,
        max_tokens=max_tokens,
    )


# =============================================================================
# Slice B — eviction policy (the "which turns to compact" decision).
# =============================================================================
#
# A PURE decision function. Given the live ``_messages`` shape + a
# :class:`~cothis.ai.context_budget.ContextBudget`, it selects WHICH older
# turns form the summarisation window (the exact shape slice A's
# :func:`build_summarisation_request` consumes) and which stay verbatim. It
# never executes the summarisation call, never mutates the input list, and
# never imports the higher-layer :mod:`cothis.agent` module (would be a
# circular import — cothis-ai is below cothis-agent). Duck-typed on
# ``list[dict]`` and reuses :func:`~cothis.ai.context_budget.estimate_input_tokens`
# for sizing; no new tokeniser dependency.
#
# Three load-bearing invariants the decision preserves:
#
# 1. **Pressure gating.** Eviction only fires at HIGH / CRITICAL pressure
#    (MEDIUM is "plan, not act"; NONE / LOW / unknown budget -> no-op).
# 2. **Tool-pair closure.** The cut never splits a ``tool_use`` /
#    ``tool_result`` pair — Anthropic rejects a dangling reference with
#    HTTP 400, so the retained tail must be a valid standalone conversation.
# 3. **Retention floor.** The last ``min_retained_turns`` turn-groups stay
#    verbatim in the tail, even under CRITICAL pressure.


@dataclass(frozen=True)
class EvictionDecision:
    """The result of one :func:`plan_eviction` call — what to compact vs keep.

    A pure value type: ``plan_eviction`` READS ``messages`` and returns a
    decision; it does not mutate the live agent state (slice C applies the
    decision by feeding :attr:`window` to
    :func:`build_summarisation_request` and splicing :attr:`retained`).

    Attributes:

    * :attr:`window` — the older turns to compact, as a contiguous PREFIX of
      ``messages`` in original order. Anthropic-shaped (the renderer in slice
      A tolerates both the projected ``{role, content}`` form and the
      stored-with-metadata assistant form). Empty when no eviction fires.
    * :attr:`retained` — the active tail, kept verbatim, ``messages[len(window):]``.
      The same dict objects as the input (not copies); identity is preserved.
    * :attr:`pressure` — snapshot of the :class:`PressureLevel` that drove the
      decision (``None`` when the budget was unknown). Observability only.
    * :attr:`evicted_token_estimate` — :func:`estimate_input_tokens` over
      :attr:`window`; ``0`` when no eviction fires. A deterministic heuristic
      (char/4), NOT a real token count — slice C re-checks the real budget
      after the summary lands.
    * :attr:`reason` — short ``"<outcome>:<cause>"`` tag for logs/telemetry,
      e.g. ``"evicted:high-pressure"``, ``"no-eviction:low-pressure"``,
      ``"no-eviction:below-floor"``, ``"no-eviction:no-safe-cut"``.

    Frozen: a decision is a snapshot; callers must not mutate it. The
    ``window`` / ``retained`` lists reference the same dict objects as the
    input (slice B decides, it does not copy) — slice C is the sole mutator.
    """

    window: list[dict[str, Any]]
    retained: list[dict[str, Any]]
    pressure: PressureLevel | None
    evicted_token_estimate: int
    reason: str


def _message_content_blocks(message: dict[str, Any]) -> list[Any]:
    """Return the block list a message carries, or ``[]`` if it has none.

    A bare-string ``content`` has no structured blocks (and therefore no
    ``tool_use`` / ``tool_result``); a list content yields the list; anything
    else (``None``, unexpected types) yields ``[]``. Never raises — every
    slice-B helper reads the message shape tolerantly.
    """
    content = message.get("content")
    if isinstance(content, list):
        return content
    return []


def _is_tool_flow_continuation(message: dict[str, Any]) -> bool:
    """``True`` when a ``user`` message is purely a ``tool_result`` reply.

    A ``tool_result`` rides in a ``user`` message but is semantically a tool
    reply, NOT a fresh user input. Such messages are NOT turn-group starts
    (they continue the ongoing tool flow of the previous assistant turn). A
    user message carrying any non-``tool_result`` block (text, or
    tool_result + text) counts as a fresh user input and DOES start a group.

    Mirrors the inverse of :func:`cothis.agent._footer_target_idx`'s
    "user-typed message" detection.
    """
    if message.get("role") != "user":
        return False
    blocks = _message_content_blocks(message)
    if not blocks:
        return False
    return all(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks
    )


def _safe_cut_indices(messages: list[dict[str, Any]]) -> list[int]:
    """Indices ``k`` where ``messages[:k]`` / ``messages[k:]`` is a valid split.

    A cut at ``k`` is **safe** iff no ``tool_use`` / ``tool_result`` pair
    straddles it: every ``tool_use`` id appearing in ``messages[:k]`` has its
    matching ``tool_result`` also in ``messages[:k]``, AND every
    ``tool_result`` in ``messages[k:]`` has its ``tool_use`` in
    ``messages[k:]``. Equivalently — the set of "open" (unanswered)
    ``tool_use`` ids is empty after processing ``messages[:k]``.

    Returns the ascending list of safe ``k`` in ``[0, len(messages)]``.
    ``0`` (empty prefix) and ``len(messages)`` (when the whole list closes
    cleanly) are always included when safe. Because :mod:`cothis.agent`
    enforces strict user/assistant alternation at storage time, ANY
    contiguous split preserves role alternation in both halves — so the
    tool-pair closure checked here is the ONLY hard API constraint on a cut.

    Never raises: malformed blocks are skipped (a ``tool_use`` without an
    ``id``, or a ``tool_result`` without a ``tool_use_id``, simply
    contributes nothing to the open-id set).
    """
    open_ids: set[str] = set()
    safe = [0]
    for k in range(1, len(messages) + 1):
        for block in _message_content_blocks(messages[k - 1]):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                bid = block.get("id")
                if bid is not None:
                    open_ids.add(str(bid))
            elif btype == "tool_result":
                tid = block.get("tool_use_id")
                if tid is not None:
                    open_ids.discard(str(tid))
        if not open_ids:
            safe.append(k)
    return safe


def _group_start_indices(messages: list[dict[str, Any]]) -> list[int]:
    """Ascending indices where a new turn-group begins.

    A turn-group is one conversational turn: a fresh user message plus the
    assistant reply and any ``tool_use`` / ``tool_result`` closure that
    follows it. Group starts are:

    * index ``0`` (always — the conversation's first message opens group 0,
      whatever role it carries), and
    * every subsequent ``user`` message that is NOT a pure ``tool_result``
      reply (i.e. a fresh user input, not a tool-flow continuation — see
      :func:`_is_tool_flow_continuation`).

    The retention floor is counted in turn-groups
    (:data:`_DEFAULT_MIN_RETAINED_TURNS`); the floor boundary
    (:func:`_floor_boundary_index`) is the start of the first retained group.
    """
    starts = []
    for i, message in enumerate(messages):
        if i == 0:
            starts.append(i)
            continue
        if message.get("role") == "user" and not _is_tool_flow_continuation(message):
            starts.append(i)
    return starts


def _floor_boundary_index(
    messages: list[dict[str, Any]], min_retained_turns: int
) -> int:
    """Index of the first RETAINED message under the turn-group floor.

    The last ``min_retained_turns`` turn-groups are always retained, so the
    window may only draw from ``messages[:floor_boundary]``. Returns ``0``
    when the conversation has ``<= min_retained_turns`` groups (nothing can
    be evicted without breaking the floor) — the caller treats that as the
    ``no-eviction:below-floor`` no-op.
    """
    starts = _group_start_indices(messages)
    group_count = len(starts)
    if group_count <= min_retained_turns:
        return 0
    return starts[group_count - min_retained_turns]


def _no_eviction(
    messages: list[dict[str, Any]],
    pressure: PressureLevel | None,
    reason: str,
) -> EvictionDecision:
    """Build a no-op :class:`EvictionDecision` (empty window, full retained tail)."""
    return EvictionDecision(
        window=[],
        retained=list(messages),
        pressure=pressure,
        evicted_token_estimate=0,
        reason=reason,
    )


def plan_eviction(
    *,
    messages: list[dict[str, Any]],
    budget: ContextBudget,
    min_retained_turns: int = _DEFAULT_MIN_RETAINED_TURNS,
    high_target_ratio: float = _DEFAULT_HIGH_TARGET_RATIO,
    critical_floor_ratio: float = _DEFAULT_CRITICAL_FLOOR_RATIO,
    summary_overhead_tokens: int = 0,
) -> EvictionDecision:
    """Decide which older turns to compact vs retain (pure; slice B).

    Selects a contiguous PREFIX of ``messages`` as the summarisation window
    and leaves the rest as the verbatim retained tail, driven by
    ``budget.pressure``. The window is fed straight into
    :func:`build_summarisation_request` by slice C; this function only
    DECIDES — it never executes the call, mutates ``messages``, or imports
    the agent loop.

    Pressure -> aggressiveness mapping (the gate):

    * budget unknown (``is_known`` False / ``pressure`` None) -> no eviction
      (``no-eviction:unknown-budget``). Honest: no signal to act on.
    * pressure in {NONE, LOW, MEDIUM} -> no eviction
      (``no-eviction:low-pressure``). MEDIUM is "plan, not act".
    * pressure HIGH -> evict oldest turn-groups from the front until the
      retained tail's estimated token count drops to
      ``high_target_ratio * capacity`` (or the floor is reached); the
      SMALLEST SAFE cut meeting the target wins (minimal eviction); if no
      safe cut meets it, evict maximally within the floor.
    * pressure CRITICAL -> target a lower ratio
      (``critical_floor_ratio * capacity``), again taking the smallest safe
      cut that meets it, and falling back to the largest safe cut (down to
      the floor) if none does. Monotonic: for a fixed input, CRITICAL evicts
      a token-superset of HIGH.

    The cut respects three invariants (see :func:`_safe_cut_indices`):

    1. **Tool-pair closure** — the cut is a safe index, so no
       ``tool_use`` / ``tool_result`` pair is split. The retained tail is a
       valid standalone Anthropic conversation (no dangling ids).
    2. **Retention floor** — the cut never passes
       :func:`_floor_boundary_index`, so the last ``min_retained_turns``
       turn-groups stay verbatim in the tail.
    3. **Role alternation** — inherited for free from the storage layer's
       strict alternation (any contiguous split of an alternating list is
       alternating); the safe-cut walk adds the tool-pair guarantee on top.

    Parameters:

    * ``messages`` — the live ``_messages`` shape (assistant dicts may carry
      ``id`` / ``model`` / ``stop_reason`` / ``usage`` metadata; only
      ``role`` + ``content`` are read). Duck-typed — no :mod:`cothis.agent`
      import.
    * ``budget`` — the context-window pressure signal.
    * ``min_retained_turns`` — retention floor in turn-groups (default 4).
    * ``high_target_ratio`` / ``critical_floor_ratio`` — post-eviction ratio
      targets (defaults 0.75 / 0.50).
    * ``summary_overhead_tokens`` — conservative reserve for the summary's
      own size, subtracted from the HIGH retained-token target so the
      post-compaction context has room for both tail and summary. Defaults
      ``0`` (slice C owns the post-summary recheck).

    Determinism: a pure function of ``(messages, budget, params)`` — no env,
    no random, no I/O. Identical input yields an equal :class:`EvictionDecision`
    (frozen dataclass ``__eq__``). The window references the SAME dict
    objects as ``messages`` (not copies); ``retained`` is
    ``messages[len(window):]``.
    """
    # Clamp a nonsensical floor: ``min_retained_turns <= 0`` would index out
    # of range in :func:`_floor_boundary_index` and semantically permit
    # compacting the active turn. Floor it at 1 so the most recent
    # turn-group is always retained.
    min_retained_turns = max(min_retained_turns, 1)
    pressure = budget.pressure

    # Gate 1: no usable signal — capacity or usage unknown. Be honest rather
    # than fabricate an eviction target from a heuristic alone.
    if not budget.is_known or pressure is None:
        return _no_eviction(messages, pressure, "no-eviction:unknown-budget")

    # Gate 2: pressure not yet at the act threshold. MEDIUM is "plan".
    if pressure in (PressureLevel.NONE, PressureLevel.LOW, PressureLevel.MEDIUM):
        return _no_eviction(messages, pressure, "no-eviction:low-pressure")

    # Gate 3: retention floor — short conversations are a no-op. The last
    # ``min_retained_turns`` turn-groups must stay verbatim; if the whole
    # conversation fits in the floor there is nothing safe to evict.
    floor_boundary = _floor_boundary_index(messages, min_retained_turns)
    if floor_boundary <= 0:
        return _no_eviction(messages, pressure, "no-eviction:below-floor")

    # Cut candidates: safe cuts in (0, floor_boundary]. These are the only
    # indices that both close every tool pair AND respect the floor.
    candidates = [
        k for k in _safe_cut_indices(messages) if 0 < k <= floor_boundary
    ]
    if not candidates:
        # floor_boundary > 0 but no safe cut exists in the evictable range
        # (e.g. the very first message opens a tool_use that never closes).
        # Refuse to emit a window that would dangle a reference.
        return _no_eviction(messages, pressure, "no-eviction:no-safe-cut")

    capacity = budget.capacity_tokens
    # ``capacity`` is non-None here (is_known is True), but guard defensively
    # so a malformed budget never crashes the decision path.
    if capacity is None or capacity <= 0:
        return _no_eviction(messages, pressure, "no-eviction:unknown-budget")

    if pressure == PressureLevel.CRITICAL:
        # CRITICAL: target a lower post-eviction ratio than HIGH (reclaim
        # more), walking smallest-first; fall back to the largest safe cut
        # (down to the floor) if no safe cut meets the target. Because
        # ``critical_floor_ratio`` < ``high_target_ratio``, CRITICAL evicts a
        # token-superset of HIGH for the same input (monotonicity).
        target_tokens = max(0, int(critical_floor_ratio * capacity))
        cut = 0
        for candidate in candidates:
            if estimate_input_tokens(messages[candidate:]) <= target_tokens:
                cut = candidate
                break
        if cut == 0:
            # No safe cut meets the CRITICAL target — evict maximally within
            # the floor (the next turn would overflow).
            cut = candidates[-1]
            reason = "evicted:critical-pressure-maximal"
        else:
            reason = "evicted:critical-pressure"
    else:
        # HIGH: minimal eviction that meets the target. Walk smallest-first
        # and stop at the first cut whose retained estimate fits the target.
        # The summary's own size is reserved out of the target so the
        # post-compaction context has room for tail + summary.
        target_tokens = max(
            0, int(high_target_ratio * capacity) - summary_overhead_tokens
        )
        cut = 0
        for candidate in candidates:
            if estimate_input_tokens(messages[candidate:]) <= target_tokens:
                cut = candidate
                break
        if cut == 0:
            # No safe cut meets the target — evict maximally within the
            # floor anyway (pressure is HIGH; refusing to act would leave
            # the next turn to overflow).
            cut = candidates[-1]
            reason = "evicted:high-pressure-maximal"
        else:
            reason = "evicted:high-pressure"

    window = messages[:cut]
    return EvictionDecision(
        window=list(window),
        retained=list(messages[cut:]),
        pressure=pressure,
        evicted_token_estimate=estimate_input_tokens(window),
        reason=reason,
    )
