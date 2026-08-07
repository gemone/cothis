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
