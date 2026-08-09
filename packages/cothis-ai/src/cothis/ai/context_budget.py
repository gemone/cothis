"""Context-window budget primitive + rolling pressure signal.

A pure-data signal describing how full the model's context window is,
assembled from two sources that already live in the agent's state:

1. The model's advertised ``contextWindow`` (input cap) — published by
   :func:`cothis.ai.model_metadata.model_info` and read on demand.
2. The provider's own ``usage`` block — the cheapest, most accurate
   "how full was the window on the last call" measurement, free of any
   estimation error.

This module is **instrumentation only**: it computes and exposes the
signal. Nothing here compacts, evicts, summarises, or otherwise reacts
to it — that behaviour is explicitly deferred. Every derived field
tolerates a ``None`` capacity (the honest advertisement for models
missing from the metadata) by collapsing to ``None`` rather than
fabricating a number.

No I/O, no SDK imports, no new third-party dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Char-based token estimate fallback (see ``estimate_input_tokens``). The
# codebase already references the "~4 chars per token" / "~1.5x token
# cost under common BPE tokenizers" rule of thumb (``tools/format.py``);
# this is the same family of estimate, bounded to roughly ±20% error,
# zero-dependency, and deterministic.
_CHARS_PER_TOKEN = 4


class PressureLevel(str, Enum):
    """Coarse pressure bucket derived from :attr:`ContextBudget.ratio`.

    A ``str`` mixin so the value serialises cleanly to JSON / logs without
    a custom encoder. ``None`` (carried on :attr:`ContextBudget.pressure`,
    not here) represents "ratio unknown" — the capacity is unknown, so no
    bucket is honest.
    """

    NONE = "none"        # < 0.50 — plenty of room
    LOW = "low"          # [0.50, 0.75) — starting to fill
    MEDIUM = "medium"    # [0.75, 0.90) — compaction should be planned
    HIGH = "high"        # [0.90, 1.00) — compact before next turn
    CRITICAL = "critical"  # >= 1.00 — at/over the cap


@dataclass(frozen=True)
class ContextBudget:
    """Pure-data snapshot of context-window pressure.

    Computed on demand from already-stored state (the last assistant
    ``usage`` + ``model_info``'s ``contextWindow``), so it always reflects
    the latest conversation state. Frozen: derived fields are computed
    once at construction and never recomputed by consumers.

    All fields tolerantly take ``None``:

    * :attr:`used_tokens` is ``None`` only when no observed usage exists
      AND the heuristic estimate is unavailable (which currently means
      "no messages at all and no usage" — the heuristic returns 0 for an
      empty list, so ``used_tokens`` is ``0``, not ``None``, in that
      case). On the build path, ``None`` is reserved for "we genuinely could
      not measure".
    * :attr:`capacity_tokens` is ``None`` when the model is missing from
      the bundled metadata (``model_info`` reports honestly, never
      invents).
    * :attr:`available_tokens`, :attr:`ratio`, :attr:`pressure` are
      ``None`` whenever *either* of the above is ``None`` — no partial
      arithmetic, no fabricated signal.
    """

    used_tokens: int | None
    capacity_tokens: int | None
    available_tokens: int | None
    ratio: float | None
    pressure: PressureLevel | None

    @property
    def is_known(self) -> bool:
        """``True`` when :attr:`ratio` (and therefore :attr:`pressure`) is known.

        The signal is "useful" only when capacity is known; consumers
        gate on this rather than re-checking ``ratio is not None``.
        """
        return self.ratio is not None


def total_input_tokens_from_usage(
    usage: dict[str, Any] | None,
) -> int | None:
    """Return the true context size for one provider ``usage`` block.

    Provider-agnostic: Anthropic's ``Usage`` breaks cached tokens out
    (``input_tokens`` is the *non-cached* portion;
    ``cache_creation_input_tokens`` + ``cache_read_input_tokens`` are
    the cached portions), while OpenAI / Google / OpenRouter map their
    totals into ``input_tokens`` and leave the cache fields absent. The
    true context size for the turn is, uniformly::

        total = input_tokens
              + (cache_creation_input_tokens or 0)
              + (cache_read_input_tokens     or 0)

    Absent cache fields contribute 0; for OpenAI/Google this collapses
    to plain ``input_tokens``. Returns ``None`` when ``input_tokens`` is
    absent (no turn completed yet, or a provider that reported nothing)
    — never invents a number.
    """
    if usage is None:
        return None
    base = usage.get("input_tokens")
    if not isinstance(base, int) or base < 0:
        return None
    # Cache fields are optional (absent on OpenAI/Google → contribute 0).
    # When present they must be non-negative ints — the same rule as
    # ``base``. A malformed value (a provider divergence or a planted test
    # dict) makes the whole total untrustworthy, so report ``None`` rather
    # than crash the signal or fabricate a partial total.
    cache_create = usage.get("cache_creation_input_tokens")
    cache_read = usage.get("cache_read_input_tokens")
    for field in (cache_create, cache_read):
        if field is not None and (not isinstance(field, int) or field < 0):
            return None
    return base + (cache_create or 0) + (cache_read or 0)


def _system_char_count(system: str | list[dict[str, Any]] | None) -> int:
    """Deterministic char count for the system prompt in the shape the
    agent holds it. A ``str`` persona is counted as-is; a pre-built
    block list (the shape after assembly — each block
    ``{type, text, cache_control?}``) is serialised whole. The
    ``cache_control`` directive is a few chars of control metadata,
    negligible against the text bodies. ``None`` / empty → 0.
    """
    if not system:
        return 0
    if isinstance(system, str):
        return len(system)
    return len(json.dumps(system, ensure_ascii=False))


def _tools_char_count(tools: list[dict[str, Any]] | None) -> int:
    """Deterministic char count for the tool/schema definitions in the
    Anthropic shape (``{name, description, input_schema}``) ``amessages``
    sends. ``None`` / empty → 0.
    """
    if not tools:
        return 0
    return len(json.dumps(tools, ensure_ascii=False))


def estimate_input_tokens(
    messages: list[dict[str, Any]],
    *,
    system: str | list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Heuristic char/4 token estimate over what the provider bills.

    Used only as a fallback when no observed ``usage`` is available
    (pre-first-turn, or a response that lacked usage). Counts every
    component the provider bills against the input cap on the first
    turn — the message bodies, the system prompt, and the tool/schema
    definitions — so the pre-first-turn signal is honest instead of
    understating pressure by a large constant (the system prompt +
    tool schemas are a non-trivial fixed cost for an agent).

    Walks each message's ``{role, content}`` projection plus the system
    prompt and tool schemas, serialising each deterministically, and
    divides the total character count by :data:`_CHARS_PER_TOKEN`.

    Properties: zero-dependency, deterministic, monotonic in input size,
    bounded to roughly ±20% error versus a real BPE tokeniser.

    ``system`` and ``tools`` default to ``None``. The context-budget
    build path passes the agent's current system prompt and tool schemas
    so the pre-first-turn estimate reflects the full input. The
    compaction path calls with messages only: it estimates a *delta*
    over a retained message window, and the (constant) system/tools
    overhead cancels out of that decision, so excluding them keeps the
    eviction math honest.

    Not on any hot path: only called when ``_last_observed_input_tokens``
    returns ``None``.
    """
    total_chars = _system_char_count(system) + _tools_char_count(tools)
    for m in messages:
        # Serialise only {role, content} — the stored assistant dicts
        # carry extra metadata (id/model/stop_reason/usage) that the
        # provider did NOT bill against the input cap, so excluding them
        # keeps the estimate honest. ``ensure_ascii=False`` counts real
        # characters (a CJK codepoint is ~1 token, not the 6 chars of
        # its ``\uXXXX`` escape that ``ensure_ascii=True`` would emit).
        projection = {"role": m.get("role", ""), "content": m.get("content", "")}
        total_chars += len(json.dumps(projection, ensure_ascii=False))
    return total_chars // _CHARS_PER_TOKEN


def pressure_from_ratio(ratio: float | None) -> PressureLevel | None:
    """Map a ``ratio`` to a :class:`PressureLevel` bucket; ``None``-safe.

    Thresholds (one tunable source; the deferred consumer may override):

    ==================  ===========  ==============================
    ratio range         level        intent
    ==================  ===========  ==============================
    ``< 0.50``          NONE         plenty of room
    ``[0.50, 0.75)``    LOW          starting to fill
    ``[0.75, 0.90)``    MEDIUM       compaction should be planned
    ``[0.90, 1.00)``    HIGH         compact before next turn
    ``>= 1.00``         CRITICAL     at/over the cap
    ==================  ===========  ==============================
    """
    if ratio is None:
        return None
    if ratio < 0.50:
        return PressureLevel.NONE
    if ratio < 0.75:
        return PressureLevel.LOW
    if ratio < 0.90:
        return PressureLevel.MEDIUM
    if ratio < 1.00:
        return PressureLevel.HIGH
    return PressureLevel.CRITICAL


def build_context_budget(
    *,
    used: int | None,
    capacity: int | None,
) -> ContextBudget:
    """Single assembly point for a :class:`ContextBudget`.

    Computes :attr:`~ContextBudget.available_tokens`,
    :attr:`~ContextBudget.ratio`, and
    :attr:`~ContextBudget.pressure` from the two raw inputs. Tolerates
    either input being ``None``: every derived field collapses to
    ``None`` rather than performing partial arithmetic. ``used_tokens``
    and ``capacity_tokens`` always preserve their input values
    (including ``None``).
    """
    available: int | None
    ratio: float | None
    pressure: PressureLevel | None
    if used is None or capacity is None:
        available = None
        ratio = None
        pressure = None
    else:
        available = capacity - used
        # ``capacity`` is read from ``model_info``, which only returns
        # positive ints (or None). Guard the division anyway so a
        # malformed positive-zero capacity degrades to "unknown" instead
        # of raising — the signal must never crash the agent loop.
        ratio = used / capacity if capacity > 0 else None
        pressure = pressure_from_ratio(ratio)
    return ContextBudget(
        used_tokens=used,
        capacity_tokens=capacity,
        available_tokens=available,
        ratio=ratio,
        pressure=pressure,
    )
