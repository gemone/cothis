"""Tests for ``cothis.ai.context_budget`` — the pressure primitive.

The primitive is pure data + pure functions: no I/O, no SDK imports, no
third-party dependency. These tests pin every branch of:

* :func:`build_context_budget` — derived-field computation + the
  ``None``-tolerant graceful-degradation path (the contract the deferred
  auto-compaction consumer will gate on).
* :func:`pressure_from_ratio` — the threshold table, one assertion per
  boundary, plus ``None`` safety.
* :func:`total_input_tokens_from_usage` — the provider-agnostic cache
  sum (Anthropic shape, OpenAI shape, missing/absent inputs).
* :func:`estimate_input_tokens` — the heuristic fallback (empty,
  monotonic, deterministic, metadata-key-safe).
* :attr:`ContextBudget.is_known` follows :attr:`~ContextBudget.ratio`.
"""

from __future__ import annotations

from typing import Any

import pytest

from cothis.ai.context_budget import (
    ContextBudget,
    PressureLevel,
    build_context_budget,
    estimate_input_tokens,
    pressure_from_ratio,
    total_input_tokens_from_usage,
)

# --- build_context_budget ---------------------------------------------------


def test_build_both_known_derives_all_fields() -> None:
    # 75 % of 131072 lands on the MEDIUM boundary exactly.
    budget = build_context_budget(used=98304, capacity=131072)
    assert budget.used_tokens == 98304
    assert budget.capacity_tokens == 131072
    assert budget.available_tokens == 32768
    assert budget.ratio == pytest.approx(0.75)
    assert budget.pressure == PressureLevel.MEDIUM
    assert budget.is_known


def test_build_capacity_none_collapses_derived_to_none() -> None:
    # Unknown model → capacity None. used is preserved; every derived
    # field is None (no partial arithmetic, no fabricated signal).
    budget = build_context_budget(used=1000, capacity=None)
    assert budget.used_tokens == 1000
    assert budget.capacity_tokens is None
    assert budget.available_tokens is None
    assert budget.ratio is None
    assert budget.pressure is None
    assert not budget.is_known


def test_build_used_none_collapses_derived_to_none() -> None:
    # Pre-first-turn with empty messages → used None. Same collapse.
    budget = build_context_budget(used=None, capacity=200000)
    assert budget.used_tokens is None
    assert budget.capacity_tokens == 200000
    assert budget.available_tokens is None
    assert budget.ratio is None
    assert budget.pressure is None
    assert not budget.is_known


def test_build_both_none_all_fields_none() -> None:
    budget = build_context_budget(used=None, capacity=None)
    assert budget.used_tokens is None
    assert budget.capacity_tokens is None
    assert budget.available_tokens is None
    assert budget.ratio is None
    assert budget.pressure is None
    assert not budget.is_known


def test_build_at_capacity_is_critical_and_available_zero() -> None:
    # ratio == 1.0 → CRITICAL; available_tokens is exactly 0 (not None).
    budget = build_context_budget(used=100, capacity=100)
    assert budget.ratio == pytest.approx(1.0)
    assert budget.pressure == PressureLevel.CRITICAL
    assert budget.available_tokens == 0


def test_build_over_capacity_is_critical_and_available_negative() -> None:
    # ratio > 1.0 → CRITICAL; available_tokens is negative (over the cap).
    budget = build_context_budget(used=150, capacity=100)
    assert budget.ratio == pytest.approx(1.5)
    assert budget.pressure == PressureLevel.CRITICAL
    assert budget.available_tokens == -50


def test_build_context_budget_is_frozen() -> None:
    # Frozen dataclass: the signal is a snapshot, never mutated in place.
    # ``setattr`` (rather than attribute assignment) bypasses the static
    # read-only-field check while still triggering the runtime
    # ``FrozenInstanceError`` the test asserts.
    budget = build_context_budget(used=1, capacity=10)
    with pytest.raises(Exception):  # noqa: B017 — frozen-attr exception type varies
        setattr(budget, "used_tokens", 2)


# --- pressure_from_ratio ----------------------------------------------------


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (0.0, PressureLevel.NONE),
        (0.49, PressureLevel.NONE),
        (0.50, PressureLevel.LOW),
        (0.74, PressureLevel.LOW),
        (0.75, PressureLevel.MEDIUM),
        (0.89, PressureLevel.MEDIUM),
        (0.90, PressureLevel.HIGH),
        (0.99, PressureLevel.HIGH),
        (1.00, PressureLevel.CRITICAL),
        (1.50, PressureLevel.CRITICAL),
    ],
)
def test_pressure_from_ratio_thresholds(ratio: float, expected: PressureLevel) -> None:
    # One assertion per boundary: the lower edge of each bucket is
    # inclusive; the upper edge is exclusive (NONE<0.50, LOW<0.75, ...).
    assert pressure_from_ratio(ratio) == expected


def test_pressure_from_ratio_none_is_none() -> None:
    # Unknown capacity → unknown ratio → unknown pressure. Never a default.
    assert pressure_from_ratio(None) is None


# --- total_input_tokens_from_usage ------------------------------------------


def test_total_input_tokens_anthropic_shape_sums_cache_fields() -> None:
    # Anthropic breaks cached tokens out: input_tokens is the non-cached
    # portion; the cache fields are the cached portions. The true context
    # size is the sum.
    usage = {
        "input_tokens": 1000,
        "cache_creation_input_tokens": 5000,
        "cache_read_input_tokens": 94000,
        "output_tokens": 200,
    }
    assert total_input_tokens_from_usage(usage) == 100000


def test_total_input_tokens_openai_shape_is_just_input_tokens() -> None:
    # OpenAI / Google / OpenRouter map totals into input_tokens and leave
    # the cache fields absent; absent fields contribute 0.
    usage = {"input_tokens": 98304, "output_tokens": 500}
    assert total_input_tokens_from_usage(usage) == 98304


def test_total_input_tokens_none_usage_returns_none() -> None:
    # No turn completed yet (assistant message never stored, or stored
    # with usage=None). Never invents a number.
    assert total_input_tokens_from_usage(None) is None


def test_total_input_tokens_empty_dict_returns_none() -> None:
    # A usage block that reported nothing useful.
    assert total_input_tokens_from_usage({}) is None


def test_total_input_tokens_missing_input_tokens_returns_none() -> None:
    # Cache fields present but input_tokens absent — provider divergence.
    # input_tokens is the load-bearing field; without it we don't guess.
    usage = {"cache_read_input_tokens": 5000, "output_tokens": 10}
    assert total_input_tokens_from_usage(usage) is None


def test_total_input_tokens_input_only_no_cache_fields() -> None:
    # input_tokens set, cache fields absent — collapses to input_tokens.
    assert total_input_tokens_from_usage({"input_tokens": 42}) == 42


def test_total_input_tokens_malformed_cache_field_returns_none() -> None:
    # A non-int cache field (provider divergence / a planted test dict)
    # makes the whole total untrustworthy — the signal must report None
    # rather than raise ValueError on int("abc") or fabricate a partial sum.
    assert (
        total_input_tokens_from_usage(
            {"input_tokens": 100, "cache_read_input_tokens": "abc"}
        )
        is None
    )
    # A negative cache field is equally untrustworthy.
    assert (
        total_input_tokens_from_usage(
            {"input_tokens": 100, "cache_creation_input_tokens": -5}
        )
        is None
    )


# --- estimate_input_tokens --------------------------------------------------


def test_estimate_empty_messages_returns_zero() -> None:
    # Pre-first-turn with no messages: the estimate is honestly 0.
    assert estimate_input_tokens([]) == 0


def test_estimate_is_monotonic_in_input_size() -> None:
    # Longer text → larger estimate (the heuristic's core contract).
    short = [{"role": "user", "content": "hi"}]
    long = [{"role": "user", "content": "x" * 400}]
    assert estimate_input_tokens(short) < estimate_input_tokens(long)


def test_estimate_is_deterministic_across_calls() -> None:
    # Same input → same output, every call (no rng, no time-of-day drift).
    messages = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi back"}]},
    ]
    first = estimate_input_tokens(messages)
    second = estimate_input_tokens(messages)
    third = estimate_input_tokens(messages)
    assert first == second == third
    assert first > 0


def test_estimate_ignores_metadata_keys() -> None:
    # Stored assistant dicts carry ``id``/``model``/``stop_reason``/
    # ``usage`` alongside ``role``/``content`` — none of those are billed
    # against the input cap, so the estimate must ignore them. Two
    # messages identical in {role, content} but differing in metadata
    # produce the same estimate.
    base: dict[str, Any] = {"role": "assistant", "content": "hi"}
    with_meta = {
        "role": "assistant",
        "content": "hi",
        "id": "msg_abc",
        "model": "gpt-x",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 9999, "output_tokens": 1},
    }
    assert estimate_input_tokens([base]) == estimate_input_tokens([with_meta])


def test_estimate_handles_block_list_content() -> None:
    # The Anthropic ``{role, content}`` shape carries content as a block
    # list for assistant messages. The estimate serialises the projection
    # whole, so block-list content is covered (no crash, positive result).
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
    ]
    assert estimate_input_tokens(messages) > 0


# --- ContextBudget.is_known -------------------------------------------------


def test_is_known_tracks_ratio() -> None:
    # is_known is True iff ratio is not None — the consumer's gate.
    known = build_context_budget(used=10, capacity=100)
    unknown = build_context_budget(used=10, capacity=None)
    assert known.is_known is True
    assert unknown.is_known is False


# --- PressureLevel serialisation -------------------------------------------


def test_pressure_level_is_str_enum_for_json_friendliness() -> None:
    # The ``str`` mixin is deliberate: the value serialises to JSON /
    # logs without a custom encoder. The deferred consumer relies on this.
    assert PressureLevel.MEDIUM.value == "medium"
    assert PressureLevel.MEDIUM == "medium"  # str equality
    # JSON-serialisable as-is (no custom encoder).
    import json

    assert json.dumps(PressureLevel.HIGH.value) == '"high"'
