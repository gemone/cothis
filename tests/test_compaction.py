"""Tests for ``cothis.ai.compaction`` — summariser selector + prompt builder.

Slice A of the compaction epic: two pure, unit-testable building blocks that
slices B (eviction policy) and C (run-loop wiring) consume. These tests pin
every branch of:

* :func:`resolve_summary_model` — precedence (override > env > session),
  ``provider/model`` slash-split, bare-model provider inheritance, and the
  empty / whitespace handling that falls through to the session pair.
* :func:`build_summarisation_request` — the system+user shape the provider
  expects, tool_use / tool_result / thinking block semantics, the
  ``system_text`` override, the oldest-first length cap, and empty-window
  tolerance.
* :func:`plan_eviction` (slice B) — the pressure gate (no-op under
  low/unknown pressure; evicts oldest under HIGH/CRITICAL), the retention
  floor, tool-pair closure / alternation preservation, determinism, and
  pressure monotonicity.
* the lazy-client invariant: the module imports without a provider SDK.
"""

from __future__ import annotations

from typing import Any

import pytest

from cothis.ai.compaction import (
    SUMMARY_SYSTEM_PROMPT,
    EvictionDecision,
    SummarisationRequest,
    SummaryTarget,
    build_summarisation_request,
    plan_eviction,
    resolve_summary_model,
)
from cothis.ai.context_budget import (
    ContextBudget,
    PressureLevel,
    build_context_budget,
    estimate_input_tokens,
)

_SESSION_MODEL = "claude-sonnet-4-5"
_SESSION_PROVIDER = "anthropic"


# --- resolve_summary_model: precedence --------------------------------------


def test_override_arg_wins_over_env_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Explicit override wins even when env and session conflict.
    monkeypatch.setenv("COTHIS_SUMMARY_MODEL", "openai/gpt-4o")
    target = resolve_summary_model(
        session_model=_SESSION_MODEL,
        session_provider=_SESSION_PROVIDER,
        override="mistral/mistral-large-latest",
    )
    assert target == SummaryTarget(
        model="mistral-large-latest", provider="mistral"
    )


def test_env_wins_over_session_when_override_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COTHIS_SUMMARY_MODEL", "openai/gpt-4o")
    target = resolve_summary_model(
        session_model=_SESSION_MODEL,
        session_provider=_SESSION_PROVIDER,
    )
    assert target == SummaryTarget(model="gpt-4o", provider="openai")


def test_env_with_slash_splits_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COTHIS_SUMMARY_MODEL", "anthropic/claude-sonnet-4-5")
    target = resolve_summary_model(
        session_model=_SESSION_MODEL,
        session_provider=_SESSION_PROVIDER,
    )
    assert target.provider == "anthropic"
    assert target.model == "claude-sonnet-4-5"


def test_env_bare_model_inherits_session_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No slash -> model inherits the session provider.
    monkeypatch.setenv("COTHIS_SUMMARY_MODEL", "claude-haiku-4-5")
    target = resolve_summary_model(
        session_model=_SESSION_MODEL,
        session_provider=_SESSION_PROVIDER,
    )
    assert target == SummaryTarget(
        model="claude-haiku-4-5", provider=_SESSION_PROVIDER
    )


def test_override_with_slash_overrides_provider_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A provider/model override swaps the provider, not just the model.
    monkeypatch.setenv("COTHIS_SUMMARY_MODEL", "anthropic/claude-sonnet-4-5")
    target = resolve_summary_model(
        session_model=_SESSION_MODEL,
        session_provider=_SESSION_PROVIDER,
        override="openai/gpt-4.1-mini",
    )
    assert target == SummaryTarget(model="gpt-4.1-mini", provider="openai")


def test_empty_override_falls_through_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Empty / whitespace-only override is "unset" -> env is consulted.
    monkeypatch.setenv("COTHIS_SUMMARY_MODEL", "openai/gpt-4o")
    target = resolve_summary_model(
        session_model=_SESSION_MODEL,
        session_provider=_SESSION_PROVIDER,
        override="   ",
    )
    assert target == SummaryTarget(model="gpt-4o", provider="openai")


def test_empty_override_and_env_fall_through_to_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both unset (empty/whitespace) -> session pair unchanged.
    monkeypatch.setenv("COTHIS_SUMMARY_MODEL", "   ")
    target = resolve_summary_model(
        session_model=_SESSION_MODEL,
        session_provider=_SESSION_PROVIDER,
        override="",
    )
    assert target == SummaryTarget(
        model=_SESSION_MODEL, provider=_SESSION_PROVIDER
    )


def test_everything_unset_returns_session_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COTHIS_SUMMARY_MODEL", raising=False)
    target = resolve_summary_model(
        session_model=_SESSION_MODEL,
        session_provider=_SESSION_PROVIDER,
    )
    assert target == SummaryTarget(
        model=_SESSION_MODEL, provider=_SESSION_PROVIDER
    )


def test_malformed_slash_only_spec_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A spec that parses to an empty model ("/" or "provider/") is unusable
    # and falls through to the session pair rather than selecting nothing.
    monkeypatch.setenv("COTHIS_SUMMARY_MODEL", "/")
    target = resolve_summary_model(
        session_model=_SESSION_MODEL,
        session_provider=_SESSION_PROVIDER,
    )
    assert target == SummaryTarget(
        model=_SESSION_MODEL, provider=_SESSION_PROVIDER
    )


def test_override_spec_with_whitespace_is_trimmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COTHIS_SUMMARY_MODEL", raising=False)
    target = resolve_summary_model(
        session_model=_SESSION_MODEL,
        session_provider=_SESSION_PROVIDER,
        override="  openai / gpt-4o  ",
    )
    assert target == SummaryTarget(model="gpt-4o", provider="openai")


def test_env_not_read_when_override_supplies_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Override supplies a usable value, so the env var is never consulted —
    # proven by leaving it unset (would fall to session if read+empty).
    monkeypatch.delenv("COTHIS_SUMMARY_MODEL", raising=False)
    target = resolve_summary_model(
        session_model=_SESSION_MODEL,
        session_provider=_SESSION_PROVIDER,
        override="openai/gpt-4o",
    )
    assert target == SummaryTarget(model="gpt-4o", provider="openai")


def test_summary_target_is_frozen() -> None:
    # Frozen dataclass: selection result is a snapshot, never mutated.
    target = SummaryTarget(model="m", provider="p")
    with pytest.raises(Exception):  # noqa: B017 — frozen-attr exception type varies
        setattr(target, "model", "other")


# --- build_summarisation_request: shape -------------------------------------


def _sample_window() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "Please add a README."}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll create the README now."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "fs_write",
                    "input": {"path": "README.md", "content": "# project"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "wrote 9 bytes",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "internal reasoning here"},
                {"type": "text", "text": "Done — README created."},
            ],
        },
    ]


def test_builder_emits_well_shaped_request() -> None:
    request = build_summarisation_request(
        window=_sample_window(), max_tokens=2048
    )
    # system is exactly one text block carrying the condensation instruction.
    assert isinstance(request, SummarisationRequest)
    assert request.max_tokens == 2048
    assert len(request.system) == 1
    assert request.system[0] == {"type": "text", "text": SUMMARY_SYSTEM_PROMPT}
    # messages is exactly one user turn with exactly one text block.
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"
    content = request.messages[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert isinstance(content[0]["text"], str)


def test_builder_renders_user_and_assistant_text_with_role_prefix() -> None:
    request = build_summarisation_request(
        window=_sample_window(), max_tokens=1024
    )
    transcript = request.messages[0]["content"][0]["text"]
    assert "User: Please add a README." in transcript
    assert "Assistant: I'll create the README now." in transcript
    assert "Assistant: Done — README created." in transcript


def test_builder_renders_tool_use_with_name_input_and_id() -> None:
    request = build_summarisation_request(
        window=_sample_window(), max_tokens=1024
    )
    transcript = request.messages[0]["content"][0]["text"]
    # name, id, and the rendered input all appear in the tool_use line.
    assert "Assistant called fs_write(" in transcript
    assert "[id=toolu_1]" in transcript
    # Input rendered as deterministic JSON carrying the path argument.
    assert '"path": "README.md"' in transcript


def test_builder_renders_tool_result_with_id_and_content() -> None:
    request = build_summarisation_request(
        window=_sample_window(), max_tokens=1024
    )
    transcript = request.messages[0]["content"][0]["text"]
    assert "Tool result (toolu_1): wrote 9 bytes" in transcript


def test_builder_skips_thinking_blocks() -> None:
    request = build_summarisation_request(
        window=_sample_window(), max_tokens=1024
    )
    transcript = request.messages[0]["content"][0]["text"]
    # The thinking block's internal text must NOT leak into the transcript.
    assert "internal reasoning here" not in transcript


def test_builder_system_text_override_replaces_default() -> None:
    custom = "Custom condensation directive: summarise in haiku form."
    request = build_summarisation_request(
        window=_sample_window(), max_tokens=1024, system_text=custom
    )
    assert request.system == [{"type": "text", "text": custom}]
    assert SUMMARY_SYSTEM_PROMPT not in request.system[0]["text"]


def test_builder_passes_max_tokens_through_unchanged() -> None:
    for cap in (1, 8192, 1_000_000):
        request = build_summarisation_request(window=[], max_tokens=cap)
        assert request.max_tokens == cap


def test_builder_system_block_has_no_cache_control() -> None:
    # One-shot ephemeral call: no breakpoint is anchored on the system block
    # (the Anthropic provider adds one idempotently if needed).
    request = build_summarisation_request(window=_sample_window(), max_tokens=512)
    assert "cache_control" not in request.system[0]


# --- build_summarisation_request: length cap --------------------------------


def test_length_cap_truncates_oldest_and_preserves_recent_verbatim() -> None:
    # Each message is small; together they exceed the cap. The most recent
    # (tail) turn survives verbatim; the oldest (head) is dropped behind the
    # truncation marker.
    window = [
        {"role": "user", "content": [{"type": "text", "text": "OLD_HEAD_TURN_X"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "middle reply"}]},
        {"role": "user", "content": [{"type": "text", "text": "RECENT_TAIL_TURN_Y"}]},
    ]
    request = build_summarisation_request(
        window=window, max_tokens=1024, max_window_chars=40
    )
    transcript = request.messages[0]["content"][0]["text"]
    assert "...[older turns truncated]..." in transcript
    assert "RECENT_TAIL_TURN_Y" in transcript  # tail preserved verbatim
    assert "OLD_HEAD_TURN_X" not in transcript  # head dropped


def test_length_cap_no_truncation_when_under_limit() -> None:
    window = [
        {"role": "user", "content": [{"type": "text", "text": "small"}]},
    ]
    request = build_summarisation_request(
        window=window, max_tokens=1024, max_window_chars=10_000
    )
    transcript = request.messages[0]["content"][0]["text"]
    assert "...[older turns truncated]..." not in transcript
    assert transcript == "User: small"


def test_length_cap_keeps_at_least_one_turn_when_single_exceeds_cap() -> None:
    # Degenerate case: one message alone exceeds the cap. The most recent
    # turn is still retained (a marker-only transcript is useless); the
    # overall cap is allowed to overflow in this corner case.
    window = [
        {"role": "user", "content": [{"type": "text", "text": "A" * 200}]},
    ]
    request = build_summarisation_request(
        window=window, max_tokens=1024, max_window_chars=50
    )
    transcript = request.messages[0]["content"][0]["text"]
    assert "A" * 200 in transcript


# --- build_summarisation_request: empty + tolerant window --------------------


def test_empty_window_does_not_crash() -> None:
    request = build_summarisation_request(window=[], max_tokens=1024)
    assert request.system == [{"type": "text", "text": SUMMARY_SYSTEM_PROMPT}]
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"
    # Transcript text is empty (or minimal) — but the shape stays valid.
    assert request.messages[0]["content"][0]["text"] == ""


def test_thinking_only_window_renders_empty_transcript() -> None:
    # A window whose every block is skipped renders to an empty transcript
    # rather than crashing or emitting empty labelled lines.
    window = [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]},
    ]
    request = build_summarisation_request(window=window, max_tokens=1024)
    assert request.messages[0]["content"][0]["text"] == ""


def test_string_content_message_is_rendered() -> None:
    # Some callers carry content as a bare string rather than a block list.
    window = [{"role": "user", "content": "hello there"}]
    request = build_summarisation_request(window=window, max_tokens=1024)
    transcript = request.messages[0]["content"][0]["text"]
    assert transcript == "User: hello there"


def test_stored_assistant_metadata_is_ignored() -> None:
    # The stored assistant shape carries id/model/stop_reason/usage alongside
    # role/content. The renderer reads only role+content, so metadata never
    # reaches the transcript.
    window: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "real answer"}],
            "id": "msg_abc",
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 9999, "output_tokens": 1},
        },
    ]
    request = build_summarisation_request(window=window, max_tokens=1024)
    transcript = request.messages[0]["content"][0]["text"]
    assert transcript == "Assistant: real answer"
    assert "msg_abc" not in transcript
    assert "end_turn" not in transcript


def test_request_is_frozen() -> None:
    request = build_summarisation_request(window=[], max_tokens=1)
    with pytest.raises(Exception):  # noqa: B017 — frozen-attr exception type varies
        setattr(request, "max_tokens", 2)


# --- lazy-client invariant --------------------------------------------------


def test_module_has_no_top_level_provider_sdk_import() -> None:
    # Importing this module must not drag in a provider SDK (matches the
    # context_budget.py / _retry.py lazy-client discipline). Guarded
    # statically so a future regression cannot slip past CI.
    import inspect
    import re

    from cothis.ai import compaction

    src = inspect.getsource(compaction)
    # Only unindented lines can be top-level import statements; an indented
    # (function-local) import is permitted in principle though this module
    # has none.
    top_level = "\n".join(
        line for line in src.splitlines() if line and not line[0].isspace()
    )
    assert not re.search(r"\bimport\s+anthropic\b", top_level)
    assert not re.search(r"\bimport\s+openai\b", top_level)
    assert not re.search(r"\bfrom\s+anthropic\b", top_level)
    assert not re.search(r"\bfrom\s+openai\b", top_level)


def test_public_symbols_reexported_from_package() -> None:
    # The frozen public surface must re-export the slice-A and slice-B symbols
    # so later slices can ``from cothis.ai import plan_eviction`` etc.
    from cothis.ai import (  # noqa: F401 — exercising the re-export
        EvictionDecision as _ED,
    )
    from cothis.ai import (
        SummarisationRequest as _R,
    )
    from cothis.ai import (
        SummaryTarget as _T,
    )
    from cothis.ai import (
        build_summarisation_request as _B,
    )
    from cothis.ai import (
        plan_eviction as _PE,
    )
    from cothis.ai import (
        resolve_summary_model as _S,
    )


# --- slice B: plan_eviction fixtures + helpers ------------------------------


# Per-message body text sized so a multi-pair conversation's estimate lands
# well inside the pressure buckets (avoids int-truncation edge cases at the
# bucket boundaries when deriving ``capacity`` from the estimate).
_LARGE_TEXT = "x" * 400


def _text_user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _text_asst(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _conversation_pairs(
    n_pairs: int, *, text: str = _LARGE_TEXT
) -> list[dict[str, Any]]:
    """Build a strictly-alternating no-tool conversation of ``n_pairs`` turns.

    Each pair carries a unique marker so dict equality distinguishes messages
    (and ``in`` membership checks are unambiguous).
    """
    msgs: list[dict[str, Any]] = []
    for i in range(n_pairs):
        msgs.append(_text_user(f"{text} pair{i}"))
        msgs.append(_text_asst(f"reply {i}"))
    return msgs


def _budget_at_ratio(
    messages: list[dict[str, Any]], ratio: float
) -> ContextBudget:
    """Build a budget whose ``used`` equals the messages' estimate and whose
    capacity lands the ratio at ``ratio``."""
    est = estimate_input_tokens(messages)
    capacity = max(1, int(est / ratio)) if est > 0 else 1
    return build_context_budget(used=est, capacity=capacity)


def _high_budget(messages: list[dict[str, Any]]) -> ContextBudget:
    return _budget_at_ratio(messages, 0.95)


def _critical_budget(messages: list[dict[str, Any]]) -> ContextBudget:
    return _budget_at_ratio(messages, 1.05)


def _critical_budget_fixed() -> ContextBudget:
    """A CRITICAL budget independent of any message list (ratio well over 1)."""
    return build_context_budget(used=10_000, capacity=1_000)


def _tool_use_ids(msgs: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for m in msgs:
        for b in m.get("content", []) if isinstance(m.get("content"), list) else []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                ids.add(str(b.get("id")))
    return ids


def _tool_result_ids(msgs: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for m in msgs:
        for b in m.get("content", []) if isinstance(m.get("content"), list) else []:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                ids.add(str(b.get("tool_use_id")))
    return ids


def _assert_strict_alternation(msgs: list[dict[str, Any]]) -> None:
    roles = [m.get("role") for m in msgs]
    for prev, cur in zip(roles, roles[1:]):
        assert prev != cur, f"retained tail is not strictly alternating: {roles}"


# --- slice B: pressure gate -------------------------------------------------


@pytest.mark.parametrize(
    "ratio,expected",
    [
        (0.10, PressureLevel.NONE),
        (0.60, PressureLevel.LOW),
        (0.80, PressureLevel.MEDIUM),
    ],
)
def test_no_eviction_under_low_pressure(
    ratio: float, expected: PressureLevel
) -> None:
    # NONE / LOW / MEDIUM are "plan, not act": the window stays empty.
    messages = _conversation_pairs(8)
    budget = _budget_at_ratio(messages, ratio)
    assert budget.pressure == expected  # sanity: landed in the right bucket
    decision = plan_eviction(messages=messages, budget=budget)
    assert isinstance(decision, EvictionDecision)
    assert decision.window == []
    assert decision.retained == messages
    assert decision.evicted_token_estimate == 0
    assert decision.reason.startswith("no-eviction:low-pressure")
    assert decision.pressure == expected


def test_no_eviction_when_budget_unknown() -> None:
    # capacity None -> is_known False -> no signal to act on.
    messages = _conversation_pairs(8)
    budget_cap_none = build_context_budget(used=1000, capacity=None)
    assert not budget_cap_none.is_known
    decision = plan_eviction(messages=messages, budget=budget_cap_none)
    assert decision.window == []
    assert decision.retained == messages
    assert decision.reason == "no-eviction:unknown-budget"
    assert decision.pressure is None
    # used None collapses the same way.
    budget_used_none = build_context_budget(used=None, capacity=1000)
    assert not budget_used_none.is_known
    assert (
        plan_eviction(messages=messages, budget=budget_used_none).reason
        == "no-eviction:unknown-budget"
    )


# --- slice B: HIGH eviction -------------------------------------------------


def test_evicts_oldest_under_high_pressure() -> None:
    # Estimate pushed into [0.90, 1.00); the oldest turn(s) enter the window.
    messages = _conversation_pairs(8)
    budget = _high_budget(messages)
    assert budget.pressure == PressureLevel.HIGH
    decision = plan_eviction(messages=messages, budget=budget)
    assert decision.pressure == PressureLevel.HIGH
    assert decision.reason.startswith("evicted:high-pressure")
    # Non-empty contiguous prefix; retained is the matching tail.
    assert len(decision.window) > 0
    assert decision.window == messages[: len(decision.window)]
    assert decision.retained == messages[len(decision.window) :]
    # The OLDEST message was evicted, the NEWEST retained.
    assert messages[0] in decision.window
    assert messages[0] not in decision.retained
    assert messages[-1] in decision.retained
    assert decision.evicted_token_estimate > 0
    assert decision.evicted_token_estimate == estimate_input_tokens(decision.window)


def test_high_pressure_meets_target_ratio() -> None:
    # HIGH evicts the MINIMAL prefix that brings the retained estimate down
    # to ~high_target_ratio * capacity; the retained tail fits the target.
    messages = _conversation_pairs(8)
    budget = _high_budget(messages)
    decision = plan_eviction(messages=messages, budget=budget)
    capacity = budget.capacity_tokens
    assert capacity is not None
    target = 0.75 * capacity
    # Retained estimate is at or under the target (minimal eviction)...
    assert estimate_input_tokens(decision.retained) <= target
    # ...and dropping one fewer turn would have exceeded it (minimality):
    # the message just before the cut, if returned to the tail, overshoots.
    if len(decision.window) >= 1:
        one_less = messages[: len(decision.window) - 1]
        retained_if_smaller = messages[len(one_less) :]
        assert estimate_input_tokens(retained_if_smaller) > target


# --- slice B: retention floor -----------------------------------------------


def test_retention_floor_holds_under_critical() -> None:
    # 6 turn-groups; CRITICAL evicts maximally but keeps the last 4 groups.
    messages = _conversation_pairs(6)
    budget = _critical_budget(messages)
    assert budget.pressure == PressureLevel.CRITICAL
    decision = plan_eviction(messages=messages, budget=budget, min_retained_turns=4)
    assert decision.reason.startswith("evicted:critical-pressure")
    # No-tool conversation -> floor boundary is itself a safe cut, so CRITICAL
    # cuts exactly there: window = first 2 groups, retained = last 4 groups.
    assert decision.window == messages[:4]
    assert decision.retained == messages[4:]


def test_critical_target_met_is_not_maximal() -> None:
    # When the critical_floor_ratio target is met by a non-maximal safe cut,
    # CRITICAL takes the minimal cut (reason 'evicted:critical-pressure', not
    # the '-maximal' fallback). Generous capacity -> the 0.50 target dwarfs
    # the retained tail, so the SMALLEST safe cut already satisfies it. This
    # is the path where critical_floor_ratio actually drives the decision.
    messages = _conversation_pairs(6)
    budget = build_context_budget(used=10_000, capacity=10_000)
    assert budget.pressure == PressureLevel.CRITICAL
    decision = plan_eviction(messages=messages, budget=budget, min_retained_turns=4)
    assert decision.reason == "evicted:critical-pressure"
    assert len(decision.window) > 0
    assert decision.retained == messages[len(decision.window) :]


def test_below_floor_is_no_op() -> None:
    # 4 turn-groups == min_retained_turns -> nothing can be evicted.
    messages = _conversation_pairs(4)
    budget = _critical_budget(messages)
    decision = plan_eviction(messages=messages, budget=budget, min_retained_turns=4)
    assert decision.window == []
    assert decision.retained == messages
    assert decision.evicted_token_estimate == 0
    assert decision.reason == "no-eviction:below-floor"


def test_empty_conversation_is_no_op() -> None:
    decision = plan_eviction(messages=[], budget=_critical_budget_fixed())
    assert decision.window == []
    assert decision.retained == []
    assert decision.evicted_token_estimate == 0
    assert decision.reason == "no-eviction:below-floor"


def test_short_conversation_is_no_op() -> None:
    # 2 turn-groups < min_retained_turns(4) -> below floor regardless of pressure.
    messages = _conversation_pairs(2)
    budget = _critical_budget(messages)
    decision = plan_eviction(messages=messages, budget=budget)
    assert decision.window == []
    assert decision.retained == messages
    assert decision.reason == "no-eviction:below-floor"


# --- slice B: window shape feeds slice A ------------------------------------


def test_window_feeds_build_summarisation_request() -> None:
    # Round-trip contract between slices A and B: the emitted window is the
    # exact shape build_summarisation_request consumes, without raising.
    messages = _conversation_pairs(8)
    budget = _high_budget(messages)
    decision = plan_eviction(messages=messages, budget=budget)
    assert len(decision.window) > 0
    request = build_summarisation_request(window=decision.window, max_tokens=1024)
    assert isinstance(request, SummarisationRequest)
    assert request.max_tokens == 1024
    assert len(request.system) == 1
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"


# --- slice B: tool-pair closure / alternation -------------------------------


def _tool_heavy_conversation() -> list[dict[str, Any]]:
    """A conversation where a tool_use/tool_result pair straddles the naive
    mid-list cut a token-only walk would pick."""
    return [
        _text_user(f"{_LARGE_TEXT} q0"),
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "fs.read", "input": {"p": 1}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "r" * 400,
                }
            ],
        },
        _text_asst("done0"),
        _text_user(f"{_LARGE_TEXT} q1"),
        _text_asst("a1"),
        _text_user(f"{_LARGE_TEXT} q2"),
        _text_asst("a2"),
        _text_user(f"{_LARGE_TEXT} q3"),
        _text_asst("a3"),
        _text_user(f"{_LARGE_TEXT} q4"),
        _text_asst("a4"),
    ]


def test_tool_pair_closure_preserved_under_eviction() -> None:
    messages = _tool_heavy_conversation()
    budget = _critical_budget(messages)
    assert budget.pressure == PressureLevel.CRITICAL
    decision = plan_eviction(messages=messages, budget=budget, min_retained_turns=4)
    assert decision.reason.startswith("evicted:critical-pressure")
    # The hard invariant: every tool_use id in the window has its tool_result
    # in the window, and every tool_result in the retained tail has its
    # tool_use in the retained tail. No dangling references either side.
    window_uses = _tool_use_ids(decision.window)
    window_results = _tool_result_ids(decision.window)
    retained_uses = _tool_use_ids(decision.retained)
    retained_results = _tool_result_ids(decision.retained)
    assert window_results <= window_uses, (
        f"tool_result in window without its tool_use: "
        f"{window_results - window_uses}"
    )
    assert retained_results <= retained_uses, (
        f"tool_result in retained without its tool_use: "
        f"{retained_results - retained_uses}"
    )
    # Role alternation preserved in both halves.
    _assert_strict_alternation(decision.retained)
    _assert_strict_alternation(decision.window)


def test_no_safe_cut_yields_empty_window() -> None:
    # Pathological: the FIRST message opens a tool_use that never closes, so
    # no cut > 0 is safe. floor_boundary > 0 but the safe-cut candidate set
    # is empty -> the decision refuses to emit a dangling window.
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tX", "name": "fs.read", "input": {}}
            ],
        },
        _text_user(f"{_LARGE_TEXT} u1"),
        _text_asst("a1"),
        _text_user(f"{_LARGE_TEXT} u2"),
        _text_asst("a2"),
        _text_user(f"{_LARGE_TEXT} u3"),
        _text_asst("a3"),
        _text_user(f"{_LARGE_TEXT} u4"),
        _text_asst("a4"),
    ]
    budget = _critical_budget_fixed()
    decision = plan_eviction(messages=messages, budget=budget, min_retained_turns=4)
    assert decision.window == []
    assert decision.retained == messages
    assert decision.reason == "no-eviction:no-safe-cut"


# --- slice B: determinism + monotonicity ------------------------------------


def test_plan_eviction_is_deterministic() -> None:
    messages = _conversation_pairs(8)
    budget = _high_budget(messages)
    first = plan_eviction(messages=messages, budget=budget)
    second = plan_eviction(messages=messages, budget=budget)
    # Frozen dataclass __eq__: identical input -> equal decision.
    assert first == second
    assert first.evicted_token_estimate == second.evicted_token_estimate
    assert [id(a) for a in first.retained] == [id(a) for a in second.retained]


def test_critical_evicts_at_least_as_much_as_high() -> None:
    # Same messages: CRITICAL's reclaimed tokens >= HIGH's; both >= MEDIUM
    # (which is empty). Monotonic aggressiveness.
    messages = _conversation_pairs(8)
    high_decision = plan_eviction(messages=messages, budget=_high_budget(messages))
    critical_decision = plan_eviction(
        messages=messages, budget=_critical_budget(messages)
    )
    assert high_decision.evicted_token_estimate > 0
    assert critical_decision.evicted_token_estimate >= high_decision.evicted_token_estimate
    # MEDIUM is the no-eviction boundary.
    medium_decision = plan_eviction(
        messages=messages, budget=_budget_at_ratio(messages, 0.80)
    )
    assert medium_decision.evicted_token_estimate == 0
    assert medium_decision.window == []


# --- slice B: stored-with-metadata tolerance + frozen -----------------------


def test_stored_assistant_metadata_tolerated_by_plan_eviction() -> None:
    # Assistant dicts carry id/model/stop_reason/usage; plan_eviction reads
    # only role+content, and retained preserves the original dict objects.
    messages: list[dict[str, Any]] = []
    for i in range(8):
        messages.append(_text_user(f"{_LARGE_TEXT} u{i}"))
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"a{i}"}],
                "id": f"msg_{i}",
                "model": "claude-sonnet-4-5",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 5},
            }
        )
    budget = _high_budget(messages)
    decision = plan_eviction(messages=messages, budget=budget)
    assert budget.pressure == PressureLevel.HIGH
    assert len(decision.window) > 0
    # Identity preserved: retained entries ARE the original dicts (not copies),
    # so their metadata survived untouched.
    for retained_msg in decision.retained:
        assert any(retained_msg is original for original in messages)
        if retained_msg.get("role") == "assistant":
            assert "id" in retained_msg and retained_msg["id"].startswith("msg_")
            assert "usage" in retained_msg


def test_eviction_decision_is_frozen() -> None:
    messages = _conversation_pairs(6)
    decision = plan_eviction(messages=messages, budget=_high_budget(messages))
    with pytest.raises(Exception):  # noqa: B017 — frozen-attr exception type varies
        setattr(decision, "reason", "tampered")
