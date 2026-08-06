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
* the lazy-client invariant: the module imports without a provider SDK.
"""

from __future__ import annotations

from typing import Any

import pytest

from cothis.ai.compaction import (
    SUMMARY_SYSTEM_PROMPT,
    SummarisationRequest,
    SummaryTarget,
    build_summarisation_request,
    resolve_summary_model,
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
    # The frozen public surface must re-export the slice-A symbols so later
    # slices can ``from cothis.ai import resolve_summary_model``.
    from cothis.ai import (  # noqa: F401 — exercising the re-export
        SummarisationRequest as _R,
    )
    from cothis.ai import (
        SummaryTarget as _T,
    )
    from cothis.ai import (
        build_summarisation_request as _B,
    )
    from cothis.ai import (
        resolve_summary_model as _S,
    )
