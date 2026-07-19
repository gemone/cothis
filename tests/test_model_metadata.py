"""Tests for ``cothis.model_metadata`` and the Agent's ``max_tokens`` wiring.

The resolver is the silent-breakage surface of slice #32: a wrong return
either cuts a generation off mid-tool-call (too small) or 400s the request
(too large). These tests cover every branch of the matching strategy
without touching the network — the bundled JSON is read via
``importlib.resources``, the same path the runtime uses.

The Agent-level tests cover the wiring: ``max_tokens=None`` resolves lazily
on first ``amessages`` call (and is cached); an explicit int wins and is
never re-resolved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from cothis.agent import Agent
from cothis.model_metadata import _FALLBACK_MAX_TOKENS, resolve_max_tokens

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from any_llm.types.messages import MessageResponse, MessageStreamEvent


# --- resolver ---------------------------------------------------------------


def test_resolve_exact_model_key_returns_max_output_tokens() -> None:
    # ``claude-sonnet-4-5`` is in litellm as a bare key with max_output=64000.
    assert resolve_max_tokens("claude-sonnet-4-5", "anthropic") == 64000


def test_resolve_prefixed_provider_model_key() -> None:
    # ``openai/gpt-oss-120b`` is NOT a bare key in litellm, but
    # ``openrouter/openai/gpt-oss-120b`` IS — the {provider}/{model} branch
    # must catch this (the default cothis model+provider combo).
    assert resolve_max_tokens("openai/gpt-oss-120b", "openrouter") == 32768


def test_resolve_prefixed_mistral_default() -> None:
    # ``mistral-small-latest`` resolves via ``mistral/mistral-small-latest``.
    assert (
        resolve_max_tokens("mistral-small-latest", "mistral")
        == resolve_max_tokens("mistral/mistral-small-latest", "mistral")
    )
    # And the value is non-trivial (> the 8192 fallback, so we know it matched).
    assert resolve_max_tokens("mistral-small-latest", "mistral") > _FALLBACK_MAX_TOKENS


def test_resolve_unknown_model_falls_back_to_8192() -> None:
    assert resolve_max_tokens("no-such-model-xyz", "openai") == _FALLBACK_MAX_TOKENS
    assert _FALLBACK_MAX_TOKENS == 8192


def test_resolve_override_always_wins() -> None:
    # Override beats both an exact match and the fallback.
    assert resolve_max_tokens("claude-sonnet-4-5", "anthropic", override=12345) == 12345
    assert resolve_max_tokens("no-such-model", "openai", override=7) == 7


@pytest.mark.parametrize("bad_override", [0, -1, -100])
def test_resolve_non_positive_override_ignored(bad_override: int) -> None:
    # A stray ``0`` from a misconfigured env var must not silently disable
    # generation; negative/zero overrides fall through to metadata lookup.
    assert (
        resolve_max_tokens(
            "claude-sonnet-4-5", "anthropic", override=bad_override
        )
        == 64000
    )


def test_resolve_legacy_max_tokens_field_used_when_max_output_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inject a fake entry that has ONLY the legacy ``max_tokens`` field.
    # ``_metadata`` is cached, so clear the cache before + after.
    from cothis import model_metadata

    fake = {"fake-legacy-model": {"max_tokens": 9999}}  # no max_output_tokens
    monkeypatch.setattr(model_metadata, "_metadata", lambda: fake)
    # ``_metadata`` is functools.cache-wrapped; bypass via the module attr.
    assert model_metadata.resolve_max_tokens("fake-legacy-model", "openai") == 9999


# --- Agent wiring -----------------------------------------------------------


def _make_agent(
    monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> Agent:
    """Build an Agent without making any LLM call.

    ``AnyLLM.create`` is patched to a MagicMock so no provider is contacted
    and no API key is required. The metadata-only fields we test don't need
    the LLM, so the resulting ``_llm`` is also a MagicMock.
    """
    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    return Agent(
        model=overrides.get("model", "openai/gpt-oss-120b"),
        provider=overrides.get("provider", "openrouter"),
        tools=[],
        max_iterations=5,
        max_tokens=overrides.get("max_tokens", None),
    )


def test_agent_effective_max_tokens_resolves_on_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)  # max_tokens=None
    # Sentinel before first call.
    assert agent._resolved_max_tokens == -1
    assert agent._effective_max_tokens() == 32768  # openrouter/openai/gpt-oss-120b
    # Cached afterwards — no re-resolve.
    assert agent._resolved_max_tokens == 32768


def test_agent_effective_max_tokens_explicit_value_wins_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch, max_tokens=1234)
    assert agent._effective_max_tokens() == 1234
    # Override wins and is cached on the instance.
    assert agent._resolved_max_tokens == 1234


def test_agent_effective_max_tokens_non_positive_override_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION: ``max_tokens=0`` must NOT be sent to the provider.

    The non-positive-override safety lives in ``resolve_max_tokens``; the
    Agent must route the override through the resolver (not short-circuit on
    ``is not None``) so that safety actually applies. Otherwise a stray
    ``COTHIS_MAX_TOKENS=0`` from a misconfigured env var reaches the provider
    and 400s cryptically.
    """
    agent = _make_agent(monkeypatch, max_tokens=0)
    assert agent._effective_max_tokens() == 32768  # resolved, not 0
    agent_neg = _make_agent(monkeypatch, max_tokens=-5)
    assert agent_neg._effective_max_tokens() == 32768


def test_agent_effective_max_tokens_unknown_model_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch, model="no-such-model", provider="openai")
    assert agent._effective_max_tokens() == _FALLBACK_MAX_TOKENS


def test_run_passes_resolved_max_tokens_to_amessages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run()`` forwards ``_effective_max_tokens()`` to ``amessages``.

    Two assertions: (a) the resolved value (not the 8192 fallback, not None)
    reaches ``amessages``; (b) an explicit ``max_tokens`` overrides the
    resolver. Guards against regressing back to a hardcoded constant.
    """
    agent = _make_agent(monkeypatch, model="openai/gpt-oss-120b", provider="openrouter")

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        # The wire contract under test: max_tokens is the resolved 32768.
        assert kwargs["max_tokens"] == 32768
        # Minimal end-turn response so the loop exits.
        return MagicMock(content=[], stop_reason="end_turn")  # type: ignore[return-value]

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    monkeypatch.setattr(agent, "_tool_schemas", lambda: [])
    monkeypatch.setattr(agent, "_ensure_messages", lambda _p: None)
    monkeypatch.setattr(agent, "_ensure_mcp", _async_noop)
    monkeypatch.setattr(agent, "_ensure_handles", _async_noop)
    monkeypatch.setattr(agent, "_execute_tool", _async_noop)  # not reached

    import asyncio

    result = asyncio.run(agent.run("hi"))
    assert result == ""  # no text blocks → empty final answer

    # Now with an explicit override, the wire value changes.
    agent2 = _make_agent(
        monkeypatch,
        model="openai/gpt-oss-120b",
        provider="openrouter",
        max_tokens=4321,
    )

    async def fake_amessages_2(**kwargs: Any) -> MessageResponse:
        assert kwargs["max_tokens"] == 4321
        return MagicMock(content=[], stop_reason="end_turn")  # type: ignore[return-value]

    monkeypatch.setattr(agent2._llm, "amessages", fake_amessages_2)
    monkeypatch.setattr(agent2, "_tool_schemas", lambda: [])
    monkeypatch.setattr(agent2, "_ensure_messages", lambda _p: None)
    monkeypatch.setattr(agent2, "_ensure_mcp", _async_noop)
    monkeypatch.setattr(agent2, "_ensure_handles", _async_noop)
    monkeypatch.setattr(agent2, "_execute_tool", _async_noop)

    asyncio.run(agent2.run("hi"))


async def _async_noop(*_args: Any, **_kwargs: Any) -> Any:
    """Coroutine stub for Agent internals the resolver test doesn't exercise."""
    return None
