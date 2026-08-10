"""Factory tests for :mod:`cothis.ai`.

Covers provider routing, the ``ValueError`` on unknown keys, and that
``api_key`` / ``api_base`` are forwarded to the underlying SDK clients
without ever needing a real credential (clients construct lazily).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

import cothis.ai as ai
from cothis.ai import AIProvider, get_provider
from cothis.ai.anthropic import AnthropicProvider
from cothis.ai.google import GoogleProvider
from cothis.ai.openai import OpenAIProvider
from cothis.ai.openrouter import OPENROUTER_BASE_URL, OpenRouterProvider


def _providers() -> dict[str, type]:
    return {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "openrouter": OpenRouterProvider,
        "google": GoogleProvider,
    }


@pytest.mark.parametrize("key,cls", list(_providers().items()))
def test_get_provider_returns_correct_class(key: str, cls: type) -> None:
    """Each known provider key resolves to its provider class."""
    provider = get_provider(key)
    assert isinstance(provider, cls)
    assert isinstance(provider, AIProvider)


def test_mistral_routed_to_openai_provider_with_mistral_base() -> None:
    """``mistral`` is OpenAI-compatible — routed via OpenAIProvider + base URL."""
    provider = get_provider("mistral")
    assert isinstance(provider, OpenAIProvider)
    assert provider._api_base == "https://api.mistral.ai/v1"


def test_get_provider_is_case_insensitive() -> None:
    assert isinstance(get_provider("Anthropic"), AnthropicProvider)
    assert isinstance(get_provider("OPENAI"), OpenAIProvider)


def test_unknown_provider_raises_value_error_naming_known() -> None:
    with pytest.raises(ValueError) as excinfo:
        get_provider("nope")
    msg = str(excinfo.value)
    assert "nope" in msg
    for known in ("anthropic", "openai", "openrouter", "google", "mistral"):
        assert known in msg


def test_factory_needs_no_api_key_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_provider returns instances even with no creds in the environment."""
    # Strip any ambient provider env vars so the test is hermetic.
    for env in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_API_KEY",
    ):
        monkeypatch.delenv(env, raising=False)
    for key in ("anthropic", "openai", "openrouter", "google", "mistral"):
        assert isinstance(get_provider(key), AIProvider)


def test_api_key_and_api_base_forwarded_to_anthropic_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """api_key/base_url reach the SDK constructor verbatim."""
    captured: dict[str, object] = {}

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        class messages:  # noqa: N801 - matches attribute access path
            @staticmethod
            async def create(**_: object) -> object:
                return MagicMock()

    monkeypatch.setattr("anthropic.AsyncAnthropic", FakeAsyncAnthropic)
    provider = get_provider(
        "anthropic", api_key="sk-test", api_base="https://example.test"
    )
    # Force lazy client construction.
    asyncio.run(
        provider.amessages(
            model="m", messages=[], max_tokens=1, system=None, tools=None
        )
    )
    assert captured == {"api_key": "sk-test", "base_url": "https://example.test"}


def test_api_key_and_api_base_forwarded_to_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        class chat:  # noqa: N801 - matches attribute access path
            class completions:
                @staticmethod
                async def create(**_: object) -> object:
                    # Minimal completion that translates cleanly through
                    # _completion_to_message (empty choices -> empty content).
                    from types import SimpleNamespace

                    return SimpleNamespace(
                        id="",
                        model="m",
                        choices=[],
                        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
                    )

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    provider = get_provider("openai", api_key="sk-oai", api_base="https://oai.test")
    asyncio.run(
        provider.amessages(
            model="m", messages=[], max_tokens=1, system=None, tools=None
        )
    )
    assert captured == {"api_key": "sk-oai", "base_url": "https://oai.test"}


def test_openrouter_defaults_to_openrouter_base_url() -> None:
    provider = get_provider("openrouter")
    assert isinstance(provider, OpenRouterProvider)
    assert provider._api_base == OPENROUTER_BASE_URL


def test_openrouter_caller_base_url_overrides_default() -> None:
    provider = get_provider("openrouter", api_base="https://custom.test/v1")
    assert isinstance(provider, OpenRouterProvider)
    assert provider._api_base == "https://custom.test/v1"


def test_module_reexports_canonical_types() -> None:
    """The frozen public type vocabulary is re-exported from cothis.ai."""
    for name in (
        "MessageResponse",
        "MessageStreamEvent",
        "Usage",
        "TextBlock",
        "ToolUseBlock",
        "TextDelta",
        "InputJSONDelta",
        "StopReason",
    ):
        assert hasattr(ai, name), f"missing re-export: {name}"


def test_provider_specific_env_key_is_read_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_provider`` maps the provider's canonical env var when no key is passed.

    The OpenAI SDK only auto-reads ``OPENAI_API_KEY``; a user with
    ``OPENROUTER_API_KEY`` set (the natural key) previously got ``Missing
    credentials`` because the SDK ignored it.
    """
    from cothis.ai import get_provider

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    provider = get_provider("openrouter")
    assert provider._api_key == "or-key"

    monkeypatch.setenv("MISTRAL_API_KEY", "mi-key")
    provider = get_provider("mistral")
    assert provider._api_key == "mi-key"

    # Explicit key still wins over the env.
    provider = get_provider("openrouter", api_key="explicit")
    assert provider._api_key == "explicit"
