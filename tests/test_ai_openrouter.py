"""Tests for :class:`cothis.ai.openrouter.OpenRouterProvider`.

OpenRouter is the OpenAI SDK pinned to the OpenRouter base URL — a thin
subclass. We smoke-test the ``base_url`` wiring (default + caller override)
and confirm the SDK client receives it. Inherited OpenAI translation
behaviour is covered by ``tests/test_ai_openai.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from cothis.ai.openrouter import OPENROUTER_BASE_URL, OpenRouterProvider

if TYPE_CHECKING:
    import pytest


def test_default_base_url_is_openrouter() -> None:
    provider = OpenRouterProvider(api_key="sk-or", api_base=None)
    assert provider._api_base == OPENROUTER_BASE_URL
    assert provider._api_base == "https://openrouter.ai/api/v1"


def test_caller_base_url_overrides_default() -> None:
    provider = OpenRouterProvider(api_key="sk-or", api_base="https://gw.test/v1")
    assert provider._api_base == "https://gw.test/v1"


def test_base_url_forwarded_to_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        class chat:  # noqa: N801
            class completions:
                @staticmethod
                async def create(**_: object) -> object:
                    return SimpleNamespace(
                        id="",
                        model="m",
                        choices=[],
                        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
                    )

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    provider = OpenRouterProvider(api_key="sk-or", api_base=None)
    asyncio.run(
        provider.amessages(model="openai/gpt-oss-120b", messages=[], max_tokens=1)
    )
    assert captured == {"api_key": "sk-or", "base_url": OPENROUTER_BASE_URL}


def test_openrouter_is_an_openai_provider() -> None:
    from cothis.ai.openai import OpenAIProvider

    provider = OpenRouterProvider(api_key=None, api_base=None)
    assert isinstance(provider, OpenAIProvider)
