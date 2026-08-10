"""AIProvider Protocol + factory. The single surface ``Agent`` imports.

The agent loop talks to its LLM through one method, ``amessages``, which
mirrors the Anthropic Messages API. Each provider implements it and
returns/streams the canonical :mod:`cothis.ai._types` shapes. This module
defines the Protocol (for type-checking and structural matching) and the
factory that resolves a provider key to a concrete instance.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cothis.ai._types import MessageResponse, MessageStreamEvent

# Provider keys cothis ships with.
_KNOWN_PROVIDERS = (
    "anthropic",
    "openai",
    "openrouter",
    "google",
    "mistral",
    "deepseek",
    "groq",
)

# OpenAI-compatible providers routed through ``OpenAIProvider`` with a
# pinned base URL. Each speaks the OpenAI Chat Completions wire format, so
# the OpenAI SDK drives them directly — no per-provider SDK dependency.
# ``api_base`` overrides the default when the caller passes one.
_OPENAI_COMPAT: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
}

# Canonical API-key env var per provider, used when the caller passes no
# explicit ``api_key``. The OpenAI SDK's default is ``OPENAI_API_KEY``, which
# is wrong for the OpenAI-compatible providers and OpenRouter — a user with
# ``OPENROUTER_API_KEY`` set (the natural key) got ``Missing credentials``
# because the SDK only auto-reads ``OPENAI_API_KEY``.
_API_KEY_ENV_BY_PROVIDER: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
}


@runtime_checkable
class AIProvider(Protocol):
    """One LLM provider speaking the Anthropic Messages shape.

    The signature mirrors the ``amessages`` method the agent consumes,
    so the agent is a near-zero diff across providers. Non-stream returns
    an ``anthropic.types.Message``; stream returns an ``AsyncIterator``
    over ``anthropic.types.RawMessageStreamEvent``.
    """

    @overload
    async def amessages(self, *, model: str, messages: list[dict[str, Any]], max_tokens: int, system: list[dict[str, Any]] | None = None, tools: list[dict[str, Any]] | None = None, stream: Literal[False] = False, session_id: str | None = None) -> MessageResponse: ...
    @overload
    async def amessages(self, *, model: str, messages: list[dict[str, Any]], max_tokens: int, system: list[dict[str, Any]] | None = None, tools: list[dict[str, Any]] | None = None, stream: Literal[True], session_id: str | None = None) -> AsyncIterator[MessageStreamEvent]: ...
    async def amessages(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        session_id: str | None = None,
    ) -> MessageResponse | AsyncIterator[MessageStreamEvent]: ...


def get_provider(
    provider: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
) -> AIProvider:
    """Resolve a provider key to a direct-SDK :class:`AIProvider` instance.

    Provider keys (case-insensitive):

    * ``anthropic``  — native Anthropic Messages API (zero translation).
    * ``openai``     — OpenAI Chat Completions via the OpenAI SDK.
    * ``openrouter`` — OpenAI-SDK-compatible; pinned to the OpenRouter base URL.
    * ``google``     — Google GenAI (Gemini) via ``google-genai``.
    * ``mistral``    — OpenAI-compatible; routed through the OpenAI SDK with
      Mistral's base URL (no separate SDK dependency).
    * ``deepseek``   — OpenAI-compatible; routed through the OpenAI SDK with
      DeepSeek's base URL.
    * ``groq``       — OpenAI-compatible; routed through the OpenAI SDK with
      Groq's base URL.

    SDK clients are constructed lazily inside each provider (on first
    ``amessages`` call), so this factory returns an instance even when no
    API key or env var is set. An unknown key raises :class:`ValueError`
    naming the key and the known providers.
    """
    p = provider.lower()
    if p == "anthropic":
        from cothis.ai.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=api_key, api_base=api_base)
    if p == "openai":
        from cothis.ai.openai import OpenAIProvider

        return OpenAIProvider(api_key=api_key, api_base=api_base)
    if p == "openrouter":
        from cothis.ai.openrouter import OpenRouterProvider

        return OpenRouterProvider(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            api_base=api_base,
        )
    if p in _OPENAI_COMPAT:
        # OpenAI-compatible providers (DeepSeek, Groq, Mistral) route
        # through the OpenAI SDK with a pinned base URL unless the caller
        # overrode ``api_base``.
        from cothis.ai.openai import OpenAIProvider

        return OpenAIProvider(
            api_key=api_key or os.environ.get(_API_KEY_ENV_BY_PROVIDER[p]),
            api_base=api_base or _OPENAI_COMPAT[p],
        )
    if p == "google":
        from cothis.ai.google import GoogleProvider

        return GoogleProvider(api_key=api_key, api_base=api_base)
    raise ValueError(
        f"Unknown provider {provider!r}. Known: {', '.join(_KNOWN_PROVIDERS)}."
    )


__all__ = ["AIProvider", "get_provider"]
