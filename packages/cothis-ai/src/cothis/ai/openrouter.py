"""OpenRouter provider — the OpenAI SDK pinned to the OpenRouter base URL.

OpenRouter exposes an OpenAI-compatible Chat Completions API, so we reuse
:class:`cothis.ai.openai.OpenAIProvider` unchanged and only default the
``base_url`` to OpenRouter. The model id (e.g. ``openai/gpt-oss-120b``) is
passed through verbatim — no rewriting needed.
"""

from __future__ import annotations

from cothis.ai.openai import OpenAIProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    """OpenAI-SDK provider pinned to the OpenRouter base URL."""

    def __init__(self, *, api_key: str | None, api_base: str | None) -> None:
        super().__init__(
            api_key=api_key,
            api_base=api_base or OPENROUTER_BASE_URL,
        )


__all__ = ["OpenRouterProvider", "OPENROUTER_BASE_URL"]
