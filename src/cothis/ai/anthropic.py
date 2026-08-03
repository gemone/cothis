"""Anthropic provider — native pass-through.

The agent already speaks the Anthropic Messages API, so this provider is a
near-zero-translation adapter over ``anthropic.AsyncAnthropic``. It forwards
``amessages`` kwargs directly to ``messages.create`` (non-stream) or
``messages.stream`` (stream) and returns the SDK's native ``Message`` /
``RawMessageStreamEvent`` objects unchanged.

The SDK client is constructed lazily on first use so :func:`get_provider`
works without an API key or ``ANTHROPIC_API_KEY`` env var set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cothis.ai._types import MessageResponse, MessageStreamEvent


class AnthropicProvider:
    """Direct ``anthropic.AsyncAnthropic`` adapter — native wire format."""

    def __init__(self, *, api_key: str | None, api_base: str | None) -> None:
        self._api_key = api_key
        self._api_base = api_base
        self._client: Any = None  # lazily constructed

    # ------------------------------------------------------------------ client
    def _get_client(self) -> Any:
        """Lazily build the ``AsyncAnthropic`` client (memoised).

        Only forward ``api_key`` / ``base_url`` when the caller supplied
        them; the SDK falls back to its env-var discovery otherwise.
        """
        if self._client is None:
            from anthropic import AsyncAnthropic

            kwargs: dict[str, Any] = {}
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            if self._api_base is not None:
                kwargs["base_url"] = self._api_base
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    # ----------------------------------------------------------------- request
    def _build_kwargs(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        return kwargs

    async def _stream(
        self, kwargs: dict[str, Any]
    ) -> AsyncIterator[MessageStreamEvent]:
        """Yield ``RawMessageStreamEvent``s from the SDK streaming helper."""
        client = self._get_client()
        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                yield event

    async def amessages(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> MessageResponse | AsyncIterator[MessageStreamEvent]:
        """Forward to ``messages.create`` / ``messages.stream`` verbatim."""
        kwargs = self._build_kwargs(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
        )
        if stream:
            return self._stream(kwargs)
        client = self._get_client()
        return await client.messages.create(**kwargs)


__all__ = ["AnthropicProvider"]
