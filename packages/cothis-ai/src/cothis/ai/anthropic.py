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

from typing import TYPE_CHECKING, Any, Literal, overload

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
            kwargs["system"] = _ensure_system_cache_breakpoint(system)
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
    ) -> MessageResponse | AsyncIterator[MessageStreamEvent]:
        """Forward to ``messages.create`` / ``messages.stream`` verbatim.

        ``session_id`` is accepted for Protocol parity but ignored: Anthropic
        routes prompt caching via the per-block ``cache_control`` breakpoint
        this provider attaches in :func:`_ensure_system_cache_breakpoint`, not
        via a per-session key.
        """
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


def _ensure_system_cache_breakpoint(
    system: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Idempotently attach ``cache_control: {type: ephemeral}`` to the last system block.

    Anthropic's prompt cache covers the prefix from the start up to AND
    including the block carrying ``cache_control``. The stable prefix for a
    cothis turn is the whole system block list, so the breakpoint belongs on
    the LAST block. The agent's prompt builder may already attach one (when
    it built the list via ``_assemble_system``); this provider-level pass
    guarantees the breakpoint is present regardless of the caller, so the
    cache fires even when ``_assemble_system`` is bypassed.

    Idempotent: when the last block already carries ``cache_control`` it is
    left untouched (no double-write). The last block is shallow-copied rather
    than mutated in place — the agent reuses the same system list across
    turns, so in-place mutation would compound ``cache_control`` writes
    silently across the conversation.
    """
    if not system:
        return system
    last = system[-1]
    if not isinstance(last, dict):
        # A non-dict block (a bare string or other unsupported shape) is a
        # caller contract violation — the type hint is ``list[dict]`` and the
        # Anthropic SDK would reject it on the wire. Don't attach a
        # breakpoint to it; leave the list untouched and let the SDK raise
        # the authoritative error rather than a ``TypeError`` here.
        return system
    if last.get("cache_control") is not None:
        return system  # already marked — leave untouched (idempotent)
    copied = list(system)
    copied[-1] = {**last, "cache_control": {"type": "ephemeral"}}
    return copied


__all__ = ["AnthropicProvider"]
