"""OpenAI provider — translation + Anthropic-compatible stream synthesis.

Calls ``openai.AsyncOpenAI.chat.completions``. The agent speaks the
Anthropic Messages API, so this provider:

1. Translates the Anthropic ``messages`` / ``system`` / ``tools`` to the
   OpenAI Chat Completions shape (shared pure helpers in
   :mod:`cothis.ai._translate`).
2. For non-stream calls, translates the returned ``ChatCompletion`` back to
   an ``anthropic.types.Message``.
3. For stream calls, synthesises ``anthropic.types.RawMessageStreamEvent``
   objects from the OpenAI ``ChatCompletionChunk`` stream. The synthesised
   lifecycle is the one the agent accumulator expects: one ``message_start``,
   one ``content_block_*`` lifecycle per block, one ``message_delta`` with
   the mapped ``stop_reason``, and exactly one ``message_stop``.

The SDK client is constructed lazily on first use.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Literal, overload

from cothis.ai._translate import (
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    map_openai_finish_reason,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cothis.ai._types import MessageResponse, MessageStreamEvent


class OpenAIProvider:
    """OpenAI Chat Completions adapter with Anthropic-shaped I/O."""

    def __init__(self, *, api_key: str | None, api_base: str | None) -> None:
        self._api_key = api_key
        self._api_base = api_base
        self._client: Any = None

    # ------------------------------------------------------------------ client
    def _get_client(self) -> Any:
        """Lazily build the ``AsyncOpenAI`` client (memoised)."""
        if self._client is None:
            from openai import AsyncOpenAI

            kwargs: dict[str, Any] = {}
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            if self._api_base is not None:
                kwargs["base_url"] = self._api_base
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    # ----------------------------------------------------------- request build
    def _build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages_to_openai(messages, system),
            "max_tokens": max_tokens,
            "tools": anthropic_tools_to_openai(tools),
        }
        # Route OpenAI's automatic prefix caching to this conversation via a
        # per-session key. Gated to real OpenAI only (``api_base is None``):
        # OpenAI-compatible backends (OpenRouter, DeepSeek, Groq, Mistral) 400
        # on unknown top-level params, and the ``OpenAIProvider`` subclasses
        # used for them always set ``api_base``. When no session is attached
        # (the ephemeral ``ask`` path) the key is omitted — OpenAI still
        # auto-caches by prefix.
        if session_id is not None and self._api_base is None:
            request["prompt_cache_key"] = _derive_prompt_cache_key(session_id)
        return request

    # ================================================================ amessages
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
        request = self._build_request(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            session_id=session_id,
        )
        client = self._get_client()
        if stream:
            # The SDK returns an AsyncStream[ChatCompletionChunk] when
            # stream=True; pass it to the synthesiser.
            chunk_stream = await client.chat.completions.create(**request, stream=True)
            return _synthesise_stream(chunk_stream, model=model)
        completion = await client.chat.completions.create(**request)
        return _completion_to_message(completion, model=model)


def _derive_prompt_cache_key(session_id: str) -> str:
    """Derive a stable, length-bounded ``prompt_cache_key`` from a session id.

    ``sha256(session_id).hexdigest()[:32]`` — 32 hex chars, well within
    OpenAI's key length bound, deterministic across process restarts (the
    session id is durable in SQLite, so a hash of it is stable too). The
    hash decouples the cache key from any future session-id format change
    and guarantees the length bound unconditionally, even if a future
    session id is longer than today's ``uuid4().hex`` (32 url-safe chars
    that would also fit raw).
    """
    return hashlib.sha256(session_id.encode()).hexdigest()[:32]


# ===========================================================================
# Non-stream: ChatCompletion -> anthropic.types.Message
# ===========================================================================


def _completion_to_message(completion: Any, *, model: str) -> Any:
    """Translate an OpenAI ``ChatCompletion`` to an ``anthropic.types.Message``."""
    from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

    choice = completion.choices[0] if completion.choices else None
    msg = getattr(choice, "message", None) if choice is not None else None
    content: list[Any] = []
    if msg is not None:
        text = getattr(msg, "content", None)
        if text:
            content.append(TextBlock(type="text", text=text))
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            args = getattr(fn, "arguments", "{}") if fn is not None else "{}"
            content.append(
                ToolUseBlock(
                    type="tool_use",
                    id=getattr(tc, "id", "") or "",
                    name=getattr(fn, "name", "") if fn is not None else "",
                    input=_parse_json_args(args),
                )
            )

    finish = getattr(choice, "finish_reason", None) if choice is not None else None
    usage_obj = getattr(completion, "usage", None)
    usage = Usage(
        input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
    )
    return Message(
        id=getattr(completion, "id", "") or "",
        model=getattr(completion, "model", None) or model,
        role="assistant",
        type="message",
        content=content,
        stop_reason=map_openai_finish_reason(finish),
        usage=usage,
    )


def _parse_json_args(raw: str) -> dict[str, Any]:
    import json

    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ===========================================================================
# Stream: ChatCompletionChunk stream -> anthropic.types.RawMessageStreamEvent
# ===========================================================================


async def _synthesise_stream(
    chunk_stream: Any,
    *,
    model: str,
) -> AsyncIterator[MessageStreamEvent]:
    """Yield ``RawMessageStreamEvent``s synthesised from OpenAI chunks.

    Lifecycle emitted (one thinking block when the model streams a reasoning
    trace; one text block per turn; one tool_use block per tool call;
    ``message_stop`` exactly once — duplicate finishes tolerated):
    ``message_start``; for each block ``content_block_start`` ->
    ``content_block_delta``* -> ``content_block_stop``; ``message_delta``
    (carrying the mapped ``stop_reason``); ``message_stop``.

    Reasoning models (o-series, gpt-5-codex) and OpenAI-compatible backends
    stream a reasoning trace as an undocumented ``reasoning_content`` /
    ``reasoning`` extra on the chunk delta; it is captured here as a
    ``ThinkingBlock`` lifecycle so the event stream is provider-agnostic.
    """
    # SDK types are imported lazily so this module is import-safe without
    # the anthropic package installed at import time.
    from anthropic.types import (
        InputJSONDelta,
        MessageDeltaUsage,
        RawContentBlockDeltaEvent,
        RawContentBlockStartEvent,
        RawContentBlockStopEvent,
        RawMessageDeltaEvent,
        RawMessageStartEvent,
        RawMessageStopEvent,
        TextBlock,
        TextDelta,
        ThinkingBlock,
        ThinkingDelta,
        ToolUseBlock,
        Usage,
    )
    from anthropic.types.message import Message

    def _reasoning_piece(delta: Any) -> str:
        """Return the reasoning delta text for ``delta`` (empty if none).

        OpenAI-compatible backends stream the reasoning trace as an
        undocumented extra on ``ChoiceDelta`` — the installed openai SDK
        (2.43.0) does not type it, so it lands in pydantic's ``model_extra``
        and is exposed as a plain attribute. Wire-name conventions differ
        across the backends routed through this provider: DeepSeek / OpenRouter
        send ``reasoning_content``; other OpenAI-compatible backends send
        ``reasoning``. Read both defensively via ``getattr`` (the file's
        universal style for upstream fields) and return the first non-empty
        hit so a single synthesiser covers every backend.
        """
        for _name in ("reasoning_content", "reasoning"):
            _piece = getattr(delta, _name, None)
            if _piece:
                return _piece
        return ""

    message_id: str | None = None
    started = False
    next_index = 0
    text_block_index: int | None = None  # open text block, if any
    thinking_block_index: int | None = None  # open thinking block, if any
    # Maps OpenAI tool_call.index -> our content_block index.
    tool_block_for: dict[int, int] = {}
    stopped = False  # message_stop latch (OpenRouter can duplicate finishes)

    async for chunk in chunk_stream:
        if message_id is None:
            message_id = getattr(chunk, "id", None) or ""
        if not started:
            started = True
            yield RawMessageStartEvent(
                type="message_start",
                message=Message(
                    id=message_id,
                    model=getattr(chunk, "model", None) or model,
                    role="assistant",
                    type="message",
                    content=[],
                    stop_reason=None,
                    usage=Usage(input_tokens=0, output_tokens=0),
                ),
            )

        choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
        if choice is None:
            # Some providers send a final usage-only chunk with no choices.
            continue
        delta = getattr(choice, "delta", None)

        # ---- reasoning (thinking) ------------------------------------------
        # OpenAI reasoning-family models (o-series, gpt-5-codex) and the
        # OpenAI-compatible backends routed through this provider stream a
        # reasoning trace alongside the answer. Capture it as a
        # ``ThinkingBlock`` lifecycle so the AI-layer event stream stays
        # provider-agnostic — the accumulator and TUI thinking-render then
        # apply unchanged. No ``signature``/``SignatureDelta`` path: the
        # Anthropic signature is provider-specific encryption this synthesis
        # has no source for, and ``_translate`` drops thinking blocks before
        # any OpenAI round-trip, so a blank signature is never validated.
        if delta is not None:
            reasoning_piece = _reasoning_piece(delta)
            if reasoning_piece:
                if thinking_block_index is None:
                    # Close any open text block before opening a thinking
                    # block — one content_block lifecycle per block.
                    if text_block_index is not None:
                        yield RawContentBlockStopEvent(
                            type="content_block_stop", index=text_block_index
                        )
                        text_block_index = None
                    thinking_block_index = next_index
                    next_index += 1
                    yield RawContentBlockStartEvent(
                        type="content_block_start",
                        index=thinking_block_index,
                        content_block=ThinkingBlock(
                            type="thinking", thinking="", signature=""
                        ),
                    )
                yield RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=thinking_block_index,
                    delta=ThinkingDelta(
                        type="thinking_delta", thinking=reasoning_piece
                    ),
                )

            # ---- text content ----------------------------------------------
            content_piece = getattr(delta, "content", None)
            if content_piece:
                if text_block_index is None:
                    # Close any open thinking block before opening a text
                    # block — reasoning typically streams before the answer,
                    # but defend against interleaving.
                    if thinking_block_index is not None:
                        yield RawContentBlockStopEvent(
                            type="content_block_stop", index=thinking_block_index
                        )
                        thinking_block_index = None
                    text_block_index = next_index
                    next_index += 1
                    yield RawContentBlockStartEvent(
                        type="content_block_start",
                        index=text_block_index,
                        content_block=TextBlock(type="text", text=""),
                    )
                yield RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=text_block_index,
                    delta=TextDelta(type="text_delta", text=content_piece),
                )

            # ---- tool calls ------------------------------------------------
            tool_calls = getattr(delta, "tool_calls", None) or []
            for tc in tool_calls:
                tc_index = getattr(tc, "index", 0)
                if tc_index not in tool_block_for:
                    # Close any open text/thinking block before opening a tool
                    # block — we keep one text block per turn; tool calls
                    # come after.
                    if text_block_index is not None:
                        yield RawContentBlockStopEvent(
                            type="content_block_stop", index=text_block_index
                        )
                        text_block_index = None
                    if thinking_block_index is not None:
                        yield RawContentBlockStopEvent(
                            type="content_block_stop", index=thinking_block_index
                        )
                        thinking_block_index = None
                    block_index = next_index
                    next_index += 1
                    tool_block_for[tc_index] = block_index
                    fn = getattr(tc, "function", None)
                    yield RawContentBlockStartEvent(
                        type="content_block_start",
                        index=block_index,
                        content_block=ToolUseBlock(
                            type="tool_use",
                            id=getattr(tc, "id", "") or "",
                            name=getattr(fn, "name", "") if fn is not None else "",
                            input={},
                        ),
                    )
                fn = getattr(tc, "function", None)
                args_piece = getattr(fn, "arguments", None) if fn is not None else None
                if args_piece:
                    yield RawContentBlockDeltaEvent(
                        type="content_block_delta",
                        index=tool_block_for[tc_index],
                        delta=InputJSONDelta(
                            type="input_json_delta", partial_json=args_piece
                        ),
                    )

        # ---- finish ---------------------------------------------------------
        finish = getattr(choice, "finish_reason", None)
        if finish is not None and not stopped:
            stopped = True
            # Close whatever blocks are still open (thinking + text + tool_use).
            if thinking_block_index is not None:
                yield RawContentBlockStopEvent(
                    type="content_block_stop", index=thinking_block_index
                )
                thinking_block_index = None
            if text_block_index is not None:
                yield RawContentBlockStopEvent(
                    type="content_block_stop", index=text_block_index
                )
                text_block_index = None
            for idx in tool_block_for.values():
                yield RawContentBlockStopEvent(type="content_block_stop", index=idx)
            tool_block_for.clear()

            usage_obj = getattr(chunk, "usage", None)
            usage = MessageDeltaUsage(
                input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            )
            from anthropic.types.raw_message_delta_event import Delta

            yield RawMessageDeltaEvent(
                type="message_delta",
                delta=Delta(stop_reason=map_openai_finish_reason(finish)),
                usage=usage,
            )
            yield RawMessageStopEvent(type="message_stop")

    # If the stream ended without an explicit finish_reason (some OpenRouter
    # backends truncate), close any dangling blocks and emit the stop pair so
    # the agent's message_stop latch fires exactly once.
    if started and not stopped:
        if thinking_block_index is not None:
            yield RawContentBlockStopEvent(
                type="content_block_stop", index=thinking_block_index
            )
        if text_block_index is not None:
            yield RawContentBlockStopEvent(
                type="content_block_stop", index=text_block_index
            )
        for idx in tool_block_for.values():
            yield RawContentBlockStopEvent(type="content_block_stop", index=idx)
        from anthropic.types.raw_message_delta_event import Delta

        yield RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason="end_turn"),
            usage=MessageDeltaUsage(input_tokens=0, output_tokens=0),
        )
        yield RawMessageStopEvent(type="message_stop")


__all__ = ["OpenAIProvider"]
