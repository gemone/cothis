"""Google GenAI provider — translation + Anthropic-compatible synthesis.

Calls ``google.genai.Client.aio.models.generate_content`` (and the
``..._stream`` variant). Translates the Anthropic Messages conversation to
Google's ``contents`` + ``function_declarations`` shape, and synthesises
``anthropic.types.RawMessageStreamEvent`` objects from the Google stream —
the same canonical lifecycle the agent accumulator expects.

The Google client is constructed lazily on first use so
:func:`cothis.ai.get_provider` works without an API key set.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, overload

from cothis.ai._translate import (
    anthropic_messages_to_google_contents,
    anthropic_tools_to_google,
    map_google_finish_reason,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cothis.ai._types import MessageResponse, MessageStreamEvent


class GoogleProvider:
    """Google GenAI (Gemini) adapter with Anthropic-shaped I/O."""

    def __init__(self, *, api_key: str | None, api_base: str | None) -> None:
        self._api_key = api_key
        self._api_base = api_base
        self._client: Any = None

    # ------------------------------------------------------------------ client
    def _get_client(self) -> Any:
        """Lazily build the ``google.genai.Client`` (memoised)."""
        if self._client is None:
            from google import genai

            kwargs: dict[str, Any] = {}
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            # ``api_base`` is exposed via http_options on the unified SDK.
            if self._api_base is not None:
                from google.genai import types

                kwargs["http_options"] = types.HttpOptions(base_url=self._api_base)
            self._client = genai.Client(**kwargs)
        return self._client

    # ----------------------------------------------------------- request build
    def _build_config(
        self,
        *,
        max_tokens: int,
        system: list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        from google.genai import types

        config_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
        sys_text = _system_to_text(system)
        if sys_text:
            config_kwargs["system_instruction"] = sys_text
        fn_decls = anthropic_tools_to_google(tools)
        if fn_decls:
            config_kwargs["tools"] = [types.Tool(function_declarations=fn_decls)]
        return types.GenerateContentConfig(**config_kwargs)

    # ================================================================ amessages
    @overload
    async def amessages(self, *, model: str, messages: list[dict[str, Any]], max_tokens: int, system: list[dict[str, Any]] | None = None, tools: list[dict[str, Any]] | None = None, stream: Literal[False] = False) -> MessageResponse: ...
    @overload
    async def amessages(self, *, model: str, messages: list[dict[str, Any]], max_tokens: int, system: list[dict[str, Any]] | None = None, tools: list[dict[str, Any]] | None = None, stream: Literal[True]) -> AsyncIterator[MessageStreamEvent]: ...
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
        contents = anthropic_messages_to_google_contents(messages)
        config = self._build_config(max_tokens=max_tokens, system=system, tools=tools)
        client = self._get_client()
        if stream:
            response_stream = await client.aio.models.generate_content_stream(
                model=model, contents=contents, config=config
            )
            return _synthesise_stream(response_stream, model=model)
        response = await client.aio.models.generate_content(
            model=model, contents=contents, config=config
        )
        return _response_to_message(response, model=model)


def _system_to_text(system: list[dict[str, Any]] | None) -> str | None:
    """Flatten the Anthropic system block list to a string for Google."""
    if not system:
        return None
    parts: list[str] = []
    for block in system:
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and block.get("text")
        ):
            parts.append(str(block["text"]))
    return "\n\n".join(parts) if parts else None


# ===========================================================================
# Non-stream: GenerateContentResponse -> anthropic.types.Message
# ===========================================================================


def _iter_parts(response: Any) -> list[Any]:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    return list(parts)


def _response_to_message(response: Any, *, model: str) -> Any:
    """Translate a Google response into an ``anthropic.types.Message``."""
    from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

    content: list[Any] = []
    for part in _iter_parts(response):
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            content.append(TextBlock(type="text", text=text))
        fc = getattr(part, "function_call", None)
        if fc is not None:
            content.append(
                ToolUseBlock(
                    type="tool_use",
                    id=_tool_id(fc),
                    name=getattr(fc, "name", "") or "",
                    input=_args_to_dict(getattr(fc, "args", None)),
                )
            )

    finish = _finish_reason(response)
    usage_meta = getattr(response, "usage_metadata", None)
    usage = Usage(
        input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
    )
    return Message(
        id=getattr(response, "response_id", "") or "",
        model=getattr(response, "model_version", None) or model,
        role="assistant",
        type="message",
        content=content,
        stop_reason=map_google_finish_reason(finish),
        usage=usage,
    )


# ===========================================================================
# Stream: Google response stream -> anthropic.types.RawMessageStreamEvent
# ===========================================================================


async def _synthesise_stream(
    response_stream: Any,
    *,
    model: str,
) -> AsyncIterator[MessageStreamEvent]:
    """Yield ``RawMessageStreamEvent``s synthesised from the Google stream.

    Lifecycle: one ``message_start``; per text part a text-block lifecycle
    (start -> TextDelta -> stop); per function_call part a tool_use block
    with one complete-JSON ``InputJSONDelta``; one ``message_delta`` carrying
    the mapped ``stop_reason``; one ``message_stop`` (latched).
    """
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
        ToolUseBlock,
        Usage,
    )
    from anthropic.types.message import Message
    from anthropic.types.raw_message_delta_event import Delta

    next_index = 0
    text_block_index: int | None = None
    started = False
    stopped = False

    async for response in response_stream:
        if not started:
            started = True
            yield RawMessageStartEvent(
                type="message_start",
                message=Message(
                    id=getattr(response, "response_id", "") or "",
                    model=getattr(response, "model_version", None) or model,
                    role="assistant",
                    type="message",
                    content=[],
                    stop_reason=None,
                    usage=Usage(input_tokens=0, output_tokens=0),
                ),
            )

        for part in _iter_parts(response):
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                if text_block_index is None:
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
                    delta=TextDelta(type="text_delta", text=text),
                )

            fc = getattr(part, "function_call", None)
            if fc is not None:
                # Close any open text block before opening a tool block.
                if text_block_index is not None:
                    yield RawContentBlockStopEvent(
                        type="content_block_stop", index=text_block_index
                    )
                    text_block_index = None
                block_index = next_index
                next_index += 1
                yield RawContentBlockStartEvent(
                    type="content_block_start",
                    index=block_index,
                    content_block=ToolUseBlock(
                        type="tool_use",
                        id=_tool_id(fc),
                        name=getattr(fc, "name", "") or "",
                        input={},
                    ),
                )
                # Google delivers parsed args in one shot — emit the full
                # JSON as a single InputJSONDelta so the agent's accumulator
                # parses a complete value at content_block_stop.
                yield RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=block_index,
                    delta=InputJSONDelta(
                        type="input_json_delta",
                        partial_json=_args_to_json(getattr(fc, "args", None)),
                    ),
                )
                yield RawContentBlockStopEvent(
                    type="content_block_stop", index=block_index
                )

        finish = _finish_reason(response)
        if finish is not None and not stopped:
            stopped = True
            if text_block_index is not None:
                yield RawContentBlockStopEvent(
                    type="content_block_stop", index=text_block_index
                )
                text_block_index = None
            usage_meta = getattr(response, "usage_metadata", None)
            usage = MessageDeltaUsage(
                input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
            )
            yield RawMessageDeltaEvent(
                type="message_delta",
                delta=Delta(stop_reason=map_google_finish_reason(finish)),
                usage=usage,
            )
            yield RawMessageStopEvent(type="message_stop")

    if started and not stopped:
        # Stream ended without an explicit finish — close dangling blocks
        # and emit the stop pair so the latch fires exactly once.
        if text_block_index is not None:
            yield RawContentBlockStopEvent(
                type="content_block_stop", index=text_block_index
            )
        yield RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason="end_turn"),
            usage=MessageDeltaUsage(input_tokens=0, output_tokens=0),
        )
        yield RawMessageStopEvent(type="message_stop")


# ===========================================================================
# Google part helpers
# ===========================================================================


def _finish_reason(response: Any) -> Any:
    """Extract the first candidate's finish reason (or ``None``)."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    return getattr(candidates[0], "finish_reason", None)


def _tool_id(function_call: Any) -> str:
    """Stable tool-use id from a Google ``FunctionCall``."""
    fc_id = getattr(function_call, "id", None)
    if fc_id:
        return str(fc_id)
    # Google does not always populate ``id``; derive one from the call name
    # plus a short random suffix so parallel calls to the same tool name in
    # one turn get distinct ids (the agent keys tool-result pairing by id)
    # while still round-tripping tool_result pairing on the next turn.
    name = getattr(function_call, "name", "tool")
    return f"google_{name}_{uuid.uuid4().hex[:8]}"


def _args_to_dict(args: Any) -> dict[str, Any]:
    """Coerce a Google ``FunctionCall.args`` (proto map) to a plain dict."""
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    # Proto Struct / MapComposite — fall back to dict() conversion.
    try:
        return dict(args)
    except TypeError:
        return {}


def _args_to_json(args: Any) -> str:
    """Serialise ``FunctionCall.args`` to a JSON string for ``InputJSONDelta``."""
    import json

    return json.dumps(
        _args_to_dict(args), separators=(",", ":"), ensure_ascii=False, default=str
    )


__all__ = ["GoogleProvider"]
