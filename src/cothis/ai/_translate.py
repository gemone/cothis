"""Pure translation helpers shared by the non-Anthropic providers.

These helpers convert between the canonical Anthropic Messages shape
(which the agent loop speaks) and each foreign provider's wire format
(OpenAI Chat Completions, Google GenAI). They are pure — no I/O, no SDK
clients — and import SDK types lazily inside the functions that need them,
so importing :mod:`cothis.ai._translate` never pulls a provider SDK at
module-load time.

Inputs are plain ``dict`` / ``list`` (the shapes the agent already passes
to ``amessages``); outputs are plain ``dict`` / ``list`` ready to hand to
the foreign SDK constructor, or built SDK model objects where labelled.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# OpenAI finish-reason -> Anthropic stop_reason
# ---------------------------------------------------------------------------

_OPENAI_FINISH_MAP: dict[str | None, str] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
    None: "end_turn",
}


def map_openai_finish_reason(finish_reason: str | None) -> str:
    """Map an OpenAI ``finish_reason`` onto an Anthropic ``stop_reason``.

    Unknown reasons fall back to ``"end_turn"`` so an unexpected provider
    finish signal (e.g. an OpenRouter backend-specific code) never breaks
    the agent's turn loop.
    """
    return _OPENAI_FINISH_MAP.get(finish_reason, "end_turn")


# ---------------------------------------------------------------------------
# Google finish-reason -> Anthropic stop_reason
# ---------------------------------------------------------------------------

_GOOGLE_FINISH_MAP: dict[str, str] = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "end_turn",
    "RECITATION": "end_turn",
    "LANGUAGE": "end_turn",
    "OTHER": "end_turn",
    "BLOCKLIST": "end_turn",
    "PROHIBITED_CONTENT": "end_turn",
    "SPII": "end_turn",
    "MALFORMED_FUNCTION_CALL": "tool_use",
    "FINISH_REASON_UNSPECIFIED": "end_turn",
}


def map_google_finish_reason(finish_reason: Any) -> str:
    """Map a Google GenAI ``FinishReason`` (enum or name) onto Anthropic.

    Accepts either the ``google.genai.types.FinishReason`` enum or its
    string ``name``; unknown values fall back to ``"end_turn"``.
    """
    name = getattr(finish_reason, "name", finish_reason)
    return _GOOGLE_FINISH_MAP.get(str(name), "end_turn")


# ---------------------------------------------------------------------------
# Anthropic system -> OpenAI system message text
# ---------------------------------------------------------------------------


def anthropic_system_to_text(system: list[dict[str, Any]] | None) -> str | None:
    """Flatten an Anthropic ``system`` block list to a single string.

    Anthropic's ``system`` is a list of content blocks (typically
    ``{"type": "text", "text": ...}``, optionally carrying
    ``cache_control``). OpenAI Chat Completions has no ``system`` param, so
    we concatenate the text blocks into one prompt. Non-text blocks are
    skipped. Returns ``None`` for an empty/absent system so the caller can
    omit the message entirely.
    """
    if not system:
        return None
    parts: list[str] = []
    for block in system:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
        elif block.get("text"):  # tolerate bare {"text": ...} blocks
            parts.append(str(block["text"]))
    return "\n\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Anthropic messages -> OpenAI chat messages
# ---------------------------------------------------------------------------


def _flatten_text_content(content: Any) -> str:
    """Reduce an Anthropic message ``content`` to a plain text string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text")
                if txt:
                    out.append(str(txt))
        return "".join(out)
    return ""


def anthropic_messages_to_openai(
    messages: list[dict[str, Any]],
    system: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Translate an Anthropic Messages conversation to OpenAI Chat format.

    - ``system`` block list -> a leading ``{"role": "system", ...}`` message.
    - assistant ``tool_use`` block -> an OpenAI assistant ``tool_calls`` entry
      (arguments serialised to JSON).
    - user ``tool_result`` block -> an OpenAI ``{"role": "tool", ...}`` message.
    - ``text`` / ``thinking`` blocks -> the message ``content`` string
      (``thinking`` blocks are dropped for OpenAI; they are reasoning trace,
      not assistant output).

    The mapping is lossy at the edges (no thinking blocks, no
    ``cache_control``) but preserves the conversation structure the agent
    loop needs: alternating user/assistant turns, tool-call pairing, and
    tool results routed back to their calls.
    """
    out: list[dict[str, Any]] = []

    sys_text = anthropic_system_to_text(system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            # A user turn may carry tool_result blocks (response to a prior
            # assistant tool_use) and/or ordinary text. Split each tool_result
            # into its own ``role: tool`` message; remaining text becomes a
            # single user message.
            tool_results: list[dict[str, Any]] = []
            text_parts: list[str] = []
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "tool_result":
                        tool_results.append(_tool_result_to_openai(block))
                    elif btype == "text" and block.get("text"):
                        text_parts.append(str(block["text"]))
            elif isinstance(content, str):
                text_parts.append(content)
            for tr in tool_results:
                out.append(tr)
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})
            elif not tool_results:
                # No usable content — preserve the turn with an empty string
                # so the alternation invariant (user/assistant/user) holds.
                out.append({"role": "user", "content": ""})
        elif role == "assistant":
            out.append(_assistant_to_openai(content))
        else:
            # Pass through any pre-shaped message verbatim.
            out.append(
                {"role": role or "user", "content": _flatten_text_content(content)}
            )
    return out


def _assistant_to_openai(content: Any) -> dict[str, Any]:
    """Translate an Anthropic assistant content list to an OpenAI message."""
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and block.get("text"):
                text_parts.append(str(block["text"]))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": _json_dumps(block.get("input", {})),
                        },
                    }
                )
            # thinking blocks: dropped (OpenAI has no equivalent).
    msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_result_to_openai(block: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic ``tool_result`` block to an OpenAI tool message."""
    raw = block.get("content")
    if isinstance(raw, list):
        # Anthropic nests content blocks inside tool_result; flatten text.
        text = "".join(
            str(b.get("text", ""))
            for b in raw
            if isinstance(b, dict) and b.get("type") == "text"
        )
    elif isinstance(raw, str):
        text = raw
    else:
        text = _json_dumps(raw) if raw is not None else ""
    return {
        "role": "tool",
        "tool_call_id": block.get("tool_use_id", ""),
        "content": text,
    }


# ---------------------------------------------------------------------------
# Anthropic tool schemas -> OpenAI tools
# ---------------------------------------------------------------------------


def anthropic_tools_to_openai(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Translate Anthropic tool schemas to OpenAI ``tools`` shape.

    Anthropic: ``{"name", "description", "input_schema"}`` (a JSON Schema).
    OpenAI:    ``{"type": "function", "function": {"name", "description",
                "parameters": <JSONSchema>}}``.
    """
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        fn = {
            "name": tool.get("name", ""),
            "parameters": tool.get("input_schema")
            or {"type": "object", "properties": {}},
        }
        if tool.get("description"):
            fn["description"] = tool["description"]
        out.append({"type": "function", "function": fn})
    return out


# ---------------------------------------------------------------------------
# Anthropic messages -> Google GenAI contents
# ---------------------------------------------------------------------------


def anthropic_messages_to_google_contents(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate an Anthropic conversation to Google GenAI ``contents``.

    Each returned item is a plain dict ``{"role": "user"|"model",
    "parts": [...]}`` matching the shape ``google.genai.types.Content``
    expects. Roles: Anthropic ``user`` -> Google ``user``; Anthropic
    ``assistant`` -> Google ``model``.

    Part kinds emitted:
    - text block -> ``{"text": ...}``
    - tool_use block -> ``{"function_call": {"id": ..., "name": ..., "args": ...}}``
    - tool_result block -> ``{"function_response": {"id": ..., "name": ...,
      "response": ...}}``
    """
    contents: list[dict[str, Any]] = []
    for msg in messages:
        arole = msg.get("role")
        role = "model" if arole == "assistant" else "user"
        content = msg.get("content")
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text" and block.get("text"):
                    parts.append({"text": str(block["text"])})
                elif btype == "tool_use":
                    fc: dict[str, Any] = {
                        "name": block.get("name", ""),
                        "args": block.get("input") or {},
                    }
                    if block.get("id"):
                        fc["id"] = block["id"]
                    parts.append({"function_call": fc})
                elif btype == "tool_result":
                    fr: dict[str, Any] = {
                        "name": block.get("name", ""),
                        "response": _tool_result_payload(block.get("content")),
                    }
                    if block.get("tool_use_id"):
                        fr["id"] = block["tool_use_id"]
                    parts.append({"function_response": fr})
                # thinking blocks: no Google equivalent in contents input.
        if parts:
            contents.append({"role": role, "parts": parts})
    return contents


def _tool_result_payload(content: Any) -> dict[str, Any]:
    """Normalise an Anthropic ``tool_result.content`` to a Google response dict."""
    if content is None:
        return {"output": ""}
    if isinstance(content, str):
        return {"output": content}
    if isinstance(content, list):
        text = "".join(
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
        return {"output": text}
    return {"output": _json_dumps(content)}


def anthropic_tools_to_google(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Translate Anthropic tool schemas to Google ``function_declarations``.

    Returns a list of ``{"name", "description", "parameters"}`` dicts ready
    to wrap in ``google.genai.types.Tool(function_declarations=[...])``.
    """
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        decl: dict[str, Any] = {
            "name": tool.get("name", ""),
            "parameters": tool.get("input_schema")
            or {"type": "object", "properties": {}},
        }
        if tool.get("description"):
            decl["description"] = tool["description"]
        out.append(decl)
    return out


# ---------------------------------------------------------------------------
# JSON helper (imported lazily so module import is SDK-free)
# ---------------------------------------------------------------------------


def _json_dumps(obj: Any) -> str:
    """Serialise ``obj`` to a compact JSON string (used for tool args)."""
    import json

    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)
