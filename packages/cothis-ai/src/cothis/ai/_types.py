"""Canonical AI-layer types — aliases onto the anthropic SDK.

The agent loop speaks the Anthropic Messages API natively (its stream
accumulator narrows ``anthropic.types.RawMessageStreamEvent`` by
``isinstance``). Each provider in :mod:`cothis.ai` normalises into this
vocabulary so the agent's accumulator stays provider-agnostic. We do not
define a parallel set of pydantic models — we reuse the anthropic SDK's
types to avoid drift and keep the Anthropic provider a zero-translation
pass-through.
"""

from __future__ import annotations

import anthropic.types as _ant

# Non-stream response. ``anthropic.types.Message`` exposes the identical
# shape the upstream facade library's ``MessageResponse`` exposed (``id``, ``model``, ``role``,
# ``type``, ``content``, ``stop_reason``, ``usage``).
type MessageResponse = _ant.Message

# Per-turn usage. Fields read by ``agent.py``: ``input_tokens``,
# ``output_tokens``, ``cache_read_input_tokens``,
# ``cache_creation_input_tokens``.
type Usage = _ant.Usage

# Content blocks the agent accumulator already handles.
type TextBlock = _ant.TextBlock
type ToolUseBlock = _ant.ToolUseBlock
type ThinkingBlock = _ant.ThinkingBlock

# Deltas carried on ``RawContentBlockDeltaEvent.delta``.
type TextDelta = _ant.TextDelta
type ThinkingDelta = _ant.ThinkingDelta
type SignatureDelta = _ant.SignatureDelta
type InputJSONDelta = _ant.InputJSONDelta

# The stream-event union the agent narrows by ``isinstance``. This is what
# a prior provider-abstraction library re-exported this as ``MessageStreamEvent``; we alias the anthropic
# union directly.
type MessageStreamEvent = _ant.RawMessageStreamEvent

# ``stop_reason`` is a string ``Literal`` in the anthropic SDK; re-export.
type StopReason = _ant.StopReason

__all__ = [
    "InputJSONDelta",
    "MessageResponse",
    "MessageStreamEvent",
    "SignatureDelta",
    "StopReason",
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ThinkingDelta",
    "ToolUseBlock",
    "Usage",
]
