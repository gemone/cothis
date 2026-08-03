"""cothis.ai — direct-provider-SDK AI layer.

A small provider abstraction over the Anthropic / OpenAI / Google GenAI /
OpenRouter SDKs. The agent loop speaks the Anthropic Messages API
natively; each provider normalises into that shape via
:meth:`AIProvider.amessages`.

Public surface (frozen — other modules depend on these names):

* :func:`get_provider` — factory resolving a provider key to an instance.
* :class:`AIProvider` — the Protocol every provider implements.
* the canonical type aliases from :mod:`cothis.ai._types`
  (``MessageResponse``, ``MessageStreamEvent``, ``Usage``, ``TextDelta``,
  ``InputJSONDelta``, ...).
"""

from __future__ import annotations

from cothis.ai._types import (
    InputJSONDelta,
    MessageResponse,
    MessageStreamEvent,
    SignatureDelta,
    StopReason,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    Usage,
)
from cothis.ai.base import AIProvider, get_provider

__all__ = [
    "AIProvider",
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
    "get_provider",
]
