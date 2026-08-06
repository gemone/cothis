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
* :class:`ContextBudget` / :class:`PressureLevel` — the context-window
  pressure signal from :mod:`cothis.ai.context_budget`.
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
from cothis.ai.context_budget import ContextBudget, PressureLevel

__all__ = [
    "AIProvider",
    "ContextBudget",
    "InputJSONDelta",
    "MessageResponse",
    "MessageStreamEvent",
    "PressureLevel",
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
