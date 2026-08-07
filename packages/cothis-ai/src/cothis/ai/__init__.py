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
* :class:`SummaryTarget` / :class:`SummarisationRequest` /
  :func:`resolve_summary_model` / :func:`build_summarisation_request` — the
  compaction summariser selector + prompt builder from
  :mod:`cothis.ai.compaction` (slice A; pure building blocks, not yet wired
  into the run loop).
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
from cothis.ai.compaction import (
    SummarisationRequest,
    SummaryTarget,
    build_summarisation_request,
    resolve_summary_model,
)
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
    "SummarisationRequest",
    "SummaryTarget",
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ThinkingDelta",
    "ToolUseBlock",
    "Usage",
    "build_summarisation_request",
    "get_provider",
    "resolve_summary_model",
]
