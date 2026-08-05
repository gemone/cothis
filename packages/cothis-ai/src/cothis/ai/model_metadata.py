"""Resolve model ``max_tokens`` from bundled litellm model metadata.

``Agent`` needs an output-token cap that matches the model: too small and a
generation can be cut off mid-tool-call (the unpaired-``tool_use`` class of
bug); too large and the provider rejects the request. The right cap is the
model's own ``max_output_tokens``, which litellm publishes in
``model_prices_and_context_window.json``. We bundle that file
(``cothis/data/model_prices.json``) and read it here — no network call at
runtime, no extra dependency.

cothis: matching is exact-key only (``model``, then ``{provider}/{model}``).
litellm provider names diverge from cothis's (e.g. ``together_ai`` vs
``together``), so we deliberately do NOT fuzzy-match on the
``litellm_provider`` field — a mismatch falls back to ``_FALLBACK_MAX_TOKENS``
and the user overrides via ``--max-tokens`` / ``COTHIS_MAX_TOKENS``. Upgrade
path: a provider-name map if mismatch rates hurt in practice.

The JSON is ~1.6 MB and the resolver is called once per ``Agent`` (cached on
the instance); we still cache the parsed dict at module level via
:func:`functools.cache` so repeated ``Agent`` constructions in the same
process (e.g. tests) don't re-parse it.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any

# Anthropic's own default when a model isn't in litellm; matches the value
# hardcoded in #31 before this slice landed, so behaviour is unchanged for
# unknown models.
_FALLBACK_MAX_TOKENS = 8192


@cache
def _metadata() -> dict[str, dict[str, Any]]:
    """Load and cache the bundled litellm model-prices JSON.

    Cached at module level (not on the Agent) so multiple ``Agent`` instances
    in one process share one parse — the file is ~1.6 MB and parsing is the
    only non-trivial work here.
    """
    path = files("cothis.ai.data") / "model_prices.json"
    return json.loads(path.read_text())


def _entry_max_tokens(entry: dict[str, Any]) -> int | None:
    """Pick the output-token cap from one litellm entry.

    Field precedence: ``max_output_tokens`` first (the modern field); fall
    back to the legacy ``max_tokens`` field ONLY when ``max_input_tokens``
    is also absent. Per litellm's own ``sample_spec.max_tokens`` contract,
    when ``max_input_tokens`` is set the legacy ``max_tokens`` duplicates
    it (the *input* cap) and must not be returned as the output cap (#64).
    Both absent → ``None`` (caller falls back).
    """
    out = entry.get("max_output_tokens")
    if isinstance(out, (int, float)) and out > 0:
        return int(out)
    # cothis: skip the legacy fallback when ``max_input_tokens`` is set —
    # in that case ``max_tokens`` duplicates the input cap, not the
    # output cap (#64).
    if "max_input_tokens" in entry:
        return None
    legacy = entry.get("max_tokens")
    if isinstance(legacy, (int, float)) and legacy > 0:
        return int(legacy)
    return None


def _entry_context_window(entry: dict[str, Any]) -> int | None:
    """Pick the context-window size (input cap) from one litellm entry.

    ``max_input_tokens`` is the modern field; the legacy ``max_tokens``
    duplicates it when both are present (litellm schema). When only the
    legacy field exists it is the input cap. Both absent -> ``None``.
    """
    inp = entry.get("max_input_tokens")
    if isinstance(inp, (int, float)) and inp > 0:
        return int(inp)
    # Only fall back to legacy when there is no modern input field at all;
    # otherwise the legacy value is the output cap (see _entry_max_tokens).
    if "max_input_tokens" not in entry:
        legacy = entry.get("max_tokens")
        if isinstance(legacy, (int, float)) and legacy > 0:
            return int(legacy)
    return None


def model_info(model: str, provider: str) -> dict[str, Any]:
    """Return the structured limits advertised for one model.

    A sibling to :func:`resolve_max_tokens` that returns both limits as a
    dict (``maxOutputTokens``, ``contextWindow``) instead of a single int.
    Reuses the same cached :func:`_metadata` and the same
    ``(model, f'{provider}/{model}')`` key walk, picking the first matching
    entry so the advertised pair is consistent with the resolver's
    precedence. Unknown fields are ``None`` — no invented numbers, no crash.
    Callers that need a concrete output cap should still use
    :func:`resolve_max_tokens` (which falls back to ``_FALLBACK_MAX_TOKENS``
    for unknown models); this function is for honest *advertisement*, where
    ``None`` is the truthful "we don't know".
    """
    data = _metadata()
    entry: dict[str, Any] | None = None
    for key in (model, f"{provider}/{model}"):
        candidate = data.get(key)
        if isinstance(candidate, dict):
            entry = candidate
            break
    if entry is None:
        return {"maxOutputTokens": None, "contextWindow": None}
    return {
        "maxOutputTokens": _entry_max_tokens(entry),
        "contextWindow": _entry_context_window(entry),
    }


def resolve_max_tokens(
    model: str,
    provider: str,
    override: int | None = None,
) -> int:
    """Resolve the ``max_tokens`` to pass to ``amessages`` for this model.

    Precedence (highest first):

    1. ``override`` — set explicitly via ``--max-tokens`` / ``COTHIS_MAX_TOKENS``.
       The user knows better than the metadata; always wins.
    2. Exact ``model`` key in litellm (e.g. ``claude-sonnet-4-5``,
       ``gpt-4.1-mini``).
    3. ``{provider}/{model}`` key (e.g. ``openrouter/openai/gpt-oss-120b``,
       ``mistral/mistral-small-latest``) — the form litellm uses for providers
       that prefix model ids.
    4. ``_FALLBACK_MAX_TOKENS`` (8192) — unknown model.

    The resolved value is the model's real ``max_output_tokens`` (e.g.
    gpt-oss-120b → 32768). A provider may reject it if the account's
    prepaid balance can't cover that many tokens at the model's rate
    (openrouter returns HTTP 402) — that's an account/credits issue, not a
    cothis bug; lower with ``--max-tokens`` if it happens.

    Negative or zero ``override`` is treated as "not set" so a stray ``0``
    from a misconfigured env var doesn't silently disable generation.
    """
    if override is not None and override > 0:
        return override

    data = _metadata()
    for key in (model, f"{provider}/{model}"):
        entry = data.get(key)
        if isinstance(entry, dict):
            resolved = _entry_max_tokens(entry)
            if resolved is not None:
                return resolved
    return _FALLBACK_MAX_TOKENS
