"""Transient-error retry with capped exponential backoff for AI calls.

Both provider call sites in the agent loop (non-stream ``amessages`` and the
streaming variant) used to abort the whole run on a single transient
failure — an HTTP 429, a 5xx, a connection reset, an SDK timeout, or the
Anthropic ``OverloadedError``. This module wraps those call sites with a
small, SDK-agnostic retry layer:

* :func:`is_transient` — classify an exception as transient without
  importing any provider SDK (status-code set, exception class-name
  substring, then a lazily-built ``isinstance`` tuple of whatever SDK
  exception bases happen to be importable).
* :func:`call_with_retry` — await an ``async`` callable, retrying
  transient failures with capped exponential backoff + uniform jitter.
* :func:`retrying_stream` — drive a streaming ``amessages`` call through
  the same retry policy, honouring a strict commit boundary (see below).

Backoff convention
-------------------
:func:`cothis.supervisor.backoff_seconds` uses a jitterless exponential
(``_BACKOFF_FLOOR_S * 2**n`` capped at ``_BACKOFF_CEILING_S``) for
crash-restart scheduling. This module mirrors the floor/ceiling shape but
applies to a *single provider request*, so the ceiling is much smaller
(20s vs 300s) and uniform jitter is added so concurrent agents retrying
the same provider do not thunder-herd on the same wall-clock tick. The
jittered delay is re-capped so the hard ceiling always holds.

Stream commit boundary
-----------------------
A streaming response cannot be safely replayed once the agent has observed
any of it: replaying would re-emit ``message_start``, re-initialise block
state, and double-yield partial content. :func:`retrying_stream` therefore
retries (a) the ``await func(stream=True)`` open and (b) iteration up to
*and including* the first forwarded event. The moment one event is yielded
to the caller the generator is **committed**; any later error propagates
as-is. Before each retry the prior stream generator is ``aclose``d so the
provider's async-with teardown runs cleanly (else the SDK connection leaks
across attempts).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

logger = logging.getLogger("cothis.ai.retry")

# HTTP statuses that are always transient: the canonical rate-limit code
# plus the common 5xx "the server side is briefly broken" set.
TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Exception class-name substrings that name transient failures across
# provider SDKs (Anthropic, OpenAI, Google, OpenRouter) plus the builtin
# ``ConnectionError``. Substring-matched over ``type(exc).__mro__`` so a
# third-party subclass whose name contains one of these is also caught.
# Over-matching here is safe — retrying a non-transient error is
# conservative (it just delays the inevitable raise).
_TRANSIENT_CLASS_NAMES: tuple[str, ...] = (
    "OverloadedError",
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
    "RetryableError",
    "RequestTimeout",
    "ConnectionError",
)

# Backoff schedule. Base/ceiling are module constants (NOT env-exposed)
# — only ``COTHIS_MAX_RETRIES`` is tunable this iteration. The ceiling is
# far below supervisor's 300s because this is a per-request retry, not a
# crash-restart.
_RETRY_BASE_DELAY_S = 0.5
_RETRY_CEILING_DELAY_S = 20.0

_DEFAULT_MAX_RETRIES = 3

# Lazily-built tuple of transient SDK exception bases (Anthropic + OpenAI).
# Filled on the first call to :func:`_lazy_transient_bases`; ``None`` means
# "not yet built". Kept at module level so the (mildly costly) SDK imports
# happen at most once per process and never at module load — preserves the
# providers' lazy-client invariant.
_transient_bases_cache: tuple[type, ...] | None = None

__all__ = [
    "TRANSIENT_STATUS_CODES",
    "call_with_retry",
    "is_transient",
    "resolve_max_retries",
    "retrying_stream",
]


def _backoff_delay(attempt: int) -> float:
    """Capped exponential backoff with non-negative uniform jitter.

    ``base * 2**attempt`` jittered by ``uniform(0, base)`` then re-capped
    at :data:`_RETRY_CEILING_DELAY_S` so the hard ceiling always holds.
    Jitter is an intentional extension of supervisor's jitterless
    convention: concurrent agents retrying one provider would otherwise
    thunder-herd on the same wall-clock tick. ``attempt`` is 0-indexed
    (0 = the delay before the first retry).
    """
    raw = _RETRY_BASE_DELAY_S * (2 ** attempt)
    jitter = random.uniform(0.0, _RETRY_BASE_DELAY_S)
    return min(raw + jitter, _RETRY_CEILING_DELAY_S)


def resolve_max_retries() -> int:
    """Read ``COTHIS_MAX_RETRIES`` (default 3; non-positive → 0, unparseable → default).

    Returns the number of *retries* — so the total attempt count is
    ``resolve_max_retries() + 1``. A non-positive value disables retry
    entirely but the single attempt still runs (and its failure still
    propagates). Unparseable values fall back to the default rather than
    silently disabling the policy, since a stray ``COTHIS_MAX_RETRIES=auto``
    should not quietly strip resilience.
    """
    raw = os.environ.get("COTHIS_MAX_RETRIES", str(_DEFAULT_MAX_RETRIES))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_RETRIES
    if value <= 0:
        return 0
    return value


def _lazy_transient_bases() -> tuple[type, ...]:
    """Build (once) the tuple of transient SDK exception bases.

    Probes whichever of ``anthropic`` / ``openai`` is importable and
    gathers their public transient exception classes by name. Imported
    inside the function (NEVER at module load) to preserve the providers'
    lazy-client invariant — importing :mod:`cothis.ai._retry` must not
    force a provider SDK load. Cached at module level on first call so
    repeated ``is_transient`` checks pay the import cost once.
    """
    global _transient_bases_cache
    if _transient_bases_cache is not None:
        return _transient_bases_cache

    bases: list[type] = []
    for module_name in ("anthropic", "openai"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        for cls_name in _TRANSIENT_CLASS_NAMES:
            cls = getattr(module, cls_name, None)
            if isinstance(cls, type):
                bases.append(cls)
    _transient_bases_cache = tuple(bases)
    return _transient_bases_cache


def is_transient(exc: BaseException) -> bool:
    """Classify ``exc`` as a transient (retryable) failure.

    Three layers, cheapest first:

    1. ``getattr(exc, "status_code", None)`` in
       :data:`TRANSIENT_STATUS_CODES` or ``>= 500``.
    2. Any class in ``type(exc).__mro__`` whose name contains one of
       :data:`_TRANSIENT_CLASS_NAMES`.
    3. ``isinstance`` against the lazily-built SDK bases from
       :func:`_lazy_transient_bases` (whichever of anthropic / openai is
       installed).
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (
        status in TRANSIENT_STATUS_CODES or status >= 500
    ):
        return True
    for klass in type(exc).__mro__:
        klass_name = klass.__name__
        for needle in _TRANSIENT_CLASS_NAMES:
            if needle in klass_name:
                return True
    sdk_bases = _lazy_transient_bases()
    if sdk_bases and isinstance(exc, sdk_bases):
        return True
    return False


async def _aclose_quietly(gen: Any) -> None:
    """``aclose`` a stream generator, suppressing teardown noise.

    Called before each stream retry so the provider's async-with teardown
    runs cleanly; without it the SDK connection leaks across attempts. Only
    ``Exception`` subclasses are suppressed — ``CancelledError`` and other
    ``BaseException`` (cancellation, ``SystemExit``, ``KeyboardInterrupt``)
    propagate so async cancellation is never swallowed.
    """
    try:
        await gen.aclose()
    except Exception:  # noqa: BLE001 — best-effort teardown
        logger.debug(
            "retry: aclose raised during stream cleanup; suppressed",
            exc_info=True,
        )


async def call_with_retry(
    func: Callable[..., Awaitable[Any]],
    **kwargs: Any,
) -> Any:
    """Await ``func(**kwargs)``, retrying transient failures with backoff.

    Makes up to ``resolve_max_retries() + 1`` total attempts. A
    non-transient error re-raises immediately (no backoff sleep). On
    exhaustion the LAST caught transient exception is re-raised unchanged
    so the caller still sees the real SDK error type and message.
    """
    max_retries = resolve_max_retries()
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func(**kwargs)
        except Exception as exc:
            if not is_transient(exc):
                raise
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = _backoff_delay(attempt)
            logger.info(
                "call_with_retry: attempt %d/%d failed transiently (%r); "
                "retrying in %.2fs",
                attempt + 1,
                max_retries + 1,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None  # loop only exits via break with last_exc set
    raise last_exc


async def retrying_stream(
    func: Callable[..., Awaitable[Any]],
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """Drive ``func(**kwargs)`` (an ``async`` callable returning an async
    iterator) through the transient-retry policy, honouring the stream
    commit boundary.

    Retries (a) the ``await func(**kwargs)`` open and (b) iteration up to
    AND INCLUDING the first forwarded event. Once a single event has been
    yielded to the caller the generator is COMMITTED; any later error
    propagates as-is (retrying past first-emit would replay
    ``message_start`` / double-yield partial content). Before each retry
    the prior stream generator is ``aclose``d.

    On open/iteration exhaustion the LAST caught transient exception is
    re-raised unchanged. An empty stream (``StopAsyncIteration`` on the
    first ``__anext__``) yields nothing and is NOT retried — committing
    with zero events is a clean, final state.
    """
    max_retries = resolve_max_retries()
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        gen: Any = None
        try:
            gen = await func(**kwargs)
            first = await gen.__anext__()
        except StopAsyncIteration:
            # Empty stream — committed with zero events; do not retry.
            return
        except Exception as exc:
            # Tear down the partially-opened generator whether or not we
            # retry, so the provider's async-with teardown always runs.
            # (Current providers unwind via async-with, but a future
            # provider holding the connection outside an async-with would
            # leak without this defensive aclose.)
            if gen is not None:
                await _aclose_quietly(gen)
            if not is_transient(exc):
                raise
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = _backoff_delay(attempt)
            logger.info(
                "retrying_stream: attempt %d/%d failed transiently before "
                "commit (%r); retrying in %.2fs",
                attempt + 1,
                max_retries + 1,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            continue
        # First event captured — commit boundary crossed. From here on any
        # error propagates as-is (no retry, no replay).
        yield first
        async for event in gen:
            yield event
        return
    assert last_exc is not None
    raise last_exc
