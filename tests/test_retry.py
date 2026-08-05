"""Tests for :mod:`cothis.ai._retry` — transient classifier + retry wrappers.

The retry layer is exercised through a ``FakeProvider`` whose ``amessages``
behaviour is parameterised by a list of per-attempt outcome dicts. Fake
exception CLASSES (named to match real SDK exception types) drive the
class-name-substring classifier path without importing the real SDKs in
the test data path. Backoff is neutralised by monkeypatching
:func:`_backoff_delay` so the suite is sub-second.
"""

from __future__ import annotations

from typing import Any

import pytest

import cothis.ai._retry as _retry
from cothis.ai._retry import (
    call_with_retry,
    is_transient,
    resolve_max_retries,
    retrying_stream,
)

# Capture the real backoff helper at import time. The ``fast_backoff`` fixture
# (below) monkeypatches ``_retry._backoff_delay`` for the retry-loop tests so
# they don't sleep; the backoff-properties tests exercise the real schedule
# via this reference, which the fixture does not touch.
_real_backoff_delay = _retry._backoff_delay

# ---------------------------------------------------------------------------
# Fake exception classes — named to exercise the class-name-substring
# classifier path. Deliberately NOT imported from the real SDKs so the
# tests don't couple to SDK internals; the classifier matches by name.
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    pass


class OverloadedError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class InternalServerError(Exception):
    pass


class BadRequestError(Exception):
    pass


class AuthenticationError(Exception):
    pass


# ---------------------------------------------------------------------------
# FakeProvider + TrackingGen
# ---------------------------------------------------------------------------


class TrackingGen:
    """Wraps an async generator, counting ``aclose`` calls for the
    aclose-before-retry assertion."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.aclose_count = 0

    def __aiter__(self) -> TrackingGen:
        return self

    async def __anext__(self) -> Any:
        return await self.inner.__anext__()

    async def aclose(self) -> None:
        self.aclose_count += 1
        await self.inner.aclose()


class FakeProvider:
    """``amessages`` whose behaviour is parameterised by outcome dicts.

    Outcome keys:
      * ``raise``      — ``amessages`` raises this on the await (covers
                         non-stream failure AND stream-open failure).
      * ``value``      — non-stream: returned verbatim.
      * ``anext_exc``  — stream: open succeeds, first ``__anext__`` raises.
      * ``events``     — stream: list of events to yield.
      * ``after_exc``  — stream: raised after yielding all events (post-commit).
    """

    def __init__(self, outcomes: list[dict[str, Any]]) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.streams: list[TrackingGen] = []

    async def amessages(self, **kwargs: Any) -> Any:
        if self.calls >= len(self.outcomes):
            raise AssertionError(
                f"FakeProvider.amessages called #{self.calls + 1} but only "
                f"{len(self.outcomes)} outcomes were provisioned"
            )
        outcome = self.outcomes[self.calls]
        self.calls += 1

        open_exc = outcome.get("raise")
        if open_exc is not None:
            raise open_exc

        if kwargs.get("stream"):
            wrapped = TrackingGen(self._make_stream(outcome))
            self.streams.append(wrapped)
            return wrapped
        return outcome.get("value")

    @staticmethod
    def _make_stream(outcome: dict[str, Any]) -> Any:
        anext_exc = outcome.get("anext_exc")
        events = outcome.get("events", [])
        after_exc = outcome.get("after_exc")

        async def gen() -> Any:
            if anext_exc is not None:
                raise anext_exc
            for ev in events:
                yield ev
            if after_exc is not None:
                raise after_exc

        return gen()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise backoff sleeps + retry-count env for the loop tests.

    Restored automatically after each test. Tests that exercise the real
    backoff helper / env reader re-monkeypatch or read the constants
    directly.
    """
    monkeypatch.setattr(_retry, "_backoff_delay", lambda attempt: 0.0)
    monkeypatch.delenv("COTHIS_MAX_RETRIES", raising=False)


# ---------------------------------------------------------------------------
# Non-stream: call_with_retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_stream_transient_then_success() -> None:
    """RateLimitError x2 then a value -> value returned; 3 invocations."""
    value = {"id": "msg_1"}
    provider = FakeProvider(
        [
            {"raise": RateLimitError("rl 1")},
            {"raise": RateLimitError("rl 2")},
            {"value": value},
        ]
    )
    result = await call_with_retry(
        provider.amessages, model="m", messages=[], max_tokens=1
    )
    assert result is value
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_non_stream_exhaustion_reraises_last(monkeypatch: pytest.MonkeyPatch) -> None:
    """OverloadedError every call with COTHIS_MAX_RETRIES=2 -> 3 invocations,
    last exception propagated unchanged."""
    monkeypatch.setenv("COTHIS_MAX_RETRIES", "2")
    provider = FakeProvider(
        [
            {"raise": OverloadedError("oload 1")},
            {"raise": OverloadedError("oload 2")},
            {"raise": OverloadedError("oload 3")},
        ]
    )
    with pytest.raises(OverloadedError) as excinfo:
        await call_with_retry(
            provider.amessages, model="m", messages=[], max_tokens=1
        )
    assert provider.calls == 3
    # LAST exception propagated, type + message unchanged.
    assert "oload 3" in str(excinfo.value)


@pytest.mark.asyncio
async def test_non_stream_non_transient_propagates_immediately() -> None:
    """BadRequestError -> exactly 1 invocation, raised unchanged, no retry."""
    provider = FakeProvider([{"raise": BadRequestError("bad")}])
    with pytest.raises(BadRequestError) as excinfo:
        await call_with_retry(
            provider.amessages, model="m", messages=[], max_tokens=1
        )
    assert "bad" in str(excinfo.value)
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_non_stream_connection_error_is_transient() -> None:
    """APIConnectionError once then success -> 2 invocations, value returned."""
    value = {"id": "msg_ok"}
    provider = FakeProvider(
        [
            {"raise": APIConnectionError("conn")},
            {"value": value},
        ]
    )
    result = await call_with_retry(
        provider.amessages, model="m", messages=[], max_tokens=1
    )
    assert result is value
    assert provider.calls == 2


# ---------------------------------------------------------------------------
# Stream: retrying_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_retry_before_first_event_anext() -> None:
    """Open succeeds but first __anext__ raises RateLimitError x2 then yields
    [A, B] -> events forwarded; 3 amessages invocations."""
    event_a, event_b = object(), object()
    provider = FakeProvider(
        [
            {"anext_exc": RateLimitError("rl 1")},
            {"anext_exc": RateLimitError("rl 2")},
            {"events": [event_a, event_b]},
        ]
    )
    out = [
        ev
        async for ev in retrying_stream(
            provider.amessages, model="m", messages=[], max_tokens=1, stream=True
        )
    ]
    assert out == [event_a, event_b]
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_stream_retry_on_open() -> None:
    """await func(stream=True) raises x2 then yields events -> 3 attempts."""
    event_a = object()
    provider = FakeProvider(
        [
            {"raise": RateLimitError("open 1")},
            {"raise": RateLimitError("open 2")},
            {"events": [event_a]},
        ]
    )
    out = [
        ev
        async for ev in retrying_stream(
            provider.amessages, model="m", messages=[], max_tokens=1, stream=True
        )
    ]
    assert out == [event_a]
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_stream_post_commit_no_retry() -> None:
    """First event yielded, second __anext__ raises InternalServerError ->
    exception propagates as-is AND the first event WAS yielded (assert
    ordering); exactly 1 amessages invocation."""
    event_a = object()
    provider = FakeProvider(
        [{"events": [event_a], "after_exc": InternalServerError("post")}]
    )
    yielded: list[Any] = []
    with pytest.raises(InternalServerError) as excinfo:
        async for ev in retrying_stream(
            provider.amessages,
            model="m",
            messages=[],
            max_tokens=1,
            stream=True,
        ):
            yielded.append(ev)
    # Ordering: the first event reached the caller BEFORE the raise.
    assert yielded == [event_a]
    assert "post" in str(excinfo.value)
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_stream_clean_empty_no_retry_loop() -> None:
    """Stream yields nothing (StopAsyncIteration on first __anext__) ->
    retrying_stream yields nothing, exactly 1 invocation, no retry."""
    provider = FakeProvider([{"events": []}])
    out = [
        ev
        async for ev in retrying_stream(
            provider.amessages, model="m", messages=[], max_tokens=1, stream=True
        )
    ]
    assert out == []
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_stream_aclose_before_retry() -> None:
    """Wrap the prior stream with a spy (TrackingGen) and assert aclose() was
    awaited once per retry when a transient error occurs pre-commit."""
    provider = FakeProvider(
        [
            {"anext_exc": RateLimitError("rl 1")},
            {"anext_exc": RateLimitError("rl 2")},
            {"events": [object()]},
        ]
    )
    _ = [
        ev
        async for ev in retrying_stream(
            provider.amessages, model="m", messages=[], max_tokens=1, stream=True
        )
    ]
    # Two pre-commit retries -> two streams aclosed; the committed (3rd) stream
    # is left to exhaust naturally and is never aclosed by the retry path.
    assert len(provider.streams) == 3
    assert provider.streams[0].aclose_count == 1
    assert provider.streams[1].aclose_count == 1
    assert provider.streams[2].aclose_count == 0


@pytest.mark.asyncio
async def test_stream_exhaustion_reraises_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every attempt fails transiently pre-commit -> the LAST caught
    exception is re-raised unchanged; exactly max_retries+1 invocations.

    The stream exhaustion tail (``raise last_exc``) mirrors
    ``call_with_retry``'s exhaustion; this pins that the *last* exception
    (not the first) propagates and the attempt count is exact.
    """
    monkeypatch.setenv("COTHIS_MAX_RETRIES", "2")
    provider = FakeProvider(
        [
            {"anext_exc": RateLimitError("rl 1")},
            {"anext_exc": RateLimitError("rl 2")},
            {"anext_exc": RateLimitError("rl 3")},
        ]
    )
    with pytest.raises(RateLimitError) as excinfo:
        _ = [
            ev
            async for ev in retrying_stream(
                provider.amessages, model="m", messages=[], max_tokens=1, stream=True
            )
        ]
    assert provider.calls == 3
    # LAST exception propagated, type + message unchanged.
    assert "rl 3" in str(excinfo.value)


# ---------------------------------------------------------------------------
# COTHIS_MAX_RETRIES env knob
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_retries_zero_disables_retry_non_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COTHIS_MAX_RETRIES", "0")
    value = {"id": "single"}
    provider = FakeProvider([{"value": value}])
    result = await call_with_retry(
        provider.amessages, model="m", messages=[], max_tokens=1
    )
    assert result is value
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_max_retries_zero_disables_retry_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COTHIS_MAX_RETRIES", "0")
    event_a = object()
    provider = FakeProvider([{"events": [event_a]}])
    out = [
        ev
        async for ev in retrying_stream(
            provider.amessages, model="m", messages=[], max_tokens=1, stream=True
        )
    ]
    assert out == [event_a]
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_max_retries_negative_disables_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-positive value disables retry -> exactly one invocation that
    still runs the single attempt. (Garbage -> default 3 is pinned by
    ``test_resolve_max_retries_garbage``; 'abc' is NOT disabled, so it is
    deliberately excluded here.)"""
    for raw in ("-1", "-2"):
        monkeypatch.setenv("COTHIS_MAX_RETRIES", raw)
        value = {"id": raw}
        provider = FakeProvider([{"value": value}])
        result = await call_with_retry(
            provider.amessages, model="m", messages=[], max_tokens=1
        )
        assert result is value
        assert provider.calls == 1


# ---------------------------------------------------------------------------
# resolve_max_retries
# ---------------------------------------------------------------------------


def test_resolve_max_retries_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COTHIS_MAX_RETRIES", raising=False)
    assert resolve_max_retries() == 3


def test_resolve_max_retries_five(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COTHIS_MAX_RETRIES", "5")
    assert resolve_max_retries() == 5


def test_resolve_max_retries_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COTHIS_MAX_RETRIES", "0")
    assert resolve_max_retries() == 0


def test_resolve_max_retries_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COTHIS_MAX_RETRIES", "-1")
    assert resolve_max_retries() == 0


def test_resolve_max_retries_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COTHIS_MAX_RETRIES", "garbage")
    assert resolve_max_retries() == 3  # default, not silent disable


# ---------------------------------------------------------------------------
# is_transient classifier
# ---------------------------------------------------------------------------


class _StatusExc(Exception):
    """Fake exception carrying a status_code, with a name that contains none
    of the classifier substrings (so only the status path can match)."""

    def __init__(self, code: int) -> None:
        super().__init__(f"status {code}")
        self.status_code = code


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_is_transient_status_codes(code: int) -> None:
    assert is_transient(_StatusExc(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403])
def test_is_transient_non_transient_status_codes(code: int) -> None:
    assert is_transient(_StatusExc(code)) is False


def test_is_transient_high_5xx_is_transient() -> None:
    """An arbitrary status >= 500 (e.g. 509) is transient."""
    assert is_transient(_StatusExc(509)) is True


def test_is_transient_class_name_path_match() -> None:
    """Fake class named RateLimitError with no status_code -> True."""
    assert is_transient(RateLimitError("rl")) is True


def test_is_transient_class_name_path_no_match() -> None:
    """Fake class named AuthenticationError -> False."""
    assert is_transient(AuthenticationError("auth")) is False


def test_is_transient_lazy_isinstance_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch _lazy_transient_bases to return a fake base; subclass
    instance -> True, unrelated -> False (name path must NOT fire first)."""

    class _MyTransientBase(Exception):
        pass

    class _MyTransientSub(_MyTransientBase):
        pass

    class _Unrelated(Exception):
        pass

    monkeypatch.setattr(_retry, "_lazy_transient_bases", lambda: (_MyTransientBase,))

    assert is_transient(_MyTransientSub("x")) is True
    assert is_transient(_Unrelated("x")) is False


# ---------------------------------------------------------------------------
# Backoff helper
# ---------------------------------------------------------------------------


def test_backoff_delay_bounds_and_capping() -> None:
    """Delays are non-negative, <= ceiling, and jitter never escapes it."""
    ceiling = _retry._RETRY_CEILING_DELAY_S
    for attempt in range(20):
        for _ in range(100):  # many draws per attempt
            d = _real_backoff_delay(attempt)
            assert 0 <= d <= ceiling


def test_backoff_delay_monotonic_non_decreasing_across_attempts() -> None:
    """One sample per attempt across the schedule is non-decreasing.

    Pre-cap ranges per attempt are disjoint and strictly increasing
    (base*2**n + [0,base)), so any single sample sequence is monotonic;
    once the raw value passes the ceiling the delay is pinned at the
    ceiling exactly.
    """
    seq = [_real_backoff_delay(i) for i in range(30)]
    assert all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))


def test_backoff_delay_pinned_at_ceiling_past_cap() -> None:
    assert _real_backoff_delay(20) == _retry._RETRY_CEILING_DELAY_S


def test_backoff_delay_1000_draws_within_bounds() -> None:
    ceiling = _retry._RETRY_CEILING_DELAY_S
    draws = [_real_backoff_delay(3) for _ in range(1000)]
    assert all(0 <= d <= ceiling for d in draws)


# ---------------------------------------------------------------------------
# Lazy-import invariant
# ---------------------------------------------------------------------------


def test_retry_module_has_no_top_level_sdk_import() -> None:
    """_retry.py must not import anthropic/openai at module top level.

    The SDK imports live inside ``_lazy_transient_bases`` (guarded by
    try/except ImportError) so importing this module never forces a
    provider SDK load. The package's ``__init__`` eagerly imports
    anthropic.types for its own type aliases — that is separate from this
    module's responsibility, so we check the source text (column-0 lines
    only) rather than ``sys.modules``.
    """
    from pathlib import Path

    src = Path(_retry.__file__).read_text()
    for line in src.splitlines():
        if not line or line[:1].isspace():
            continue  # blank or indented (function body / TYPE_CHECKING)
        stripped = line.lstrip()
        for sdk in ("anthropic", "openai"):
            assert not (
                stripped.startswith(f"import {sdk}")
                or stripped.startswith(f"from {sdk}")
            ), f"_retry.py has a top-level {sdk!r} import: {line!r}"


def test_openai_not_eagerly_imported_by_retry_module() -> None:
    """Importing cothis.ai._retry must not pull ``openai`` into sys.modules.

    (``anthropic`` is pulled by the ``cothis.ai`` package ``__init__`` via
    ``_types`` for its type aliases — that is independent of this module;
    ``openai`` is not imported by either, which is the checkable signal
    here.)
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import cothis.ai._retry  # noqa: F401\n"
        "assert 'openai' not in sys.modules, sys.modules.get('openai')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
