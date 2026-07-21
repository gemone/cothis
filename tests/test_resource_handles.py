"""Tests for the ``ResourceHandle`` + ``HandleManager``.

The handle system gives tools a managed external resource: declared
independently (``@resource``), bound to tools (``@tool(handle=…)``), with
keepalive reclamation + LRU eviction. These tests cover the lifecycle
contract — acquire/release, idle reclamation, LRU eviction, multi-tool
sharing, and the self-healing path — plus the critical guarantee that a
tool WITHOUT a handle is untouched (the no-op duck-typed path that keeps
the existing 222 tests green).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import time_machine

from cothis.tools import ResourceHandle, resource, tool
from cothis.tools.core import HandleManager, ensure_handle_ready


def test_resource_decorator_sets_keepalive() -> None:
    """``@resource(keepalive=N)`` stamps the keepalive onto the class."""

    @resource(keepalive=120)
    class H(ResourceHandle):
        pass

    assert H.keepalive == 120


def test_resource_decorator_bare() -> None:
    """``@resource`` without parens uses the default keepalive (600s)."""

    @resource
    class H(ResourceHandle):
        pass

    assert H.keepalive == 600.0


def test_tool_without_handle_has_none() -> None:
    """A plain ``@tool`` has ``_handle_cls`` None and ``handle`` None."""

    @tool("plain")
    def f(x: str) -> str:
        return x

    assert f._handle_cls is None
    assert f.handle is None


@pytest.mark.asyncio
async def test_handle_acquire_on_first_use() -> None:
    """``ensure_handle_ready`` acquires on first call, assigns to ``.handle``."""
    calls: list[str] = []

    @resource(keepalive=600)
    class Counter(ResourceHandle):
        async def acquire(self) -> None:
            calls.append("acquire")
            self.value = 0

        async def release(self) -> None:
            calls.append("release")

    @tool("inc", handle=Counter)
    def inc() -> str:
        assert inc.handle is not None
        return str(getattr(inc.handle, "value"))

    mgr = HandleManager()
    mgr.bind(inc)
    assert inc.handle is None  # not yet acquired

    await ensure_handle_ready(inc)
    assert inc.handle is not None
    assert inc.handle.value == 0
    assert calls == ["acquire"]


@pytest.mark.asyncio
async def test_handle_not_reacquired_while_live() -> None:
    """A second call within the live window does NOT re-acquire."""
    calls: list[str] = []

    @resource(keepalive=600)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            calls.append("acquire")

        async def release(self) -> None:
            calls.append("release")

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager()
    mgr.bind(t)
    await ensure_handle_ready(t)
    await ensure_handle_ready(t)
    assert calls == ["acquire"]  # only once


@pytest.mark.asyncio
async def test_idle_reclamation_releases() -> None:
    """``reclaim_idle`` releases handles past their keepalive window."""
    calls: list[str] = []

    @resource(keepalive=300)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            calls.append("acquire")

        async def release(self) -> None:
            calls.append("release")

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    with time_machine.travel(0.0, tick=False) as t_clock:
        mgr = HandleManager()
        mgr.bind(t)
        await ensure_handle_ready(t)
        assert calls == ["acquire"]

        t_clock.shift(301)  # past keepalive=300
        n = await mgr.reclaim_idle()
        assert n == 1
        assert "release" in calls


@pytest.mark.asyncio
async def test_self_healing_after_reclamation() -> None:
    """After idle reclamation, ``ensure_handle_ready`` re-acquires (self-heal)."""
    calls: list[str] = []

    @resource(keepalive=300)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            calls.append("acquire")

        async def release(self) -> None:
            calls.append("release")

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    with time_machine.travel(0.0, tick=False) as t_clock:
        mgr = HandleManager()
        mgr.bind(t)
        await ensure_handle_ready(t)
        t_clock.shift(301)  # past keepalive
        await mgr.reclaim_idle()

        # Re-acquired on next ensure — the self-healing path.
        await ensure_handle_ready(t)
        assert calls.count("acquire") == 2


@pytest.mark.asyncio
async def test_lru_eviction_under_pressure() -> None:
    """When ``max_handles`` is reached, the coldest handle is evicted."""
    released: list[str] = []

    def make_handle(name: str) -> type[ResourceHandle]:
        @resource(keepalive=999)
        class H(ResourceHandle):
            async def acquire(self) -> None:
                self.name = name

            async def release(self) -> None:
                released.append(name)

        return H

    tools: list[Any] = []
    for i in range(3):
        h = make_handle(f"h{i}")

        @tool(f"t{i}", handle=h)
        def t() -> str:
            assert t.handle is not None
            return str(getattr(t.handle, "name"))

        tools.append(t)

    mgr = HandleManager(max_handles=2)
    for t in tools:
        mgr.bind(t)

    # Acquire all three — third one triggers eviction of the coldest (h0).
    await ensure_handle_ready(tools[0])
    await ensure_handle_ready(tools[1])
    await ensure_handle_ready(tools[2])
    assert "h0" in released


@pytest.mark.asyncio
async def test_shared_handle_one_instance() -> None:
    """Two tools binding the same handle class share one instance."""

    @resource(keepalive=600)
    class Shared(ResourceHandle):
        async def acquire(self) -> None:
            self.n = 42

        async def release(self) -> None:
            pass

    @tool("a", handle=Shared)
    def a() -> str:
        assert a.handle is not None
        return str(getattr(a.handle, "n"))

    @tool("b", handle=Shared)
    def b() -> str:
        assert b.handle is not None
        return str(getattr(b.handle, "n"))

    mgr = HandleManager()
    mgr.bind(a)
    mgr.bind(b)

    await ensure_handle_ready(a)
    await ensure_handle_ready(b)
    assert a.handle is b.handle  # same instance


@pytest.mark.asyncio
async def test_no_handle_tool_is_noop() -> None:
    """A tool without a handle: ``ensure_handle_ready`` is a silent no-op."""

    @tool("plain")
    def f(x: str) -> str:
        return x

    # No manager bound — no-op.
    await ensure_handle_ready(f)
    assert f.handle is None


@pytest.mark.asyncio
async def test_release_all() -> None:
    """``release_all`` tears down every live handle."""
    calls: list[str] = []

    @resource(keepalive=600)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            calls.append("acquire")

        async def release(self) -> None:
            calls.append("release")

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager()
    mgr.bind(t)
    await ensure_handle_ready(t)
    await mgr.release_all()
    assert calls == ["acquire", "release"]


@pytest.mark.asyncio
async def test_release_is_idempotent_safe() -> None:
    """``release`` errors are swallowed (teardown must not raise)."""

    @resource(keepalive=600)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            pass

        async def release(self) -> None:
            raise RuntimeError("boom")

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager()
    mgr.bind(t)
    await ensure_handle_ready(t)
    await mgr.release_all()  # must not raise


@pytest.mark.asyncio
async def test_handle_not_in_llm_schema() -> None:
    """The handle does NOT appear in the tool's LLM schema — only real params."""

    @resource(keepalive=600)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            pass

        async def release(self) -> None:
            pass

    @tool("q", handle=H)
    def q(sql: str) -> str:
        """Run a query.

        Args:
            sql: The query.
        """
        return sql

    props = q.__cothis_schema__["input_schema"]["properties"]
    assert list(props) == ["sql"]
    assert "handle" not in props


@pytest.mark.asyncio
async def test_reaper_reclaims_during_idle() -> None:
    """The background reaper reclaims idle handles without a turn running.

    Simulates the ``chat`` idle gap: acquire a handle with a short keepalive,
    then await (no ``reclaim_idle`` call from the agent loop) and let the
    reaper fire. The handle must be released automatically.
    """
    calls: list[str] = []

    @resource(keepalive=0.01)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            calls.append("acquire")

        async def release(self) -> None:
            calls.append("release")

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager(reaper_interval=0.02)
    mgr.bind(t)
    await ensure_handle_ready(t)
    assert calls == ["acquire"]

    # Let the reaper fire — no turn/reclaim_idle call, just wait.
    await asyncio.sleep(0.08)
    assert "release" in calls
    await mgr.release_all()


# --- in-flight protection (finding #4) ---------------------------------


@pytest.mark.asyncio
async def test_inflight_handle_not_reclaimed_mid_call() -> None:
    """A handle marked in-flight is skipped by ``reclaim_idle``."""

    @resource(keepalive=0.01)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            pass

        async def release(self) -> None:
            pass

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager()
    mgr.bind(t)
    await ensure_handle_ready(t)
    mgr.mark_inflight(t)

    # Past keepalive, but in-flight → not reclaimed.
    import time

    mgr._slots[H].last_used = time.time() - 100
    reclaimed = await mgr.reclaim_idle()
    assert reclaimed == 0
    assert mgr._slots[H].is_live
    await mgr.release_all()


@pytest.mark.asyncio
async def test_call_done_refreshes_last_used() -> None:
    """``call_done`` decrements in-flight and refreshes ``last_used``."""

    @resource(keepalive=0.01)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            pass

        async def release(self) -> None:
            pass

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager()
    mgr.bind(t)
    await ensure_handle_ready(t)
    mgr.mark_inflight(t)

    import time

    old = time.time() - 100
    mgr._slots[H].last_used = old
    mgr.call_done(t)

    assert mgr._slots[H].inflight == 0
    assert mgr._slots[H].last_used > old
    await mgr.release_all()


@pytest.mark.asyncio
async def test_inflight_balanced_when_repr_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: if ``repr(arg)`` raises inside ``_execute`` (between
    ``mark_inflight`` and the body's ``finally``), the in-flight window must
    still close. Before the fix, ``mark_inflight`` sat outside the ``try``,
    so a broken ``__repr__`` leaked the handle as permanently in-flight —
    the reaper would then never reclaim it. Drives the real ``Agent._execute``
    with an arg whose ``repr`` explodes during the debug log line.
    """
    from unittest.mock import MagicMock

    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )

    from cothis.agent import Agent

    @resource(keepalive=0.01)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            pass

        async def release(self) -> None:
            pass

    @tool("reprboom", handle=H)
    def reprboom(x: str) -> str:  # body never reached
        return "ok"

    class BoomRepr:
        def __repr__(self) -> str:
            raise ValueError("repr broken")

    agent = Agent(model="x", provider="openrouter", tools=[reprboom])
    mgr = agent._handle_manager
    # The handle isn't live until ensure_acquired runs in _execute_tool; that's
    # fine — we assert on the post-call refcount, not pre-call state.

    # tool_use.input is already a dict (Messages API delivers it parsed), so
    # inject BoomRepr directly to make the debug repr raise inside _execute_tool.
    tu = {"name": "reprboom", "input": {"x": BoomRepr()}}

    is_error, result = await agent._execute_tool(tu)

    # repr raised inside _execute_tool's debug logging → surfaced as error to LLM.
    assert is_error is True
    assert "Error" in result
    # The fix: mark_inflight is inside the try, so the finally ran and the
    # refcount balanced. Pre-fix this would be 1 (leaked).
    assert mgr._slots[H].inflight == 0
    await agent.aclose()


# --- eager / pin --------------------------------------------------------


@pytest.mark.asyncio
async def test_eager_acquired_on_start() -> None:
    """``start_eager`` acquires handles with ``eager=True``."""

    calls: list[str] = []

    @resource(eager=True)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            calls.append("acquire")

        async def release(self) -> None:
            calls.append("release")

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager()
    mgr.bind(t)
    assert not mgr._slots[H].is_live
    await mgr.start_eager()
    assert mgr._slots[H].is_live
    assert calls == ["acquire"]
    await mgr.release_all()


@pytest.mark.asyncio
async def test_non_eager_not_started() -> None:
    """Handles without ``eager`` are not acquired by ``start_eager``."""

    @resource
    class H(ResourceHandle):
        async def acquire(self) -> None:
            pass

        async def release(self) -> None:
            pass

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager()
    mgr.bind(t)
    await mgr.start_eager()
    assert not mgr._slots[H].is_live


@pytest.mark.asyncio
async def test_pin_exempt_from_reclaim_idle() -> None:
    """A pinned handle is never reclaimed by ``reclaim_idle``."""

    @resource(pin=True)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            pass

        async def release(self) -> None:
            pass

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager()
    mgr.bind(t)
    await mgr.start_eager()
    assert mgr._slots[H].is_live

    import time

    mgr._slots[H].last_used = time.time() - 9999
    reclaimed = await mgr.reclaim_idle()
    assert reclaimed == 0
    assert mgr._slots[H].is_live
    await mgr.release_all()


@pytest.mark.asyncio
async def test_pin_exempt_from_eviction_and_budget() -> None:
    """Pinned handles don't count toward ``max_handles`` and aren't evicted."""

    calls: list[str] = []

    @resource(pin=True)
    class Pinned(ResourceHandle):
        async def acquire(self) -> None:
            calls.append("pinned-acquire")

        async def release(self) -> None:
            calls.append("pinned-release")

    @resource
    class Normal(ResourceHandle):
        async def acquire(self) -> None:
            calls.append("normal-acquire")

        async def release(self) -> None:
            calls.append("normal-release")

    @tool("p", handle=Pinned)
    def p() -> str:
        return "ok"

    @tool("n", handle=Normal)
    def n() -> str:
        return "ok"

    mgr = HandleManager(max_handles=1)
    mgr.bind(p)
    mgr.bind(n)

    # Pinned handle fills the "budget" (but doesn't count).
    await mgr.start_eager()
    assert mgr._slots[Pinned].is_live

    # Normal handle still acquires despite max_handles=1 — pinned doesn't
    # count against the budget.
    await ensure_handle_ready(n)
    assert mgr._slots[Normal].is_live
    assert mgr._slots[Pinned].is_live  # pinned was not evicted
    await mgr.release_all()


@pytest.mark.asyncio
async def test_pin_implies_eager() -> None:
    """``@resource(pin=True)`` sets ``eager=True`` on the class."""

    @resource(pin=True)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            pass

        async def release(self) -> None:
            pass

    assert H.pin is True
    assert H.eager is True


# --- adopt --------------------------------------------------------------


@pytest.mark.asyncio
async def test_adopt_seeds_live_instance() -> None:
    """``adopt`` registers an already-acquired instance as live (no acquire)."""

    calls: list[str] = []

    class H(ResourceHandle):
        async def acquire(self) -> None:
            calls.append("acquire")

        async def release(self) -> None:
            calls.append("release")

    mgr = HandleManager()
    instance = H()
    mgr.adopt(H, instance)
    assert mgr._slots[H].is_live
    assert mgr._slots[H].instance is instance
    assert calls == []  # adopt never calls acquire

    # ensure_acquired finds it already live → no re-acquire.
    fake_tool = MagicMock()
    fake_tool._handle_cls = H
    fake_tool._handle_manager = mgr
    await mgr.ensure_acquired(fake_tool)
    assert calls == []  # still no acquire (already live)
    await mgr.release_all()
    assert calls == ["release"]
