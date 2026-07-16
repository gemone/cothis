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

    props = q.__cothis_schema__["function"]["parameters"]["properties"]
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
