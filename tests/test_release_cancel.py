"""Tests for ``HandleManager._release`` swallowing ``CancelledError`` (#185).

MCP session teardown cancels in-flight tasks on the event loop, which
surfaces as ``asyncio.CancelledError`` from
``MCPSessionHandle.release``. Since Python 3.8 ``CancelledError``
inherits from ``BaseException`` (not ``Exception``), the existing
``except Exception`` in ``_release`` didn't catch it — the error
escaped, killed the reaper task, and crashed ``cothis chat`` after
10 min of idle MCP keepalive. These tests pin the fix: ``_release``
swallows ``CancelledError`` like any other release-path error.
"""

from __future__ import annotations

import asyncio

import pytest

from cothis.tools.core import (
    HandleManager,
    ResourceHandle,
    ensure_handle_ready,
    resource,
    tool,
)


@pytest.mark.asyncio
async def test_release_swallows_cancelled_error() -> None:
    """``release`` raising CancelledError must not escape ``_release``."""

    @resource(keepalive=600)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            pass

        async def release(self) -> None:
            raise asyncio.CancelledError()

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager()
    mgr.bind(t)
    await ensure_handle_ready(t)
    # Must not raise CancelledError.
    await mgr.release_all()


@pytest.mark.asyncio
async def test_release_swallows_cancelled_error_in_reclaim_idle() -> None:
    """The reaper path (``reclaim_idle`` → ``_release``) also swallows it.

    This is the actual crash path from #185: an idle MCP handle's
    release raises CancelledError, which used to escape the reaper
    task and kill it.
    """

    @resource(keepalive=0.0)  # immediately idle → reaper picks it up
    class H(ResourceHandle):
        async def acquire(self) -> None:
            pass

        async def release(self) -> None:
            raise asyncio.CancelledError()

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager()
    mgr.bind(t)
    await ensure_handle_ready(t)
    # Must not raise CancelledError.
    reclaimed = await mgr.reclaim_idle()
    assert reclaimed == 1


@pytest.mark.asyncio
async def test_release_cancelled_error_still_clears_live_at() -> None:
    """Cleanup (``slot.live_at = None``) runs even when release raises.

    Without this, the slot would stay marked live and the next
    ``ensure_handle_ready`` would skip re-acquisition.
    """

    @resource(keepalive=600)
    class H(ResourceHandle):
        async def acquire(self) -> None:
            pass

        async def release(self) -> None:
            raise asyncio.CancelledError()

    @tool("t", handle=H)
    def t() -> str:
        return "ok"

    mgr = HandleManager()
    mgr.bind(t)
    await ensure_handle_ready(t)
    # Find the slot.
    slots = [s for s in mgr._slots.values() if s.instance.__class__ is H]
    assert len(slots) == 1
    slot = slots[0]
    assert slot.is_live

    await mgr.release_all()

    assert not slot.is_live


@pytest.mark.asyncio
async def test_release_still_swallows_regular_exception() -> None:
    """Regression: existing ``except Exception`` path still works."""

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
