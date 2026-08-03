"""Tests that MCPSessionHandle is always pinned.

The HandleManager's background reaper uses anyio cancel scopes in
``disconnect_from_server`` during MCP teardown, which propagates
``CancelledError`` to unrelated background tasks on the event loop
(prompt_toolkit's input pump). Pinning MCP handles at adopt time
guarantees the reaper never touches them — MCP sessions are
framework-managed (created at startup, released at ``aclose``) and
must not be reclaimed mid-chat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import Implementation

if TYPE_CHECKING:
    from mcp.client.session import ClientSession


def _make_server() -> FastMCP:
    """Minimal in-memory MCP server with an ``add`` tool."""
    server = FastMCP("test-server")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    return server


def _mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``cothis.ai.get_provider`` so ``Agent(...)`` needs no provider/network."""
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **kw: MagicMock())


def _patch_in_memory_transport(
    monkeypatch: pytest.MonkeyPatch,
    fastmcp: FastMCP,
) -> None:
    """Monkeypatch ``ClientSessionGroup.connect_to_server`` to be in-memory.

    Mirrors ``tests/test_mcp_tools._patch_in_memory_transport`` so Agent
    integration tests can reach the real ``MCPServer.connect_into`` path
    without spawning a subprocess.
    """
    from mcp import ClientSessionGroup

    async def _in_memory_connect_to_server(
        self: ClientSessionGroup, params: Any, session_params: Any = None
    ) -> ClientSession:
        session = await self._exit_stack.enter_async_context(  # noqa: SLF001
            create_connected_server_and_client_session(fastmcp)
        )
        await self.connect_with_session(
            Implementation(name=fastmcp.name or "server", version="1.0"), session
        )
        return session

    monkeypatch.setattr(
        ClientSessionGroup, "connect_to_server", _in_memory_connect_to_server
    )


@pytest.mark.asyncio
async def test_mcp_handle_is_pinned_at_adopt_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP session handles are always pinned — framework-managed lifecycle.

    Pinning is forced by ``Agent._ensure_mcp`` because MCP sessions are
    created at startup + released at ``aclose``. The reaper must not
    reclaim them mid-chat (anyio cancel scopes during teardown crash
    prompt_toolkit). Historically ``MCPServer`` had a ``pin`` field
    that was silently ignored; it's removed (#287), but the forced
    behaviour is unchanged.
    """
    from cothis.agent import Agent
    from cothis.tools.mcp import MCPServer

    _mock_llm(monkeypatch)
    fastmcp = _make_server()
    _patch_in_memory_transport(monkeypatch, fastmcp)

    server = MCPServer(name="mcp:test", params=None)
    agent = Agent(model="x", provider="openrouter", tools=[server])
    await agent._ensure_mcp()
    try:
        add_keys = [k for k in agent._tool_map if "add" in k]
        assert add_keys, f"expected an 'add' tool, got {list(agent._tool_map)}"
        tool = agent._tool_map[add_keys[0]]
        cls = getattr(tool, "_handle_cls")
        slot = agent._handle_manager._slots[cls]
        assert cls.pin is True, "MCPSessionHandle must be pinned at adopt time"
        assert slot.is_pinned, (
            "adopted slot must report is_pinned=True so the reaper skips it"
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_mcp_handle_not_reclaimed_by_reaper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backdating ``last_used`` on a pinned MCP slot is a no-op for the reaper."""
    import time

    from cothis.agent import Agent
    from cothis.tools.mcp import MCPServer

    _mock_llm(monkeypatch)
    fastmcp = _make_server()
    _patch_in_memory_transport(monkeypatch, fastmcp)

    server = MCPServer(name="mcp:test", params=None)
    agent = Agent(model="x", provider="openrouter", tools=[server])
    await agent._ensure_mcp()
    try:
        add_keys = [k for k in agent._tool_map if "add" in k]
        assert add_keys, f"expected an 'add' tool, got {list(agent._tool_map)}"
        tool = agent._tool_map[add_keys[0]]
        cls = getattr(tool, "_handle_cls")
        slot = agent._handle_manager._slots[cls]
        slot.last_used = time.time() - 9999
        await agent._handle_manager.reclaim_idle()
        assert slot.is_live, (
            "reaper reclaimed a pinned MCP handle — would crash prompt_toolkit"
        )
    finally:
        await agent.aclose()
