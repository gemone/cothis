"""``cothis.tools`` — built-in tools, YAML/MCP loaders, and output formatting.

Public interface for the tools package. Re-exported here:
``discover_tools`` (the aggregator the CLI calls), ``tool`` / ``ToolDef``
(the Python-tool authoring API), ``Tool`` / ``MCPServer`` / ``MCPClientTool``
(types the Agent and tests consume), ``read`` / ``write`` (the fs builtins
re-exported for ``from cothis import read`` convenience), ``format_tool_output``, and the four
hook-related names (``run_hooks_safe`` / ``schema_for`` / ``AfterExecuteError``
/ ``logger``).

Submodules (``core``, ``yaml``, ``mcp``, ``format``, ``builtins``) host the
implementation; tests import from them directly for white-box checks, but
author-facing code should import from this package root.

cothis: ceiling — ``TOOLS`` (the builtin registry) is intentionally NOT
re-exported here. PRD story 39 asked for ``from cothis.tools import TOOLS``
backward compat; it was dropped (see ADR-0005) because ``TOOLS`` is the
builtin-layer input to ``discover_tools``, not the public aggregator.
Upgrade path: none planned — if a future caller needs the builtin set,
it should call ``discover_tools(Path("."), Path("."))`` or import
``cothis.tools.builtins.TOOLS`` explicitly with the understanding that
it's a layer input, not a public API.
"""

from __future__ import annotations

from cothis.tools.builtins import read
from cothis.tools.core import (
    AfterExecuteError,
    HandleManager,
    ResourceHandle,
    Tool,
    ToolDef,
    discover_tools,
    ensure_handle_ready,
    handle_call_done,
    logger,
    mark_inflight,
    resource,
    run_hooks_safe,
    schema_for,
    tool,
)
from cothis.tools.format import format_tool_output
from cothis.tools.mcp import (
    MCPClientTool,
    MCPServer,
)

__all__ = [
    "AfterExecuteError",
    "HandleManager",
    "MCPClientTool",
    "MCPServer",
    "ResourceHandle",
    "Tool",
    "ToolDef",
    "discover_tools",
    "ensure_handle_ready",
    "format_tool_output",
    "handle_call_done",
    "logger",
    "mark_inflight",
    "read",
    "resource",
    "run_hooks_safe",
    "schema_for",
    "tool",
]
