"""``cothis.tools`` — built-in tools, YAML/MCP loaders, and output formatting.

Public interface for the tools package. Submodules (``core``, ``yaml``,
``mcp``, ``format``, ``builtins``) are internal; import from here, not from
them.
"""

from __future__ import annotations

from cothis.tools.builtins import read, write
from cothis.tools.core import (
    AfterExecuteError,
    Tool,
    ToolDef,
    discover_tools,
    logger,
    run_hooks_safe,
    schema_for,
    tool,
)
from cothis.tools.format import format_tool_output
from cothis.tools.mcp import (
    MCPClientTool,
    MCPServer,
)
from cothis.tools.yaml import shell
