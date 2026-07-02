"""``cothis.tools`` — built-in tools, YAML/MCP loaders, and output formatting.

Public interface for the tools package. Submodules (``core``, ``mcp``,
``format``, ``builtins``) are internal; import from here, not from them.
"""

from __future__ import annotations

from cothis.tools.builtins import TOOLS, read, write
from cothis.tools.core import (
    AfterExecuteError,
    CommandBlock,
    Tool,
    ToolDef,
    load_tools_from_layer,
    load_yaml_tools,
    logger,
    preview,
    run_hooks_safe,
    schema_for,
    tool,
)
from cothis.tools.format import (
    flatten_dict,
    format_tool_output,
    to_tabular,
)
from cothis.tools.mcp import (
    MCPClientTool,
    MCPServer,
)
