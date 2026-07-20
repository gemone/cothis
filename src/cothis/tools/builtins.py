"""Built-in filesystem tools shipped with cothis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cothis.tools.fs.list import list as _list
from cothis.tools.fs.read import read
from cothis.tools.fs.write import write

if TYPE_CHECKING:
    from cothis.tools.core import Tool

TOOLS: list[Tool] = [read, _list, write]
