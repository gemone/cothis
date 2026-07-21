"""Built-in filesystem tools shipped with cothis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cothis.tools.fs.list import _list
from cothis.tools.fs.read import read
from cothis.tools.fs.search import _search
from cothis.tools.fs.write import write

if TYPE_CHECKING:
    from cothis.tools.core import Tool

TOOLS: list[Tool] = [read, _list, write, _search]
