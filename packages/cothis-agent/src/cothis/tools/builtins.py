"""Built-in filesystem tools shipped with cothis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cothis.tools.fs.create import _create
from cothis.tools.fs.delete import _delete
from cothis.tools.fs.list import _list
from cothis.tools.fs.modify import _modify
from cothis.tools.fs.read import read
from cothis.tools.fs.search import _search

if TYPE_CHECKING:
    from cothis.tools.core import Tool

TOOLS: list[Tool] = [read, _list, _search, _create, _modify, _delete]
