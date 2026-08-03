"""``cothis.tools.fs`` — filesystem tool subpackage.

Holds the read/list/search/create/modify/delete tools. The subpackage
replaced the codex ``apply_patch`` writer (``fs.write``) with three
simpler per-operation tools (#198):

- ``fs.read`` — numbered-line file reader.
- ``fs.list`` — directory listing.
- ``fs.search`` — regex content search.
- ``fs.create`` — new-file writer (rejects existing).
- ``fs.modify`` — line-range anchored edit.
- ``fs.delete`` — file removal.
"""

from cothis.tools.fs.create import _create as fs_create
from cothis.tools.fs.delete import _delete as fs_delete
from cothis.tools.fs.list import _list as fs_list
from cothis.tools.fs.modify import _modify as fs_modify
from cothis.tools.fs.read import read
from cothis.tools.fs.search import _search as fs_search

__all__ = ["read", "fs_list", "fs_search", "fs_create", "fs_modify", "fs_delete"]
