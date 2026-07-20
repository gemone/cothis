"""In-memory session fork tree.

Built once at startup from the ``sessions`` table's flat rows. The tree
is stored as ``parent_id``/``parent_seq`` columns on each session row
(``NULL`` on roots); this module layers tree operations on top without
touching SQLite — every operation is a dict lookup or a stack walk, no
SQL, no recursive CTE (measured 3-10x slower than the in-memory graph
on a 10k-session fixture).

Single-threaded by design: the CLI builds one ``SessionGraph`` per
invocation, walks it to list/resume/fork/delete, then exits. The graph
is mutable only through :meth:`add`, used after a fork to keep the
in-memory tree consistent with the row the fork just wrote lazily; no
concurrent mutation is supported.

Each node is identified by its session id (a 32-char hex string). Edges
point child → parent: ``_parent_of[child_id] == parent_id``. Roots have
no entry. ``children_of`` is the inverse map, used by ``subtree`` and
the leaf-only delete check.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cothis.session.storage import SessionRow


class SessionNotFoundError(KeyError):
    """A session id passed to a graph operation is not in the graph."""


class SessionGraph:
    """In-memory directed tree of session forks.

    Construct from :class:`~cothis.session.storage.SessionRow` instances
    (any iterable; the graph does not own the rows). After construction,
    call :meth:`add` when a fork lazily writes its row so subsequent
    walks see the new node.
    """

    def __init__(self, rows: Iterable[SessionRow]) -> None:
        self._parent_of: dict[str, str] = {}
        self._parent_seq_of: dict[str, int] = {}
        self._children_of: dict[str, list[str]] = defaultdict(list)
        self._known: set[str] = set()
        for row in rows:
            self._insert(row)

    def _insert(self, row: SessionRow) -> None:
        sid = row.id
        self._known.add(sid)
        if row.parent_id is not None:
            self._parent_of[sid] = row.parent_id
            self._parent_seq_of[sid] = (
                row.parent_seq if row.parent_seq is not None else 0
            )
            self._children_of[row.parent_id].append(sid)

    def add(self, row: SessionRow) -> None:
        """Register a freshly-forked row so later walks see it.

        Idempotent on the id: re-adding a known id is a no-op (the lazy
        ``sessions`` row may already be in the graph if the user forked
        from a session whose row was flushed before this method ran).
        """
        if row.id in self._known:
            return
        self._insert(row)

    def __contains__(self, sid: object) -> bool:
        return sid in self._known

    def __len__(self) -> int:
        return len(self._known)

    def parent_of(self, sid: str) -> str | None:
        """Direct parent id, or ``None`` if ``sid`` is a root."""
        self._require(sid)
        return self._parent_of.get(sid)

    def parent_seq_of(self, sid: str) -> int | None:
        """``parent_seq`` cutoff for ``sid``'s parent, or ``None`` on roots.

        For a forked session, this is the inclusive ``seq`` cap applied
        to the parent's blocks during ancestor-chain context assembly.
        """
        self._require(sid)
        return self._parent_seq_of.get(sid)

    def children_of(self, sid: str) -> list[str]:
        """Direct children of ``sid`` (unordered). Empty for leaves."""
        self._require(sid)
        return list(self._children_of.get(sid, ()))

    def is_leaf(self, sid: str) -> bool:
        """``True`` if ``sid`` has no children — the only nodes ``delete`` accepts."""
        self._require(sid)
        return not self._children_of.get(sid)

    def roots(self) -> list[str]:
        """All root ids (sessions with no parent). Order is insertion order."""
        return [sid for sid in self._known if sid not in self._parent_of]

    def ancestors(self, sid: str) -> list[str]:
        """``[root, …, direct_parent]`` — empty when ``sid`` is itself a root.

        Walked via the ``_parent_of`` map (no SQL). The caller prepends
        ``sid`` to assemble the full chain root → current. If a parent
        id is missing from the graph (the row was deleted from the DB
        without cascading to its children — should not happen given the
        leaf-only delete contract, but defended against), the walk stops
        at the deepest reachable ancestor.
        """
        self._require(sid)
        chain: list[str] = []
        current = self._parent_of.get(sid)
        seen: set[str] = {sid}
        while current is not None and current not in seen:
            chain.append(current)
            seen.add(current)
            current = self._parent_of.get(current)
        chain.reverse()
        return chain

    def subtree(self, sid: str) -> list[str]:
        """``sid`` plus every descendant (BF order, ``sid`` first).

        Used by tests and by future tree-pruning operations; ``delete``
        only needs :meth:`is_leaf`.
        """
        self._require(sid)
        out: list[str] = [sid]
        i = 0
        while i < len(out):
            current = out[i]
            i += 1
            out.extend(self._children_of.get(current, ()))
        return out

    def _require(self, sid: str) -> None:
        if sid not in self._known:
            raise SessionNotFoundError(sid)
