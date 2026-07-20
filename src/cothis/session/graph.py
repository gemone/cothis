"""In-memory session fork tree operations.

Module of free functions over a flat ``dict[str, SessionRow]`` map.
The tree is stored on the rows themselves (``parent_id`` /
``parent_seq`` columns); these functions layer tree operations on top
without touching SQLite. Every operation is a dict lookup or a stack
walk — no SQL, no recursive CTE (measured 3-10x slower than the
in-memory graph on a 10k-session fixture).

The map is built once at startup from :meth:`Storage.list_sessions`
and treated as immutable by these functions. ``Session.fork`` does not
mutate the map: it passes ``(parent_id, parent_seq)`` directly to
:func:`walk_ancestors`, so the ancestor walk sees the in-flight fork
link without synthesising a fake row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cothis.session.storage import SessionRow


class SessionNotFoundError(KeyError):
    """A session id passed to a graph operation is not in the map."""


def build(rows: Iterable[SessionRow]) -> dict[str, SessionRow]:
    """Index rows by ``id``. The returned map is the input to every other helper."""
    return {row.id: row for row in rows}


def parent_of(graph: dict[str, SessionRow], sid: str) -> str | None:
    """Direct parent id, or ``None`` if ``sid`` is a root."""
    _require(graph, sid)
    return graph[sid].parent_id


def parent_seq_of(graph: dict[str, SessionRow], sid: str) -> int | None:
    """``parent_seq`` cutoff for ``sid``'s parent, or ``None`` on roots."""
    _require(graph, sid)
    return graph[sid].parent_seq


def children_of(graph: dict[str, SessionRow], sid: str) -> list[str]:
    """Direct children of ``sid``. Empty for leaves."""
    _require(graph, sid)
    return [r.id for r in graph.values() if r.parent_id == sid]


def is_leaf(graph: dict[str, SessionRow], sid: str) -> bool:
    """``True`` if ``sid`` has no children — the only nodes ``delete`` accepts."""
    _require(graph, sid)
    return not any(r.parent_id == sid for r in graph.values())


def roots(graph: dict[str, SessionRow]) -> list[str]:
    """All root ids (sessions with no parent)."""
    return [sid for sid, row in graph.items() if row.parent_id is None]


def ancestors(graph: dict[str, SessionRow], sid: str) -> list[str]:
    """``[root, …, direct_parent]`` — empty when ``sid`` is itself a root."""
    _require(graph, sid)
    chain: list[str] = []
    current = graph[sid].parent_id
    seen: set[str] = {sid}
    while current is not None and current in graph and current not in seen:
        chain.append(current)
        seen.add(current)
        current = graph[current].parent_id
    chain.reverse()
    return chain


def subtree(graph: dict[str, SessionRow], sid: str) -> list[str]:
    """``sid`` plus every descendant (BF order, ``sid`` first)."""
    _require(graph, sid)
    out: list[str] = [sid]
    i = 0
    while i < len(out):
        current = out[i]
        i += 1
        out.extend(r.id for r in graph.values() if r.parent_id == current)
    return out


def walk_ancestors(
    graph: dict[str, SessionRow],
    start_id: str,
    *,
    start_parent_id: str | None = None,
    start_parent_seq: int | None = None,
) -> list[tuple[str, int | None]]:
    """Ancestor chain for ``start_id``, root → direct parent.

    Returns ``[(ancestor_id, seq_cap), …]`` where ``seq_cap`` is the
    inclusive ``seq`` cutoff applied to ``ancestor_id``'s blocks during
    context assembly. The cutoff comes from the *child* link's
    ``parent_seq`` — i.e. the descendant that brought this ancestor
    into the chain.

    ``start_parent_id`` / ``start_parent_seq`` are used when
    ``start_id`` is not yet in ``graph`` (an in-flight fork row that
    hasn't been flushed). ``Session.fork`` passes the constructor
    args here so the ancestor walk sees the link without mutating the
    graph. When ``start_id`` IS in the graph, the overrides are
    ignored — the graph's stored link is authoritative.
    """
    if start_id in graph:
        parent_id = graph[start_id].parent_id
        cap = graph[start_id].parent_seq
    else:
        parent_id = start_parent_id
        cap = start_parent_seq
    out: list[tuple[str, int | None]] = []
    seen: set[str] = {start_id}
    while parent_id is not None and parent_id in graph and parent_id not in seen:
        out.append((parent_id, cap))
        seen.add(parent_id)
        ancestor = graph[parent_id]
        cap = ancestor.parent_seq
        parent_id = ancestor.parent_id
    out.reverse()
    return out


def _require(graph: dict[str, SessionRow], sid: str) -> None:
    if sid not in graph:
        raise SessionNotFoundError(sid)
