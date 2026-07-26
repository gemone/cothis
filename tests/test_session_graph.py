"""Tests for ``cothis.session.graph`` — fork-tree operations.

The graph is a flat ``dict[str, SessionRow]``; these tests cover the
free-function API: ``build``, ``roots``, ``ancestors``, ``subtree``,
``children_of``, ``is_leaf``, ``walk_ancestors`` (with the
in-flight-fork ``override_link`` path).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cothis.session import graph
from cothis.session.graph import SessionNotFoundError
from cothis.session.storage import SessionRow


def _row(
    sid: str,
    *,
    parent_id: str | None = None,
    parent_seq: int | None = None,
) -> SessionRow:
    return SessionRow(
        id=sid,
        parent_id=parent_id,
        parent_seq=parent_seq,
        cwd="/tmp",
        cli_version="x",
        model="m",
        title="t",
        created_at="2026-07-20T00:00:00",
        updated_at="2026-07-20T00:00:00",
    )


def _tree() -> dict[str, SessionRow]:
    """A → {C, E}; C → D. Two roots (A, B)."""
    return graph.build([
        _row("a"),
        _row("b"),
        _row("c", parent_id="a", parent_seq=5),
        _row("d", parent_id="c", parent_seq=10),
        _row("e", parent_id="a", parent_seq=3),
    ])


def test_roots_returns_all_rootless_nodes() -> None:
    g = _tree()
    assert sorted(graph.roots(g)) == ["a", "b"]


def test_ancestors_walks_root_to_direct_parent() -> None:
    """Order matters for context assembly — root first."""
    g = _tree()
    assert graph.ancestors(g, "d") == ["a", "c"]
    assert graph.ancestors(g, "c") == ["a"]
    assert graph.ancestors(g, "a") == []


def test_ancestors_walks_three_level_chain() -> None:
    """Deep chain: F → D → C → A. Coverage for >2 levels."""
    g = graph.build([
        _row("a"),
        _row("c", parent_id="a", parent_seq=5),
        _row("d", parent_id="c", parent_seq=10),
        _row("f", parent_id="d", parent_seq=15),
    ])
    assert graph.ancestors(g, "f") == ["a", "c", "d"]


def test_subtree_includes_self_and_all_descendants_bf() -> None:
    g = _tree()
    sub_a = graph.subtree(g, "a")
    assert sub_a[0] == "a"
    assert set(sub_a) == {"a", "c", "d", "e"}
    assert "b" not in sub_a


def test_subtree_on_leaf_returns_just_self() -> None:
    g = _tree()
    assert graph.subtree(g, "d") == ["d"]


def test_children_of_lists_direct_children() -> None:
    g = _tree()
    assert sorted(graph.children_of(g, "a")) == ["c", "e"]
    assert graph.children_of(g, "c") == ["d"]
    assert graph.children_of(g, "d") == []


def test_is_leaf_true_for_nodes_with_no_children() -> None:
    """``delete`` accepts only leaves — this is the predicate."""
    g = _tree()
    assert graph.is_leaf(g, "d") is True
    assert graph.is_leaf(g, "e") is True
    assert graph.is_leaf(g, "b") is True
    assert graph.is_leaf(g, "a") is False
    assert graph.is_leaf(g, "c") is False


def test_parent_of_returns_direct_parent_or_none_for_roots() -> None:
    g = _tree()
    assert graph.parent_of(g, "d") == "c"
    assert graph.parent_of(g, "c") == "a"
    assert graph.parent_of(g, "a") is None


def test_parent_seq_of_returns_cutoff_for_chain_assembly() -> None:
    g = _tree()
    assert graph.parent_seq_of(g, "c") == 5
    assert graph.parent_seq_of(g, "d") == 10
    assert graph.parent_seq_of(g, "a") is None


def test_walk_ancestors_returns_pairs_of_ancestor_id_and_seq_cap() -> None:
    """``walk_ancestors`` returns ``[(ancestor_id, cap), …]``.

    The cap on each entry is the ``parent_seq`` of the link that brought
    the ancestor into the chain (i.e. its child in the chain).
    """
    g = _tree()
    assert graph.walk_ancestors(g, "d") == [("a", 5), ("c", 10)]


def test_walk_ancestors_override_link_for_in_flight_fork() -> None:
    """When ``start_id`` isn't in the graph, use the override link.

    ``Session.fork`` uses this to walk the new fork's ancestors before
    its lazy row is flushed, without mutating the graph.
    """
    g = _tree()
    chain = graph.walk_ancestors(
        g, "f",
        start_parent_id="d",
        start_parent_seq=20,
    )
    assert chain == [("a", 5), ("c", 10), ("d", 20)]


def test_walk_ancestors_override_ignored_when_start_in_graph() -> None:
    """If ``start_id`` is in the graph, the stored link wins."""
    g = _tree()
    chain = graph.walk_ancestors(
        g, "d", start_parent_id="bogus", start_parent_seq=99
    )
    assert chain == [("a", 5), ("c", 10)]


def test_walk_ancestors_root_returns_empty() -> None:
    g = _tree()
    assert graph.walk_ancestors(g, "a") == []


def test_unknown_session_raises() -> None:
    g = _tree()
    with pytest.raises(SessionNotFoundError):
        graph.ancestors(g, "zzz")
    with pytest.raises(SessionNotFoundError):
        graph.is_leaf(g, "zzz")


def test_empty_graph_has_no_roots() -> None:
    g = graph.build([])
    assert graph.roots(g) == []
    assert len(g) == 0


# ---------------------------------------------------------------------
# children_index + indexed variants (#271 perf)
#
# Pre-#271: ``children_of`` / ``is_leaf`` / ``subtree`` scan the whole
# graph per call (O(N)). For loops that walk >1 node, callers should
# build ``children_index`` once and use the ``*_indexed`` variants —
# O(answer-size) instead of O(N²).
# ---------------------------------------------------------------------


def test_children_index_matches_children_of_for_all_nodes() -> None:
    """``children_index`` produces the same answers as ``children_of``."""
    g = _tree()
    idx = graph.children_index(g)
    for sid in g:
        assert idx.get(sid, []) == graph.children_of(g, sid), (
            f"index mismatch for {sid}: idx={idx.get(sid)}, scan={graph.children_of(g, sid)}"
        )


def test_children_indexed_is_o1_lookup() -> None:
    """``children_of_indexed`` returns the indexed list directly (no scan)."""
    g = _tree()
    idx = graph.children_index(g)
    # a → {c, e} per _tree()
    assert sorted(graph.children_of_indexed(g, idx, "a")) == ["c", "e"]
    # b is a leaf root → empty
    assert graph.children_of_indexed(g, idx, "b") == []
    # d (a leaf under c) → empty
    assert graph.children_of_indexed(g, idx, "d") == []


def test_is_leaf_indexed_matches_is_leaf_for_all_nodes() -> None:
    """``is_leaf_indexed`` agrees with the scan-based ``is_leaf``."""
    g = _tree()
    idx = graph.children_index(g)
    for sid in g:
        assert graph.is_leaf_indexed(g, idx, sid) is graph.is_leaf(g, sid), (
            f"is_leaf mismatch for {sid}"
        )


def test_subtree_indexed_matches_subtree_for_all_nodes() -> None:
    """``subtree_indexed`` produces the same BF traversal as ``subtree``."""
    g = _tree()
    idx = graph.children_index(g)
    for sid in g:
        assert graph.subtree_indexed(g, idx, sid) == graph.subtree(g, sid), (
            f"subtree mismatch for {sid}"
        )


def test_subtree_indexed_skips_full_graph_scan_per_iteration() -> None:
    """AC #271: ``subtree_indexed`` reads ``children.get(...)`` — no ``graph.values()`` walk.

    Regression guard for the perf claim. The non-indexed ``subtree`` calls
    ``graph.values()`` once per descendant; the indexed version calls it
    zero times (children dict is pre-built). Verified via a custom Mapping
    subclass whose ``values()`` raises if invoked.
    """
    from collections.abc import Mapping

    class _NoValuesDict(dict, Mapping):
        """``dict`` subclass whose ``values()`` raises (scan detector)."""
        def values(self):  # type: ignore[override]
            raise AssertionError(
                "subtree_indexed must not call graph.values() — "
                "the index is pre-built"
            )

    # Seed the no-values dict with the tree's rows.
    g = _NoValuesDict(_tree())
    idx = graph.children_index(_tree())  # build index from a plain copy
    out = graph.subtree_indexed(g, idx, "a")
    assert set(out) == {"a", "c", "d", "e"}
