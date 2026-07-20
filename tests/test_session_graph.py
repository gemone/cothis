"""Tests for ``cothis.session.graph.SessionGraph`` — the in-memory fork tree.

Covers the operations the CLI/Session use: ``roots``, ``ancestors``,
``subtree``, ``children_of``, ``is_leaf``, ``add``. Tree shape:

- root A (no parent)
- root B (no parent)
- A → C (parent_seq=5)
- C → D (parent_seq=10)
- A → E (parent_seq=3)

Plus the leaf-only delete predicate and the lazy-add path used by
``Session.fork`` after the row is written.
"""

from __future__ import annotations

import pytest

from cothis.session.graph import SessionGraph, SessionNotFoundError
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


def _build_tree() -> SessionGraph:
    """A → {C, E}; C → D. Two roots (A, B)."""
    return SessionGraph(
        [
            _row("a"),
            _row("b"),
            _row("c", parent_id="a", parent_seq=5),
            _row("d", parent_id="c", parent_seq=10),
            _row("e", parent_id="a", parent_seq=3),
        ]
    )


def test_roots_returns_all_rootless_nodes() -> None:
    """``roots`` lists every node whose ``parent_id`` is NULL."""
    g = _build_tree()
    assert sorted(g.roots()) == ["a", "b"]


def test_ancestors_walks_root_to_direct_parent() -> None:
    """``ancestors`` returns the chain from root down to the direct parent.

    Order matters for ancestor-chain context assembly — context must
    prepend root first, then deeper ancestors, then the current
    session's own blocks last.
    """
    g = _build_tree()
    assert g.ancestors("d") == ["a", "c"]
    assert g.ancestors("c") == ["a"]
    assert g.ancestors("a") == []


def test_ancestors_handles_multi_root_tree() -> None:
    """Each root has an empty ancestor chain."""
    g = _build_tree()
    assert g.ancestors("a") == []
    assert g.ancestors("b") == []


def test_subtree_includes_self_and_all_descendants_bf() -> None:
    """``subtree(sid)`` is ``sid`` plus every descendant, BF order."""
    g = _build_tree()
    sub_a = g.subtree("a")
    assert sub_a[0] == "a"
    assert set(sub_a) == {"a", "c", "d", "e"}
    # B and its (empty) subtree are not in A's subtree.
    assert "b" not in sub_a


def test_subtree_on_leaf_returns_just_self() -> None:
    g = _build_tree()
    assert g.subtree("d") == ["d"]


def test_children_of_lists_direct_children() -> None:
    g = _build_tree()
    assert sorted(g.children_of("a")) == ["c", "e"]
    assert g.children_of("c") == ["d"]
    assert g.children_of("d") == []


def test_is_leaf_true_for_nodes_with_no_children() -> None:
    """``delete`` accepts only leaves — this is the predicate it queries."""
    g = _build_tree()
    assert g.is_leaf("d") is True
    assert g.is_leaf("e") is True
    assert g.is_leaf("b") is True
    assert g.is_leaf("a") is False
    assert g.is_leaf("c") is False


def test_parent_of_returns_direct_parent_or_none_for_roots() -> None:
    g = _build_tree()
    assert g.parent_of("d") == "c"
    assert g.parent_of("c") == "a"
    assert g.parent_of("a") is None


def test_parent_seq_of_returns_cutoff_for_chain_assembly() -> None:
    """``parent_seq_of`` is the inclusive seq cap on the parent's blocks."""
    g = _build_tree()
    assert g.parent_seq_of("c") == 5
    assert g.parent_seq_of("d") == 10
    assert g.parent_seq_of("a") is None


def test_add_registers_new_fork_link() -> None:
    """``add`` extends the graph with a freshly-forked row.

    ``Session.fork`` constructs the Session before its lazy row is
    flushed, so the in-memory graph must be augmented explicitly for
    the ancestor-chain walk to see the new node.
    """
    g = _build_tree()
    g.add(_row("f", parent_id="d", parent_seq=15))
    assert "f" in g
    assert g.parent_of("f") == "d"
    assert g.ancestors("f") == ["a", "c", "d"]
    assert g.is_leaf("d") is False  # d now has a child
    assert g.is_leaf("f") is True


def test_add_idempotent_on_known_id() -> None:
    """Re-adding a known id is a no-op (lazy row may already be flushed)."""
    g = _build_tree()
    g.add(_row("a"))  # a is already a root
    assert sorted(g.roots()) == ["a", "b"]


def test_unknown_session_raises() -> None:
    """Every operation validates its input against the known set."""
    g = _build_tree()
    with pytest.raises(SessionNotFoundError):
        g.ancestors("zzz")
    with pytest.raises(SessionNotFoundError):
        g.is_leaf("zzz")


def test_empty_graph_has_no_roots() -> None:
    g = SessionGraph([])
    assert g.roots() == []
    assert len(g) == 0
