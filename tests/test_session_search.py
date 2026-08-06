"""Tests for ``Storage.search`` — the FTS5 full-text index over ``blocks``.

Covers the iteration-1 search contract:

- **migration + backfill**: a DB written at ``SCHEMA_VERSION=2`` (pre-FTS)
  is re-opened and the legacy rows become searchable with no explicit
  reindex — the v2→v3 migration's ``'rebuild'`` repopulates the index.
- **trigger sync on insert / delete**: the AFTER-row triggers keep the
  index in step with every write path; appends appear, deletes vanish.
- **search semantics**: multi-row MATCH, BM25 rank ordering (more-relevant
  first), phrase queries (``"..."``), prefix queries (``term*``), and a
  no-match query returns ``[]`` (not an error).
- **limit cap**: ``limit=k`` truncates the result list.
- **non-text blocks stay quiet**: a ``tool_use`` / ``tool_result`` block
  has ``NULL`` ``content`` and is absent from the index, so unrelated
  queries never surface it.
- **graceful FTS5 absence**: when the index is unavailable, ``search``
  raises a clear ``sqlite3.OperationalError`` instead of running a doomed
  query or corrupting state.

All tests are offline: they build sessions via the ``Session`` API against
a temp DB and assert on ``Storage.search`` output.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from cothis.session import Session
from cothis.session.storage import SCHEMA_VERSION, SearchHit, Storage

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _user_text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _assistant_tool_use(tool_id: str, tool_name: str = "fs.read") -> dict[str, Any]:
    """A ``tool_use`` block — has ``tool_input`` JSON but ``NULL`` ``content``."""
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": tool_name,
        "input": {"path": "/etc/hosts"},
    }


def _tool_result(tool_id: str, content: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": content,
    }


def _make_session(
    db_path: Path,
    cwd: Path,
    *,
    model: str = "m",
    texts: list[str] | None = None,
) -> str:
    """Create a session, append ``texts`` as user/assistant alternation, return id.

    Even-indexed ``texts`` become user, odd-indexed become assistant, so each
    lands in its own ``msg_idx`` (the alternation invariant forces merges
    otherwise).
    """
    s = Session.new(db_path, cwd=cwd, model=model, flush_sync=True)
    sid = s.session_id
    for i, t in enumerate(texts or []):
        role = "user" if i % 2 == 0 else "assistant"
        s.append_message(role, [_user_text(t)])
    s.close()
    return sid


def _downgrade_to_v2(db_path: Path) -> None:
    """Strip the FTS5 index + triggers and pin ``user_version=2``.

    Simulates a legacy DB written before #I21 so the next ``Storage`` open
    exercises the v2→v3 migration + backfill path (not the fresh-DDL path).
    """
    conn = sqlite3.connect(db_path)
    try:
        # Triggers must drop before the virtual table (they reference it).
        for trg in ("blocks_fts_ai", "blocks_fts_ad", "blocks_fts_au"):
            conn.execute(f"DROP TRIGGER IF EXISTS {trg}")
        conn.execute("DROP TABLE IF EXISTS blocks_fts")
        conn.execute("PRAGMA user_version=2")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------
# migration + backfill
# ---------------------------------------------------------------------


def test_migration_backfills_index_for_legacy_db(tmp_path: Path) -> None:
    """A v2 DB with rows is searchable on the first v3 reopen (backfill)."""
    db_path = tmp_path / "session.db"
    _make_session(db_path, tmp_path, texts=["deploy the alpha service"])
    _downgrade_to_v2(db_path)

    storage = Storage(db_path)
    try:
        # The v3 schema (virtual table + triggers) is now present.
        kind = storage._conn.execute(
            "SELECT type FROM sqlite_master WHERE name='blocks_fts'"
        ).fetchone()
        assert kind is not None and kind[0] == "table"
        trg = storage._conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='trigger' "
            "AND name LIKE 'blocks_fts_%'"
        ).fetchone()
        assert trg[0] == 3
        assert SCHEMA_VERSION == 3

        # The legacy row was indexed by the one-time 'rebuild' backfill.
        hits = storage.search("alpha")
        assert len(hits) == 1
        assert "alpha" in hits[0].snippet
    finally:
        storage.close()


def test_fresh_db_is_searchable_without_explicit_reindex(tmp_path: Path) -> None:
    """A brand-new DB creates the FTS table on construction and works immediately."""
    db_path = tmp_path / "session.db"
    _make_session(db_path, tmp_path, texts=["hello world brand new"])
    storage = Storage(db_path)
    try:
        hits = storage.search("brand")
        assert len(hits) == 1
        assert hits[0].session_id is not None
    finally:
        storage.close()


# ---------------------------------------------------------------------
# trigger sync
# ---------------------------------------------------------------------


def test_trigger_sync_on_insert(tmp_path: Path) -> None:
    """Newly appended text is searchable with no explicit reindex."""
    db_path = tmp_path / "session.db"
    _make_session(db_path, tmp_path, texts=["first turn"])

    storage = Storage(db_path)
    try:
        assert storage.search("second") == []
    finally:
        storage.close()

    # Reopen the session and append — the AFTER INSERT trigger indexes it.
    sid = Session.list_visible(db_path, tmp_path)[0].id
    s = Session.load(db_path, sid, cwd=tmp_path, flush_sync=True)
    s.append_message("assistant", [_user_text("the second turn arrived")])
    s.close()

    storage = Storage(db_path)
    try:
        hits = storage.search("second")
        assert len(hits) == 1
        assert "second" in hits[0].snippet
    finally:
        storage.close()


def test_trigger_sync_on_delete(tmp_path: Path) -> None:
    """``Storage.delete_session`` removes the row from the index."""
    db_path = tmp_path / "session.db"
    sid = _make_session(db_path, tmp_path, texts=["solo searchable term"])

    storage = Storage(db_path)
    try:
        assert len(storage.search("solo")) == 1
        storage.delete_session(sid)
        assert storage.search("solo") == []
    finally:
        storage.close()


def test_trigger_sync_on_update(tmp_path: Path) -> None:
    """An ``UPDATE`` to a block's content is reflected in the index.

    The AFTER UPDATE trigger deletes the old FTS row and inserts the new
    one, so rewriting a block's searchable text is picked up with no
    explicit reindex — and the old wording no longer matches. Covers the
    third mutating path (insert/delete tested above) the plan promised.
    """
    db_path = tmp_path / "session.db"
    sid = _make_session(db_path, tmp_path, texts=["the original wording"])

    storage = Storage(db_path)
    try:
        assert len(storage.search("original")) == 1
        assert storage.search("rewritten") == []
        # Rewrite the block's content on the same connection the triggers
        # fire through, mirroring how ``archive_skill_blocks`` issues its
        # ``UPDATE blocks SET state='archived'`` (the real UPDATE path).
        storage._conn.execute(
            "UPDATE blocks SET content=? WHERE session_id=? AND content=?",
            ("the rewritten wording", sid, "the original wording"),
        )
        storage._conn.commit()
        # New wording is indexed; the old wording is gone (delete + reinsert).
        rewritten = storage.search("rewritten")
        assert len(rewritten) == 1
        assert "rewritten" in rewritten[0].snippet
        assert storage.search("original") == []
    finally:
        storage.close()


# ---------------------------------------------------------------------
# search semantics
# ---------------------------------------------------------------------


def test_search_returns_multi_row_matches_ranked(tmp_path: Path) -> None:
    """Multiple matches return ordered by BM25 relevance (more-relevant first).

    The doc that repeats the term more often ranks higher (lower ``rank``).
    """
    db_path = tmp_path / "session.db"
    # Three sessions, each with a single user message; the odd one repeats
    # the term so BM25 ranks it most relevant.
    sid_dense = _make_session(db_path, tmp_path, texts=["alpha alpha alpha beta"])
    _make_session(db_path, tmp_path, texts=["alpha gamma"])
    _make_session(db_path, tmp_path, texts=["delta alpha"])

    storage = Storage(db_path)
    try:
        hits = storage.search("alpha")
        assert len(hits) == 3
        # Most-relevant (densest) first.
        assert hits[0].session_id == sid_dense
        # rank is monotonic non-decreasing over the ordered list.
        ranks = [h.rank for h in hits]
        assert ranks == sorted(ranks)
    finally:
        storage.close()


def test_phrase_and_prefix_queries(tmp_path: Path) -> None:
    """FTS5 phrase (``"..."``) and prefix (``term*``) expressions work pass-through."""
    db_path = tmp_path / "session.db"
    _make_session(
        db_path,
        tmp_path,
        texts=["the deployment pipeline failed", "a separate unrelated note"],
    )

    storage = Storage(db_path)
    try:
        phrase = storage.search('"deployment pipeline"')
        assert len(phrase) == 1
        assert "deployment" in phrase[0].snippet

        prefix = storage.search("deploy*")
        assert len(prefix) == 1
    finally:
        storage.close()


def test_search_no_match_returns_empty_not_error(tmp_path: Path) -> None:
    """A query that matches nothing returns ``[]`` (no exception)."""
    db_path = tmp_path / "session.db"
    _make_session(db_path, tmp_path, texts=["real content here"])

    storage = Storage(db_path)
    try:
        assert storage.search("zzzznomatch") == []
    finally:
        storage.close()


def test_search_limit_truncates(tmp_path: Path) -> None:
    """``limit=k`` caps the number of returned hits."""
    db_path = tmp_path / "session.db"
    # Five sessions, all containing the shared term so each is one hit.
    for i in range(5):
        _make_session(
            db_path, tmp_path, texts=[f"common term number {i}"]
        )

    storage = Storage(db_path)
    try:
        full = storage.search("common")
        assert len(full) == 5
        capped = storage.search("common", limit=2)
        assert len(capped) == 2
    finally:
        storage.close()


# ---------------------------------------------------------------------
# non-text blocks
# ---------------------------------------------------------------------


def test_non_text_blocks_are_not_indexed(tmp_path: Path) -> None:
    """A ``tool_use`` block (``NULL`` content) never surfaces in text search.

    FTS5 ignores NULL content, so a session holding only structured blocks
    produces no hits for any text query — the tool input JSON stays out of
    the text index (folding it in is deferred ranking tuning).
    """
    db_path = tmp_path / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    sid = s.session_id
    # Assistant tool_use → user tool_result. Neither has ``content`` text.
    s.append_message(
        "assistant", [_assistant_tool_use("tu1", "fs.read")]
    )
    s.append_block("user", _tool_result("tu1", "0.0.0.0 localhost"))
    s.close()

    storage = Storage(db_path)
    try:
        # Tokens that live only in tool_input / tool_output / tool_name JSON
        # return nothing — those columns are not in the FTS index. Each query
        # is a valid bare-token FTS5 expression (the indexed ``content`` is
        # NULL for these blocks, so there is simply nothing to match).
        assert storage.search("localhost") == []
        assert storage.search("etc") == []
        assert storage.search("hosts") == []
        # Sanity: the session itself exists; the index just doesn't surface it.
        assert storage.load_session(sid) is not None
    finally:
        storage.close()


# ---------------------------------------------------------------------
# graceful FTS5 absence
# ---------------------------------------------------------------------


def test_search_raises_clear_error_when_fts5_unavailable(
    tmp_path: Path,
) -> None:
    """If the FTS5 index is unavailable, ``search`` raises a clear error.

    A sqlite build lacking the FTS5 extension is detected at ``Storage``
    construction (the virtual-table create fails). This test simulates that
    degraded state by toggling the availability flag a real no-FTS5 build
    sets, and asserts the user-facing contract: ``search`` raises
    ``sqlite3.OperationalError`` (a catchable error with a message), never
    a silent empty list or a raw crash.
    """
    db_path = tmp_path / "session.db"
    _make_session(db_path, tmp_path, texts=["indexed before fts went away"])

    storage = Storage(db_path)
    try:
        storage._fts5_available = False
        with pytest.raises(sqlite3.OperationalError):
            storage.search("indexed")
    finally:
        storage.close()


# ---------------------------------------------------------------------
# SearchHit shape
# ---------------------------------------------------------------------


def test_search_hit_fields_are_storage_shaped(tmp_path: Path) -> None:
    """``SearchHit`` exposes the documented fields with usable snippet + rank."""
    db_path = tmp_path / "session.db"
    _make_session(db_path, tmp_path, texts=["matchable content"])

    storage = Storage(db_path)
    try:
        hits = storage.search("matchable")
        assert len(hits) == 1
        hit: SearchHit = hits[0]
        assert isinstance(hit, SearchHit)
        assert isinstance(hit.session_id, str)
        assert isinstance(hit.seq, int)
        assert isinstance(hit.msg_idx, int)
        assert hit.role in ("user", "assistant")
        assert hit.type == "text"
        assert "matchable" in hit.snippet
        assert isinstance(hit.rank, float)
    finally:
        storage.close()
