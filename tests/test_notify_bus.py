"""Tests for ``cothis.notify.NotifyBus`` (#223).

Durable append-log over the per-session SQLite DB. ``seq`` is
monotonic + unique (AUTOINCREMENT primary key); consumers dedupe by
``seq``. Heavy payloads live in session storage; events carry only
metadata + a pointer.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import TYPE_CHECKING

import pytest

from cothis.notify import NotifyBus, NotifyEvent
from cothis.session.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[Storage]:
    s = Storage(tmp_path / "test.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def bus(storage: Storage) -> NotifyBus:
    return NotifyBus(storage._conn)


def test_notify_bus_append_returns_monotonic_seq(bus: NotifyBus) -> None:
    seq1 = bus.append(topic="tool_call", event_type="started")
    seq2 = bus.append(topic="tool_call", event_type="completed")
    seq3 = bus.append(topic="agent_message", event_type="delta")
    assert seq1 < seq2 < seq3
    assert seq1 >= 1


def test_notify_bus_fetch_since_filters_by_seq(bus: NotifyBus) -> None:
    s1 = bus.append(topic="t", event_type="a")
    s2 = bus.append(topic="t", event_type="b")
    s3 = bus.append(topic="t", event_type="c")

    events = bus.fetch_since(last_seq=0)
    assert [e.seq for e in events] == [s1, s2, s3]

    events = bus.fetch_since(last_seq=s1)
    assert [e.seq for e in events] == [s2, s3]

    events = bus.fetch_since(last_seq=s3)
    assert events == []


def test_notify_bus_fetch_since_filters_by_session_id(bus: NotifyBus) -> None:
    bus.append(topic="t", event_type="a", session_id="s1")
    bus.append(topic="t", event_type="b", session_id="s2")
    bus.append(topic="t", event_type="c", session_id="s1")

    s1_events = bus.fetch_since(last_seq=0, session_id="s1")
    assert len(s1_events) == 2
    assert all(e.session_id == "s1" for e in s1_events)

    s2_events = bus.fetch_since(last_seq=0, session_id="s2")
    assert len(s2_events) == 1
    assert s2_events[0].session_id == "s2"


def test_notify_bus_append_preserves_payload_and_meta(bus: NotifyBus) -> None:
    seq = bus.append(
        topic="tool_call",
        event_type="completed",
        session_id="abc",
        meta={"tool": "fs.read", "duration_ms": 42, "ok": True},
        payload_pointer="session:abc:tool:call_1",
    )
    events = bus.fetch_since(last_seq=0)
    assert len(events) == 1
    e = events[0]
    assert e.seq == seq
    assert e.topic == "tool_call"
    assert e.event_type == "completed"
    assert e.session_id == "abc"
    assert e.meta == {"tool": "fs.read", "duration_ms": 42, "ok": True}
    assert e.payload_pointer == "session:abc:tool:call_1"


def test_notify_bus_append_without_optional_fields(bus: NotifyBus) -> None:
    seq = bus.append(topic="t", event_type="a")
    e = bus.fetch_since(last_seq=0)[0]
    assert e.seq == seq
    assert e.session_id is None
    assert e.meta is None
    assert e.payload_pointer is None


def test_notify_bus_table_created_lazily_on_existing_db(
    tmp_path: Path,
) -> None:
    db = tmp_path / "existing.db"
    s = Storage(db)
    s.close()

    raw = sqlite3.connect(db)
    try:
        cols = {r[1] for r in raw.execute("PRAGMA table_info(notify_events)")}
    finally:
        raw.close()
    assert cols == set(), "notify_events must not exist before NotifyBus init"

    s2 = Storage(db)
    try:
        NotifyBus(s2._conn)
        raw2 = sqlite3.connect(db)
        try:
            cols = {r[1] for r in raw2.execute("PRAGMA table_info(notify_events)")}
        finally:
            raw2.close()
        assert cols == {
            "seq", "ts", "topic", "event_type",
            "session_id", "meta", "payload_pointer",
        }
    finally:
        s2.close()


def test_notify_bus_concurrent_appends_are_safe(bus: NotifyBus) -> None:
    """N threads each append M events; all land with unique monotonic seqs."""
    n_threads = 4
    per_thread = 25

    def worker(tid: int) -> list[int]:
        seqs: list[int] = []
        for i in range(per_thread):
            seq = bus.append(
                topic="t",
                event_type="a",
                session_id=f"s{tid}",
                meta={"tid": tid, "i": i},
            )
            seqs.append(seq)
        return seqs

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    events = bus.fetch_since(last_seq=0)
    assert len(events) == n_threads * per_thread
    seqs = [e.seq for e in events]
    assert len(set(seqs)) == len(seqs), "seqs must be unique"
    assert seqs == sorted(seqs), "seqs must be monotonic"


def test_notify_event_is_namedtuple_for_call_site_clarity() -> None:
    e = NotifyEvent(
        seq=1, ts="2026-07-23", topic="t", event_type="a",
        session_id=None, meta=None, payload_pointer=None,
    )
    assert e.seq == 1
    assert e.topic == "t"


# ---------------------------------------------------------------------
# Redundant index guard (#263)
#
# ``seq`` is INTEGER PRIMARY KEY → aliases rowid → already has a
# covering B-tree. An explicit ``idx_notify_seq`` doubles INSERT write
# amplification for zero query-plan benefit. These tests guard the
# removal + the lazy migration (DROP IF EXISTS on reopen).
# ---------------------------------------------------------------------


def _has_index(conn: sqlite3.Connection, name: str) -> bool:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (name,),
    ).fetchall()
    return bool(rows)


def test_notify_bus_does_not_create_redundant_seq_index(bus: NotifyBus) -> None:
    """AC #263: ``idx_notify_seq`` must not be created — rowid PK covers it."""
    assert not _has_index(bus._conn, "idx_notify_seq"), (
        "idx_notify_seq is redundant with the rowid PK and must not exist"
    )
    # Sanity: the genuine session-filter index is still there.
    assert _has_index(bus._conn, "idx_notify_session")


def test_notify_bus_drops_legacy_redundant_seq_index_on_reopen(
    tmp_path: Path,
) -> None:
    """AC #263 migration: legacy DBs with ``idx_notify_seq`` get it dropped."""
    db = tmp_path / "legacy.db"
    # Simulate a pre-#263 database: notify_events exists with the redundant index.
    s1 = Storage(db)
    try:
        NotifyBus(s1._conn)  # creates the table the modern way
        s1._conn.execute("CREATE INDEX idx_notify_seq ON notify_events(seq)")
        s1._conn.commit()
        assert _has_index(s1._conn, "idx_notify_seq"), "pre-seed: index should exist"
    finally:
        s1.close()

    # Reopen — DROP INDEX IF EXISTS in _DDL must remove it.
    s2 = Storage(db)
    try:
        NotifyBus(s2._conn)
        assert not _has_index(s2._conn, "idx_notify_seq"), (
            "legacy idx_notify_seq must be dropped on reopen"
        )
    finally:
        s2.close()


def test_notify_bus_fetch_since_uses_primary_key_not_seq_index(
    bus: NotifyBus,
) -> None:
    """AC #263 regression: planner uses the rowid PK, never ``idx_notify_seq``."""
    bus.append(topic="t", event_type="a")
    bus.append(topic="t", event_type="b")

    plan_rows = bus._conn.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT seq, ts, topic, event_type, session_id, meta, payload_pointer "
        "FROM notify_events WHERE seq > ? ORDER BY seq",
        (0,),
    ).fetchall()
    plan_text = " ".join(str(r[-1]) for r in plan_rows)
    assert "PRIMARY KEY" in plan_text, (
        f"expected rowid PK in plan, got: {plan_text!r}"
    )
    assert "idx_notify_seq" not in plan_text, (
        f"planner must not reference dropped index, got: {plan_text!r}"
    )
