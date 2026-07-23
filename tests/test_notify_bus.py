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
