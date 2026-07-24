"""Tests for ``NotifyBus.compact`` retention (#236).

The compaction job deletes ``notify_events`` rows older than the
configured retention window. Retention is read from
``COTHIS_NOTIFY_RETENTION_DAYS`` at compact time (0 or negative =
disabled, no rows deleted). Returns the count of deleted rows.

Snapshot-preservation (skipping rows pinned by an active consumer
cursor) lands here per #246. ``compact`` takes an optional
``preserve_seqs: set[int] | None``; pinned ``seq`` values survive
compaction regardless of age. The cursor helper
``NotifyBus.preserve_since(*cursors)`` builds that set from the
minimum live consumer high-water mark (Supervisor checkpoint / TUI
``last_seq``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from cothis.notify import NotifyBus

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _make_bus() -> tuple[NotifyBus, Any]:
    """Build a NotifyBus backed by an in-memory SQLite DB."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    bus = NotifyBus(conn)
    return bus, conn


def _seed_old_event(bus: NotifyBus, conn: Any, days_old: int) -> int:
    """Insert an event with ``ts`` set to N days ago; return its seq."""
    old_ts = (datetime.now(UTC) - timedelta(days=days_old)).isoformat()
    with conn:
        cur = conn.execute(
            "INSERT INTO notify_events(ts, topic, event_type, session_id, meta, payload_pointer) "
            "VALUES (?, 'tool_call', 'completed', NULL, NULL, NULL)",
            (old_ts,),
        )
    return int(cur.lastrowid)


def _count_rows(conn: Any) -> int:
    return conn.execute("SELECT COUNT(*) FROM notify_events").fetchone()[0]


# ---------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------


def test_compact_deletes_rows_older_than_retention_window(tmp_path: Path) -> None:
    """Rows older than the cutoff are deleted; younger ones stay."""
    bus, conn = _make_bus()

    _seed_old_event(bus, conn, days_old=45)  # older than 30d
    _seed_old_event(bus, conn, days_old=60)  # older than 30d
    fresh_seq = bus.append(topic="tool_call", event_type="completed")  # today

    assert _count_rows(conn) == 3
    deleted = bus.compact(retention_days=30)
    assert deleted == 2
    assert _count_rows(conn) == 1
    remaining = bus.fetch_since(last_seq=0)
    assert [e.seq for e in remaining] == [fresh_seq]


def test_compact_preserves_rows_within_window(tmp_path: Path) -> None:
    """Rows inside the retention window are kept."""
    bus, conn = _make_bus()

    _seed_old_event(bus, conn, days_old=10)  # within 30d
    _seed_old_event(bus, conn, days_old=20)  # within 30d
    bus.append(topic="tool_call", event_type="completed")

    deleted = bus.compact(retention_days=30)
    assert deleted == 0
    assert _count_rows(conn) == 3


def test_compact_just_past_boundary_is_deleted(tmp_path: Path) -> None:
    """A row 31 days old is deleted under retention_days=30 (deterministic)."""
    bus, conn = _make_bus()
    _seed_old_event(bus, conn, days_old=31)

    deleted = bus.compact(retention_days=30)
    assert deleted == 1


# ---------------------------------------------------------------------
# Disabled cases
# ---------------------------------------------------------------------


def test_compact_zero_retention_is_noop(tmp_path: Path) -> None:
    """retention_days=0 → no rows deleted."""
    bus, conn = _make_bus()
    _seed_old_event(bus, conn, days_old=365)

    deleted = bus.compact(retention_days=0)
    assert deleted == 0
    assert _count_rows(conn) == 1


def test_compact_negative_retention_is_noop(tmp_path: Path) -> None:
    """Negative retention_days → no rows deleted (defensive)."""
    bus, conn = _make_bus()
    _seed_old_event(bus, conn, days_old=365)

    deleted = bus.compact(retention_days=-1)
    assert deleted == 0
    assert _count_rows(conn) == 1


def test_compact_empty_table(tmp_path: Path) -> None:
    """Empty table → deleted=0, no error."""
    bus, conn = _make_bus()
    deleted = bus.compact(retention_days=30)
    assert deleted == 0


# ---------------------------------------------------------------------
# Env var COTHIS_NOTIFY_RETENTION_DAYS
# ---------------------------------------------------------------------


def test_compact_reads_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """compact() with no arg reads COTHIS_NOTIFY_RETENTION_DAYS."""
    bus, conn = _make_bus()
    _seed_old_event(bus, conn, days_old=45)

    monkeypatch.setenv("COTHIS_NOTIFY_RETENTION_DAYS", "30")
    deleted = bus.compact()
    assert deleted == 1


def test_compact_env_var_zero_disables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """COTHIS_NOTIFY_RETENTION_DAYS=0 → disabled."""
    bus, conn = _make_bus()
    _seed_old_event(bus, conn, days_old=365)

    monkeypatch.setenv("COTHIS_NOTIFY_RETENTION_DAYS", "0")
    deleted = bus.compact()
    assert deleted == 0


def test_compact_env_var_unset_disables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var unset → no compaction (opt-in feature)."""
    bus, conn = _make_bus()
    _seed_old_event(bus, conn, days_old=365)

    monkeypatch.delenv("COTHIS_NOTIFY_RETENTION_DAYS", raising=False)
    deleted = bus.compact()
    assert deleted == 0


def test_compact_env_var_garbage_disables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-integer env var → no compaction, no crash."""
    bus, conn = _make_bus()
    _seed_old_event(bus, conn, days_old=365)

    monkeypatch.setenv("COTHIS_NOTIFY_RETENTION_DAYS", "garbage")
    deleted = bus.compact()
    assert deleted == 0


# ---------------------------------------------------------------------
# Integration: real Storage-backed connection + seeded old rows
# ---------------------------------------------------------------------


def test_integration_storage_backed_bus_compact(tmp_path: Path) -> None:
    """End-to-end against a Storage-created DB (exercises the
    real connection + schema)."""
    from cothis.session.storage import Storage

    storage = Storage(tmp_path / "test.db")
    try:
        bus = NotifyBus(storage._conn)
        # Seed one old row via SQL (append always uses now).
        old_ts = (datetime.now(UTC) - timedelta(days=45)).isoformat()
        with storage._conn:
            storage._conn.execute(
                "INSERT INTO notify_events(ts, topic, event_type, session_id, meta, payload_pointer) "
                "VALUES (?, 'tool_call', 'completed', NULL, NULL, NULL)",
                (old_ts,),
            )
        # And a fresh row via the bus.
        bus.append(topic="tool_call", event_type="completed")

        deleted = bus.compact(retention_days=30)
        assert deleted == 1
        events = bus.fetch_since(last_seq=0)
        assert len(events) == 1
    finally:
        storage.close()


# ---------------------------------------------------------------------
# preserve_seqs / preserve_since — snapshot preservation (#246)
# ---------------------------------------------------------------------


def test_compact_preserve_seqs_keeps_pinned_old_row() -> None:
    """An old row whose seq is in preserve_seqs survives compaction."""
    bus, conn = _make_bus()
    pinned_seq = _seed_old_event(bus, conn, days_old=45)  # would be deleted
    _seed_old_event(bus, conn, days_old=60)  # would be deleted

    deleted = bus.compact(retention_days=30, preserve_seqs={pinned_seq})

    assert deleted == 1  # only the unpinned old row removed
    remaining = [e.seq for e in bus.fetch_since(last_seq=0)]
    assert pinned_seq in remaining


def test_compact_preserve_seqs_does_not_protect_young_rows_in_count() -> None:
    """preserve_seqs only changes the fate of rows that would otherwise be
    deleted; rows within the window are unaffected, and the returned
    deleted-count reflects only actually-removed rows."""
    bus, conn = _make_bus()
    pinned_old = _seed_old_event(bus, conn, days_old=45)
    _seed_old_event(bus, conn, days_old=45)  # unpinned old
    bus.append(topic="tool_call", event_type="completed")  # young

    deleted = bus.compact(retention_days=30, preserve_seqs={pinned_old})

    assert deleted == 1
    assert _count_rows(conn) == 2  # pinned old + young


def test_compact_empty_preserve_seqs_is_equivalent_to_none() -> None:
    """An empty preserve_seqs set behaves like no pinning (no rows saved)."""
    bus, conn = _make_bus()
    _seed_old_event(bus, conn, days_old=45)

    deleted = bus.compact(retention_days=30, preserve_seqs=set())

    assert deleted == 1


def test_compact_preserve_seqs_unknown_seq_is_harmless() -> None:
    """A preserve_seqs value with no matching row is ignored (no error)."""
    bus, conn = _make_bus()
    _seed_old_event(bus, conn, days_old=45)

    deleted = bus.compact(retention_days=30, preserve_seqs={999_999})

    assert deleted == 1  # the real old row still goes


def test_preserve_since_returns_floor_prefix() -> None:
    """preserve_since(*cursors) returns every seq at or below the min cursor."""
    bus, conn = _make_bus()
    s1 = bus.append(topic="t", event_type="e")
    s2 = bus.append(topic="t", event_type="e")
    s3 = bus.append(topic="t", event_type="e")
    s4 = bus.append(topic="t", event_type="e")

    pinned = bus.preserve_since(3)  # one consumer at last_seq=3

    assert pinned == {s1, s2, s3}
    assert s4 not in pinned


def test_preserve_since_takes_minimum_of_multiple_cursors() -> None:
    """The lagging consumer's cursor determines the floor."""
    bus, _ = _make_bus()
    s1 = bus.append(topic="t", event_type="e")
    s2 = bus.append(topic="t", event_type="e")
    s3 = bus.append(topic="t", event_type="e")
    bus.append(topic="t", event_type="e")  # s4, ahead of the floor

    # Two consumers: one read up to 3, one lagging at 2 → floor = 2.
    pinned = bus.preserve_since(3, 2)

    assert pinned == {s1, s2}


def test_preserve_since_zero_cursor_pins_nothing() -> None:
    """A cursor of 0 (nothing read yet) contributes nothing; an all-zero
    call returns an empty set (no pinning, safe to pass through)."""
    bus, _ = _make_bus()
    bus.append(topic="t", event_type="e")

    assert bus.preserve_since(0) == set()
    assert bus.preserve_since() == set()


def test_preserve_since_then_compact_end_to_end() -> None:
    """The documented Supervisor flow: build the pin set from cursors,
    then compact — the lagging consumer's unread prefix survives."""
    bus, conn = _make_bus()
    # Two old events a lagging TUI hasn't fully read, plus one it has.
    old1 = _seed_old_event(bus, conn, days_old=45)  # unread by lagging TUI
    old2 = _seed_old_event(bus, conn, days_old=45)  # the lagging cursor
    _seed_old_event(bus, conn, days_old=60)  # ahead, unpinned → deleted

    pinned = bus.preserve_since(old2)
    assert old1 in pinned and old2 in pinned

    deleted = bus.compact(retention_days=30, preserve_seqs=pinned)

    assert deleted == 1  # only the unpinned 60-day row removed
    remaining = {e.seq for e in bus.fetch_since(last_seq=0)}
    assert {old1, old2}.issubset(remaining)
