"""``cothis.notify`` — durable notify bus over the per-session SQLite DB.

A thin append-log layered on the session storage connection. Events
are metadata + a payload pointer; heavy payloads stay in session
storage (``blocks`` table). ``seq`` is ``INTEGER PRIMARY KEY
AUTOINCREMENT``, so monotonic + unique — consumers dedupe by ``seq``.

Controlled by feature flag ``COTHIS_NOTIFY_BUS`` at the call site
(``Agent._execute_tool``); this module is always importable, it just
isn't wired up unless the flag is on.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


# Lazy DDL — runs on every ``NotifyBus.__init__``. CREATE TABLE IF NOT
# EXISTS makes re-opening an existing db a no-op. No migration
# framework for a single table (see ADR-0016).
_DDL = (
    """
    CREATE TABLE IF NOT EXISTS notify_events(
        seq    INTEGER PRIMARY KEY AUTOINCREMENT,
        ts     TEXT NOT NULL,
        topic  TEXT NOT NULL,
        event_type TEXT NOT NULL,
        session_id TEXT,
        meta   TEXT,
        payload_pointer TEXT)
    """,
    "CREATE INDEX IF NOT EXISTS idx_notify_seq ON notify_events(seq)",
    "CREATE INDEX IF NOT EXISTS idx_notify_session ON notify_events(session_id)",
)

_FETCH_COLUMNS = (
    "seq, ts, topic, event_type, session_id, meta, payload_pointer"
)


class NotifyEvent(NamedTuple):
    seq: int
    ts: str
    topic: str
    event_type: str
    session_id: str | None
    meta: dict | None
    payload_pointer: str | None


class NotifyBus:
    """Append-log over the per-session SQLite DB.

    Reuses an existing ``sqlite3.Connection`` (typically the one owned
    by ``Storage``). WAL + busy_timeout + sqlite3's deferred-transaction
    mode handle concurrent writes: two ``with self._conn:`` blocks
    cannot overlap on the same connection, and two connections writing
    concurrently serialize via SQLite's internal locking (WAL allows
    multiple readers, one writer at a time).

    Not thread-safe across connections in the same process: callers
    that share a connection across threads rely on sqlite3's own
    serialization (``check_same_thread=False``). Cross-process
    concurrent writers are safe (each opens its own connection to the
    same db_path; fcntl session lock prevents two Sessions on the same
    id, and the notify bus is a secondary client on the same db).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._append_lock = threading.Lock()
        for stmt in _DDL:
            self._conn.execute(stmt)
        self._conn.commit()

    def append(
        self,
        *,
        topic: str,
        event_type: str,
        session_id: str | None = None,
        meta: dict | None = None,
        payload_pointer: str | None = None,
    ) -> int:
        """Append one event; return its monotonic ``seq``."""
        ts = datetime.now(UTC).isoformat()
        meta_json = json.dumps(meta) if meta is not None else None
        with self._append_lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    INSERT INTO notify_events
                        (ts, topic, event_type, session_id, meta, payload_pointer)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (ts, topic, event_type, session_id, meta_json, payload_pointer),
                )
        lastrowid = cur.lastrowid
        assert lastrowid is not None, "INSERT always produces a rowid"
        return int(lastrowid)

    def compact(self, retention_days: int | None = None) -> int:
        """Delete events older than the retention window; return count removed.

        ``retention_days`` defaults to ``COTHIS_NOTIFY_RETENTION_DAYS``;
        0, negative, or unparseable → no-op (compaction is opt-in).
        The Supervisor (#227) calls this periodically — default cadence
        daily. Snapshot-preservation (skipping rows pinned by an active
        snapshot) is a documented follow-up; no snapshot table exists
        yet, so the current implementation is pure age-based retention.
        """
        if retention_days is None:
            try:
                retention_days = int(os.environ.get("COTHIS_NOTIFY_RETENTION_DAYS", "0"))
            except ValueError:
                logger.warning(
                    "COTHIS_NOTIFY_RETENTION_DAYS=%r is not an integer; "
                    "skipping compaction.",
                    os.environ.get("COTHIS_NOTIFY_RETENTION_DAYS"),
                )
                return 0
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self._append_lock:
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM notify_events WHERE ts < ?",
                    (cutoff,),
                )
        deleted = cur.rowcount
        if deleted:
            logger.info(
                "notify_events compaction: deleted %d rows older than %d days",
                deleted,
                retention_days,
            )
        return int(deleted)

    def fetch_since(
        self,
        last_seq: int = 0,
        session_id: str | None = None,
    ) -> list[NotifyEvent]:
        """Return events with ``seq > last_seq``, optionally filtered by session.

        Ordered by ``seq`` ascending so consumers see a monotonic
        stream they can dedupe against their high-water mark.
        """
        if session_id is None:
            rows = self._conn.execute(
                f"SELECT {_FETCH_COLUMNS} FROM notify_events "
                "WHERE seq > ? ORDER BY seq",
                (last_seq,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {_FETCH_COLUMNS} FROM notify_events "
                "WHERE seq > ? AND session_id = ? ORDER BY seq",
                (last_seq, session_id),
            ).fetchall()
        return [_row_to_event(r) for r in rows]


def _row_to_event(row: tuple) -> NotifyEvent:
    return NotifyEvent(
        seq=int(row[0]),
        ts=row[1],
        topic=row[2],
        event_type=row[3],
        session_id=row[4],
        meta=json.loads(row[5]) if row[5] else None,
        payload_pointer=row[6],
    )
