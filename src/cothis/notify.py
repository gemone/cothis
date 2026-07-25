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

    def compact(
        self,
        retention_days: int | None = None,
        preserve_seqs: set[int] | None = None,
    ) -> int:
        """Delete events older than the retention window; return count removed.

        ``retention_days`` defaults to ``COTHIS_NOTIFY_RETENTION_DAYS``;
        0, negative, or unparseable → no-op (compaction is opt-in).
        The Supervisor (#227) calls this periodically — default cadence
        daily.

        ``preserve_seqs`` (optional, #246) holds event ``seq`` values that
        must survive compaction regardless of age — i.e. events an active
        consumer (a Supervisor checkpoint, a live TUI cursor) still
        references. The criterion from #236 ("events referenced by an
        active snapshot are preserved") is satisfied this way: the
        caller pins the lowest ``seq`` its consumers haven't read past
        (see :meth:`preserve_since`) and compaction skips those rows.

        We deliberately do NOT model a separate ``notify_snapshots``
        table: the real need is a per-consumer high-water mark, not
        arbitrary event-pinning, and a set of ``seq`` values is the
        minimal structure that captures it. A new table + lock/expiry
        lifecycle would be abstraction the current callers don't ask
        for (AGENTS.md § Project rules: no abstraction that wasn't
        requested).
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
        # Build the DELETE with a seq-exclusion guard when callers pin
        # rows. Using a placeholder per value (not string-formatting seqs
        # into SQL) keeps the parametrised-statement contract intact —
        # seqs are ints from AUTOINCREMENT, but defence-in-depth anyway.
        if preserve_seqs:
            pinned = {int(s) for s in preserve_seqs}
            placeholders = ",".join("?" for _ in pinned)
            sql = (
                f"DELETE FROM notify_events "
                f"WHERE ts < ? AND seq NOT IN ({placeholders})"
            )
            params: tuple = (cutoff, *pinned)
        else:
            sql = "DELETE FROM notify_events WHERE ts < ?"
            params = (cutoff,)
        with self._append_lock:
            with self._conn:
                cur = self._conn.execute(sql, params)
        deleted = cur.rowcount
        if deleted:
            logger.info(
                "notify_events compaction: deleted %d rows older than %d days",
                deleted,
                retention_days,
            )
        return int(deleted)

    def preserve_since(self, *cursors: int) -> set[int]:
        """Return the ``seq`` set to pass as ``preserve_seqs`` for cursor-based pinning.

        Given one or more consumer high-water marks (the largest ``seq``
        each consumer has already read), return every ``seq`` at or below
        the *minimum* cursor that still exists in the log. Compaction with
        this set leaves intact the contiguous prefix any lagging consumer
        might still fetch — the log-compaction cursor rule.

        A cursor of 0 (consumer hasn't read anything) pins nothing,
        because such a consumer can't reference any specific event yet;
        passing no cursors returns an empty set (no pinning).

        Rationale: the Supervisor restarts workers and must not delete
        ``session_lifecycle`` rows a re-attaching TUI hasn't seen. The
        TUI tracks ``last_seq``; the Supervisor forwards the live
        cursors here at compact time. This is the #236 snapshot
        criterion, expressed as a cursor instead of a snapshot table.
        """
        active = [c for c in cursors if c > 0]
        if not active:
            return set()
        floor = min(active)
        rows = self._conn.execute(
            "SELECT seq FROM notify_events WHERE seq <= ?", (floor,)
        ).fetchall()
        return {int(r[0]) for r in rows}

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
