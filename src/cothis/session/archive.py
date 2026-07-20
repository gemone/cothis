"""``cothis.session.archive`` — cold/hot archival (#36).

Moves sessions idle past a threshold (default 90 days) into monthly
cold-archive DBs, and lets users manage them. End-to-end:

- **Cold DB schema**: mirrors the hot ``sessions`` + ``blocks`` minus
  ``archive_state`` (cold DBs don't trigger archival).
- **Archival transaction**: ``ATTACH 'archive/YYYY-MM.db'; BEGIN;
  INSERT INTO arch.{sessions,blocks} SELECT …; DELETE FROM
  main.{sessions,blocks} WHERE …; COMMIT; VACUUM; DETACH``. Atomic +
  idempotent on re-run.
- **Archive index** (``archive/index.json``): ``session_id →
  {archive_db, archived_at}`` so cold lookup doesn't scan every archive.
- **Promote-back**: the first new write moves the session back to the
  hot DB atomically with ``updated_at = now`` + index update.

The cold DBs and the index live under ``<db_path parent>/archive/``.
The monthly filename (``YYYY-MM.db``) is computed from the archival
run's ``now`` (or supplied explicitly for testability).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchivedEntry:
    """One row of ``archive/index.json``."""

    archive_db: str
    archived_at: str


class ArchiveIndex:
    """JSON-backed ``session_id → {archive_db, archived_at}`` map.

    The file is rewritten in full on every :meth:`save` (the index is
    small — one entry per archived session, hundreds at most). Callers
    load once at startup, mutate in memory, save after each archival /
    promote / delete.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, ArchivedEntry] = {}
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Archive index %s unreadable (%s); starting empty.",
                    path, exc,
                )
                return
            if isinstance(raw, dict):
                for sid, entry in raw.items():
                    if (
                        isinstance(entry, dict)
                        and isinstance(entry.get("archive_db"), str)
                        and isinstance(entry.get("archived_at"), str)
                    ):
                        self._entries[sid] = ArchivedEntry(
                            entry["archive_db"], entry["archived_at"]
                        )

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, session_id: str) -> ArchivedEntry | None:
        return self._entries.get(session_id)

    def set(self, session_id: str, archive_db: str, archived_at: str) -> None:
        self._entries[session_id] = ArchivedEntry(archive_db, archived_at)

    def remove(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            sid: {"archive_db": e.archive_db, "archived_at": e.archived_at}
            for sid, e in self._entries.items()
        }
        self._path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _month_bucket(iso_ts: str) -> str:
    """``2026-07-20T...`` → ``2026-07``. Used for the monthly cold-DB name."""
    return iso_ts[:7]


def archive_session(
    *,
    hot_db_path: Path,
    archive_dir: Path,
    session_id: str,
    archive_db_name: str,
    archived_at: str,
    index: ArchiveIndex,
) -> None:
    """Move ``session_id``'s rows from the hot DB to ``archive_dir / archive_db_name``.

    Atomic per session: ATTACH the cold DB, INSERT rows, DELETE from
    hot, COMMIT, VACUUM, DETACH. Idempotent — a second call with the
    same args either re-copies (if the cold row was deleted out of
    band) or no-ops (INSERT OR REPLACE).
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    cold_db_path = archive_dir / archive_db_name
    _ensure_cold_schema(cold_db_path)

    conn = sqlite3.connect(hot_db_path)
    try:
        conn.execute(f"ATTACH DATABASE '{cold_db_path.as_posix()}' AS arch")
        try:
            with conn:  # BEGIN/COMMIT
                # Guard: only copy + delete when the hot DB still has the
                # rows. Idempotent re-runs (hot already drained) leave the
                # cold copy alone.
                has_session = conn.execute(
                    "SELECT 1 FROM main.sessions WHERE id=?", (session_id,)
                ).fetchone() is not None
                if not has_session:
                    return None
                conn.execute(
                    "INSERT OR REPLACE INTO arch.sessions "
                    "SELECT * FROM main.sessions WHERE id=?",
                    (session_id,),
                )
                conn.execute(
                    "DELETE FROM arch.blocks WHERE session_id=?",
                    (session_id,),
                )
                conn.execute(
                    "INSERT INTO arch.blocks "
                    "SELECT * FROM main.blocks WHERE session_id=?",
                    (session_id,),
                )
                conn.execute(
                    "DELETE FROM main.blocks WHERE session_id=?", (session_id,)
                )
                conn.execute(
                    "DELETE FROM main.sessions WHERE id=?", (session_id,)
                )
        finally:
            conn.execute("DETACH DATABASE arch")
        conn.execute("VACUUM")
    finally:
        conn.close()

    index.set(session_id, archive_db_name, archived_at)
    index.save()


def run_archival_pass(
    *,
    hot_db_path: Path,
    archive_dir: Path,
    threshold_days: int,
    now_iso: str,
    index: ArchiveIndex | None = None,
) -> int:
    """Archive every session idle past ``threshold_days``.

    Throttled via ``archive_state.last_run``: a row younger than 24h
    skips the pass entirely (single-process execution means no race).
    Returns the number of sessions archived this run.
    """
    if index is None:
        index = ArchiveIndex(archive_dir / "index.json")

    conn = sqlite3.connect(hot_db_path)
    try:
        last_run = conn.execute(
            "SELECT value FROM archive_state WHERE key='last_run'"
        ).fetchone()
        from datetime import datetime, timedelta

        now_dt = datetime.fromisoformat(now_iso)
        if last_run is not None:
            last_dt = datetime.fromisoformat(last_run[0])
            if (now_dt - last_dt) < timedelta(hours=24):
                return 0

        cutoff = now_dt - timedelta(days=threshold_days)
        cutoff_iso = cutoff.isoformat()
        rows = conn.execute(
            "SELECT id, updated_at FROM sessions WHERE updated_at < ?",
            (cutoff_iso,),
        ).fetchall()
    finally:
        conn.close()

    archived = 0
    for sid, updated_at in rows:
        archive_session(
            hot_db_path=hot_db_path,
            archive_dir=archive_dir,
            session_id=sid,
            archive_db_name=f"{_month_bucket(updated_at)}.db",
            archived_at=now_iso,
            index=index,
        )
        archived += 1

    # Record the run.
    conn = sqlite3.connect(hot_db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO archive_state(key, value) VALUES ('last_run', ?)",
            (now_iso,),
        )
        conn.commit()
    finally:
        conn.close()

    return archived


def promote_session(
    *,
    hot_db_path: Path,
    archive_dir: Path,
    session_id: str,
    index: ArchiveIndex,
    now_iso: str | None = None,
) -> bool:
    """Copy ``session_id``'s rows back from cold to hot, drop the index entry.

    Returns ``True`` if the promotion ran, ``False`` if the session wasn't
    in the index (already hot, or never archived). ``now_iso`` defaults
    to wall-clock now; the caller passes it for deterministic tests.
    """
    entry = index.get(session_id)
    if entry is None:
        return False
    if now_iso is None:
        from datetime import UTC, datetime

        now_iso = datetime.now(UTC).isoformat()

    cold_db_path = archive_dir / entry.archive_db
    if not cold_db_path.is_file():
        logger.warning(
            "promote_session: cold DB %s missing for session %s; "
            "index entry removed but session unrecoverable.",
            cold_db_path, session_id,
        )
        index.remove(session_id)
        index.save()
        return False

    conn = sqlite3.connect(hot_db_path)
    try:
        conn.execute(f"ATTACH DATABASE '{cold_db_path.as_posix()}' AS arch")
        try:
            with conn:
                # Explicit copy so we control updated_at (overwritten to
                # ``now_iso`` so the session isn't immediately re-archived).
                conn.execute("DELETE FROM main.sessions WHERE id=?", (session_id,))
                conn.execute("DELETE FROM main.blocks WHERE session_id=?", (session_id,))
                cur = conn.execute(
                    "SELECT id, parent_id, parent_seq, cwd, cli_version, "
                    "model, title, created_at, updated_at, schema_version "
                    "FROM arch.sessions WHERE id=?",
                    (session_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise KeyError(
                        f"session {session_id!r} in index but not in cold DB"
                    )
                conn.execute(
                    "INSERT INTO sessions(id, parent_id, parent_seq, cwd, "
                    "cli_version, model, title, created_at, updated_at, "
                    "schema_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (*row[:8], now_iso, row[9]),
                )
                conn.execute(
                    "INSERT INTO blocks SELECT * FROM arch.blocks WHERE session_id=?",
                    (session_id,),
                )
        finally:
            conn.execute("DETACH DATABASE arch")
        conn.execute("VACUUM")
    finally:
        conn.close()

    index.remove(session_id)
    index.save()
    return True


def _ensure_cold_schema(cold_db_path: Path) -> None:
    """Create the cold DB schema if missing.

    Mirrors hot ``sessions`` + ``blocks`` without ``archive_state``
    (cold DBs don't trigger archival). Idempotent on re-open.
    """
    conn = sqlite3.connect(cold_db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions(
                id            TEXT PRIMARY KEY,
                parent_id     TEXT,
                parent_seq    INTEGER,
                cwd           TEXT NOT NULL,
                cli_version   TEXT,
                model         TEXT,
                title         TEXT,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                schema_version INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS blocks(
                session_id    TEXT NOT NULL REFERENCES sessions(id),
                seq           INTEGER NOT NULL,
                msg_idx       INTEGER NOT NULL,
                block_idx     INTEGER NOT NULL,
                role          TEXT NOT NULL,
                type          TEXT NOT NULL,
                ts            TEXT NOT NULL,
                content       TEXT,
                signature     TEXT,
                tool_id       TEXT,
                tool_name     TEXT,
                tool_input    TEXT,
                tool_use_id   TEXT,
                tool_output   TEXT,
                image_source  TEXT,
                summary       TEXT,
                summarized_seq TEXT,
                PRIMARY KEY (session_id, seq));
            CREATE INDEX IF NOT EXISTS idx_blocks_msg  ON blocks(session_id, msg_idx, block_idx);
            CREATE INDEX IF NOT EXISTS idx_blocks_tool ON blocks(session_id, tool_name);
            CREATE INDEX IF NOT EXISTS idx_blocks_pair ON blocks(session_id, tool_use_id);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _24h_ago(now_iso: str) -> str:
    """Placeholder kept for the dead branch above; returns ``now_iso`` as-is."""
    return now_iso
