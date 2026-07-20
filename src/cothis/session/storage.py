"""``cothis.session.storage`` — SQLite persistence for sessions.

The durable layer under :class:`cothis.session.Session`. One
``sqlite3.Connection`` per ``Storage`` (therefore per ``Session``), opened
with ``check_same_thread=False``; access is temporally partitioned (see
``Session`` docstring — load runs on the main thread before the consumer
starts; all writes are consumer-thread), so no mutex is needed.

This module is **CRUD-only**. It owns:

- the connection + ``PRAGMA journal_mode=WAL`` + ``busy_timeout=5000`` +
  ``user_version=1`` (R8);
- idempotent ``CREATE TABLE IF NOT EXISTS`` for ``sessions`` / ``blocks`` /
  ``archive_state`` plus the three indexes the issue fixes;
- one atomic writer (:meth:`Storage.write_atomic`) that takes pre-built row
  tuples — the single transaction boundary the consumer drains through;
- two readers (:meth:`Storage.load_session`, :meth:`Storage.load_blocks`)
  used once at ``Session.load``.

It knows **nothing** about: the write queue, the consumer thread, fcntl
locks, Anthropic block shape, or the lazy-session-row / title / .gitignore
policy. Those are ``Session``'s concern. Field mapping (Anthropic dict →
:class:`BlockRow`) lives in ``Session``; the inverse (``BlockRow`` →
Anthropic dict) lives in ``Session.load``'s rebuild.

Row types (:class:`SessionRow`, :class:`BlockRow`) are ``NamedTuple`` subclasses
so the 14-field block row is self-documenting at the call site instead of a
positional mystery. They are storage-shaped, not Anthropic-shaped:
``tool_input`` is a JSON string, ``content`` collapses both ``text`` and
``thinking`` (disambiguated by ``type``), etc.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# cothis: schema_version is a pure placeholder in #34 — column writes 1 and
# PRAGMA user_version is set to 1, but neither is read. Dispatch lands with
# the first real migration (#30 adds blocks.skill/blocks.state, or later).
# Bump this constant when a migration actually ships; the writer below pins
# it on every sessions row so future per-row dispatch has the data.
SCHEMA_VERSION = 1

_DDL = (
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
        schema_version INTEGER NOT NULL DEFAULT 1)
    """,
    """
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
        PRIMARY KEY (session_id, seq))
    """,
    "CREATE INDEX IF NOT EXISTS idx_blocks_msg  ON blocks(session_id, msg_idx, block_idx)",
    "CREATE INDEX IF NOT EXISTS idx_blocks_tool ON blocks(session_id, tool_name)",
    "CREATE INDEX IF NOT EXISTS idx_blocks_pair ON blocks(session_id, tool_use_id)",
    "CREATE TABLE IF NOT EXISTS archive_state(key TEXT PRIMARY KEY, value TEXT)",
)


class SessionRow(NamedTuple):
    """A row of the ``sessions`` table.

    ``parent_id`` / ``parent_seq`` are NULL in #34 — the fork tree lands in
    #35. Kept in the type (and the schema) so #35 doesn't need a migration.
    """

    id: str
    parent_id: str | None
    parent_seq: int | None
    cwd: str
    cli_version: str | None
    model: str | None
    title: str | None
    created_at: str
    updated_at: str
    schema_version: int = SCHEMA_VERSION


class BlockRow(NamedTuple):
    """A row of the ``blocks`` table — one Anthropic content block, persisted.

    Field mapping is fixed in :func:`cothis.session._block_to_row` (Anthropic
    dict → this tuple) and :func:`cothis.session._row_to_block` (this tuple
    → Anthropic dict). Storage handles these opaquely.
    """

    session_id: str
    seq: int
    msg_idx: int
    block_idx: int
    role: str
    type: str
    ts: str
    content: str | None = None
    signature: str | None = None
    tool_id: str | None = None
    tool_name: str | None = None
    tool_input: str | None = None
    tool_use_id: str | None = None
    tool_output: str | None = None
    image_source: str | None = None
    summary: str | None = None
    summarized_seq: str | None = None


def _restrict_to_owner(path: str) -> None:
    """Tighten file permissions to owner-only, cross-platform.

    Transcripts carry tool output (routinely secrets). SQLite creates db +
    WAL/SHM via open() under the umask (0o644 typical), and the sidecars do
    NOT inherit the main db's mode — so the caller chmod's each explicitly.

    POSIX: ``os.chmod(0o600)`` — the Unix permission model.
    Windows: ``os.chmod`` is near-useless (only toggles the read-only bit,
    doesn't express owner-only). Use ``icacls`` to remove inherited ACEs
    and grant only the current user full control. ``icacls`` ships with
    every Windows since Vista, so no dependency is added.

    Best-effort: filesystems without permission bits (FAT) or a missing
    ``icacls`` log a warning and continue — durability is still correct,
    only the access-control tightening is skipped. Sibling
    ``session/__init__.py`` follows the same convention on its best-effort
    paths (.gitignore write, atexit drain).
    """
    if os.name == "nt":
        user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
        if not user:
            logger.warning(
                "_restrict_to_owner: skipping (no USERNAME/USER env); "
                "db remains default-umask: %s", path,
            )
            return
        # /inheritance:r — remove inherited ACEs (drops Everyone/Users).
        # /grant:r      — replace (not merge) with current-user full control.
        cmd = ["icacls", path, "/inheritance:r", "/grant:r", f"{user}:F"]
        try:
            subprocess.run(
                cmd, check=True, capture_output=True, timeout=5,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "_restrict_to_owner: icacls failed on %s: %s", path, exc,
            )
    else:
        try:
            os.chmod(path, 0o600)
        except OSError as exc:
            logger.warning(
                "_restrict_to_owner: chmod 0o600 failed on %s: %s", path, exc,
            )


class Storage:
    """SQLite layer for one session's persistence.

    One connection, opened in WAL mode with a 5s busy timeout. The schema
    is created idempotently on construction (re-opening an existing db is
    a no-op for DDL). All writes flow through :meth:`write_atomic`, which
    is the transaction boundary the queue consumer drains through; the
    ``BEGIN deferred`` semantics validated in issue #34's research
    (fairness=1.0 across 4 concurrent writers; ``BEGIN IMMEDIATE`` starves)
    come from Python sqlite3's default deferred transaction mode.

    The connection is ``check_same_thread=False`` because the consumer
    thread writes while (during ``load`` only) the main thread may read.
    The two never overlap in time: ``load`` completes before the consumer
    starts; ``close`` joins the consumer before the connection is touched
    again. Temporal partitioning, not a mutex.
    """

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        # PRAGMA journal_mode is persistent (sqlite remembers), but setting
        # it on every open is cheap and self-documenting. busy_timeout is
        # per-connection. user_version is database-level; writing the same
        # value is a no-op.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self._conn.execute("PRAGMA foreign_keys=ON")
        for stmt in _DDL:
            self._conn.execute(stmt)
        self._conn.commit()
        # Tighten db + WAL/SHM sidecars to owner-only. SQLite creates all
        # three via open() under the umask; sidecars do NOT inherit the
        # main db's mode — restrict each explicitly. WAL holds
        # un-checkpointed tool output (same secrets); SHM is the index.
        for suffix in ("", "-wal", "-shm"):
            _restrict_to_owner(f"{db_path}{suffix}")

    def write_atomic(
        self,
        session_row: SessionRow | None,
        block_rows: list[BlockRow],
        updated_at: str,
    ) -> None:
        """One transaction: optional session-row insert + block inserts + updated_at bump.

        This is the atomic unit the consumer drains. ``session_row`` is set
        exactly once per session (the lazy row, on the first drain); after
        that callers pass ``None``. ``updated_at`` is bumped on every call
        so ``cothis history`` sort order and the archival threshold (#36)
        reflect last write, not creation.

        ``with self._conn:`` is Python sqlite3's deferred-transaction
        context manager: BEGIN before the first DML, COMMIT on a clean
        block exit, ROLLBACK on an exception. That's the atomicity the
        assistant-message invariant (Q2-A) depends on — N blocks share
        one txn, all-or-nothing.
        """
        # session_id for the updated_at UPDATE: prefer the blocks (covers
        # the no-session-row case after first drain), fall back to the
        # freshly-inserted session_row (covers a hypothetical blocks-empty
        # drain), empty string is a guaranteed no-op UPDATE.
        if block_rows:
            update_sid = block_rows[0].session_id
        elif session_row is not None:
            update_sid = session_row.id
        else:
            update_sid = ""

        with self._conn:
            if session_row is not None:
                self._conn.execute(
                    """
                    INSERT INTO sessions
                        (id, parent_id, parent_seq, cwd, cli_version, model,
                         title, created_at, updated_at, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(session_row),
                )
            if block_rows:
                self._conn.executemany(
                    """
                    INSERT INTO blocks
                        (session_id, seq, msg_idx, block_idx, role, type, ts,
                         content, signature, tool_id, tool_name, tool_input,
                         tool_use_id, tool_output, image_source, summary,
                         summarized_seq)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [tuple(b) for b in block_rows],
                )
            self._conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (updated_at, update_sid),
            )

    def load_session(self, session_id: str) -> SessionRow | None:
        """Return the ``sessions`` row, or ``None`` if no such session."""
        cur = self._conn.execute(
            """
            SELECT id, parent_id, parent_seq, cwd, cli_version, model,
                   title, created_at, updated_at, schema_version
            FROM sessions WHERE id=?
            """,
            (session_id,),
        )
        row = cur.fetchone()
        return SessionRow(*row) if row is not None else None

    def load_blocks(self, session_id: str) -> list[BlockRow]:
        """All blocks for ``session_id`` ordered for Anthropic-shape rebuild.

        Ordered by ``msg_idx, block_idx`` so grouping by ``msg_idx`` in
        ``Session.load`` reconstructs one Anthropic message per group with
        blocks in their original order.
        """
        cur = self._conn.execute(
            """
            SELECT session_id, seq, msg_idx, block_idx, role, type, ts,
                   content, signature, tool_id, tool_name, tool_input,
                   tool_use_id, tool_output, image_source, summary,
                   summarized_seq
            FROM blocks WHERE session_id=?
            ORDER BY msg_idx, block_idx
            """,
            (session_id,),
        )
        return [BlockRow(*row) for row in cur.fetchall()]

    def delete_blocks_from_msg_idx(
        self, session_id: str, cut_msg_idx: int
    ) -> None:
        """DELETE every block at ``msg_idx >= cut_msg_idx``.

        Used by ``Session.load`` after orphan-``tool_use`` drop-trailing:
        the in-memory truncate alone isn't enough, because the orphan rows
        persist in ``blocks`` and would cause the next reload to re-truncate
        at the same point — silently erasing any messages appended between
        reloads. This DELETE commits the truncate so the DB matches the
        in-memory view. ``msg_idx`` is monotonic per session, so the
        ``>=`` predicate is exactly "the orphan tail and everything after".
        """
        with self._conn:
            self._conn.execute(
                "DELETE FROM blocks WHERE session_id=? AND msg_idx >= ?",
                (session_id, cut_msg_idx),
            )

    def close(self) -> None:
        """Close the connection. Idempotent — safe to call twice."""
        if self._conn is not None:
            self._conn.close()
