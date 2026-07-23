# Durable notify bus — SQLite append-log per session

Issue #216 (Textual TUI + Durable Notify Bus) needs a communication
channel between the TUI, SessionWorker processes, and tools running
inside an Agent. The channel must be durable (TUI can detach, re-attach,
and read what it missed), multi-consumer (tool emits → many readers),
and not require a network service.

This ADR records the design of that channel for the MVP layer (#223):
an append-only ``notify_events`` table inside the existing per-session
SQLite DB, with a ``NotifyBus`` client that reuses the ``Storage``
connection.

## 1. SQLite append-log, not a message broker

The notify bus is a SQLite table, not Redis/Kafka/NATS. Events are
rows; consumers poll with ``fetch_since(last_seq)``.

### Considered

- **Redis pub/sub.** Rejected: cothis is local-first; introducing a
  mandatory external service breaks the "single binary, no deps" model.
  Cross-host brokers are explicitly out of scope per #216.
- **In-process ``asyncio.Queue`` per consumer.** Rejected: not durable.
  A TUI that detaches loses every event emitted while detached.
  SessionWorkers in a separate process can't subscribe at all.
- **Unix domain socket / WebSocket only.** Rejected: real-time delivery
  is necessary (SessionWorker streams deltas to TUI) but not sufficient
  — durability is the other half. The notify bus is the durable
  backplane; the WS is a separate transport layered on top (#225).

### Decision

SQLite. WAL mode + ``busy_timeout=5000`` (already configured by
``Storage``) give us multi-reader / single-writer concurrency, which
is all cothis needs: event emission is low-frequency (tens per turn)
and writes serialize in ~ms.

## 2. Per-session DB, not a separate ``notify.db``

``notify_events`` lives in the same SQLite file as ``sessions`` and
``blocks``. One ``NotifyBus`` instance per session.

### Considered

- **Separate ``~/.cache/cothis/notify.db`` shared across sessions.**
  Rejected: re-introduces cross-session write contention on a single
  file, and forces a second fcntl lock model on top of the per-session
  one. The PRD scopes the bus to per-session events; a global bus is a
  later slice if a concrete need surfaces.
- **Standalone ``notify/`` module with its own connection class.**
  Rejected for MVP: would duplicate WAL/busy_timeout/foreign-key
  setup. ``NotifyBus`` takes the live ``Storage._conn``; the
  connection's PRAGMAs already match what the bus needs.

### Decision

Per-session DB. ``NotifyBus.__init__(conn)`` runs ``CREATE TABLE IF
NOT EXISTS`` idempotently — old session DBs get the table on first
open with no migration step.

## 3. Payload-pointer pattern

Events carry metadata (``topic``, ``event_type``, ``session_id``,
``meta``) and a ``payload_pointer`` string. The pointer is a
session-local reference (e.g. ``session:<sid>:tool:<call_id>``); the
full payload lives in the existing ``blocks`` table as a tool_result.

### Considered

- **Inline JSON payload on every event.** Rejected: a ``fs.read`` on a
  2 MB file produces a 2 MB tool_result; inlining that in the notify
  event duplicates bytes and bloats the append-log. Consumers that
  don't need the payload pay the cost anyway.
- **Separate ``notify_payloads`` table with ``event_id`` FK.**
  Rejected: ``blocks`` already stores tool results; a second payload
  table would duplicate that storage. The pointer lets consumers fetch
  from ``blocks`` when they actually need the bytes.

### Decision

Pointers only. ``meta`` carries small structured fields (tool name,
duration, call_id); anything large is referenced.

## 4. Single ``seq`` column (INTEGER PRIMARY KEY AUTOINCREMENT)

The issue sketch proposed two columns — ``id INTEGER PRIMARY KEY
AUTOINCREMENT`` and ``seq INTEGER UNIQUE NOT NULL``. This ADR collapses
them into one.

### Considered

- **Two columns (issue sketch).** Rejected: both would be monotonic and
  unique; keeping them in sync requires either a trigger or
  ``UPDATE seq = id`` on every insert, with no functional benefit. The
  sketch treated ``seq`` and ``id`` as interchangeable in its own
  description ("``seq`` / ``event_id``").
- **Application-assigned ``seq`` via ``MAX(seq) + 1``.** Rejected:
  introduces a race window even inside a transaction; relies on
  serialization that AUTOINCREMENT already provides for free.

### Decision

``seq INTEGER PRIMARY KEY AUTOINCREMENT``. SQLite guarantees
monotonic, never-reused seqs. Consumers dedupe by keeping their
high-water mark.

## 5. ``threading.Lock`` around append

``NotifyBus.append`` takes a process-local ``threading.Lock`` before
the ``with self._conn:`` transaction. SQLite's deferred-transaction
mode raises ``OperationalError: cannot start a transaction within a
transaction`` if two threads enter ``with self._conn:`` concurrently
on the same connection.

### Considered

- **``isolation_level=None`` (autocommit) + ``BEGIN IMMEDIATE``
  manually.** Rejected: same problem — the BEGIN/COMMIT pair still
  needs a Python-side lock to be atomic across threads, so we'd
  reimplement the mutex without removing it.
- **One connection per thread.** Rejected: violates ``Storage``'s
  single-connection invariant (the WAL reader/writer model assumes one
  writer per db per process).

### Decision

``threading.Lock`` for same-process thread serialization; SQLite's
own file locking handles cross-process serialization via the existing
fcntl session lock (only one process holds a given session at a time,
and the notify bus is a secondary client on the same db).

## 6. Feature flag ``COTHIS_NOTIFY_BUS``

The bus ships importable and tested but is not wired into
``Agent._execute_tool`` in this slice. The wiring (#224) is gated by
``COTHIS_NOTIFY_BUS``; when unset, ``Agent._bus is None`` and all
``if self._bus:`` branches are skipped — tool behavior is
byte-for-byte identical to today.

This ADR covers the bus infrastructure only. The rollout path
(dev → staging → global) is #224's concern.
