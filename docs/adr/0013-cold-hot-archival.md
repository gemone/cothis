# Cold/hot archival

The session store (#29 / #34) keeps every conversation in one hot
SQLite DB. Without archival the file grows unbounded; on a long-lived
workspace `cothis history` listing slows and `VACUUM` reclaim after
`delete` becomes expensive. #36 introduces a two-tier layout: hot DB
for active sessions, monthly cold DBs (`YYYY-MM.db`) for sessions idle
past a threshold (default 90 days).

This ADR records the four interdependent decisions behind the layout;
ADR-0011 and ADR-0012 record the wiring-level sub-decisions (startup
trigger, cross-DB delete). #36's sibling sub-issues:

- #84 — storage primitives (`archive_session`, `run_archival_pass`,
  `promote_session`, `ArchiveIndex`).
- #85 — CLI (`cothis archive`, `restore`, `compress`).
- #86 — startup trigger + cold read in place + promote-on-first-write
  (ADR-0011).
- #87 — `cothis delete` extended across hot + cold (ADR-0012).

## 1. ATTACH-based cross-DB transactions

Archival and promote-back are each one atomic SQLite transaction
across two DBs: `ATTACH 'archive/YYYY-MM.db' AS arch; BEGIN; INSERT
INTO arch.{sessions,blocks} SELECT … FROM main.{sessions,blocks};
DELETE FROM main.{sessions,blocks}; COMMIT; DETACH`. The hot DB's
`FileLock` serializes same-process work; the parameter-bound ATTACH
rejects tampered index entries that point at `../../etc/passwd`.

### Considered

- **Separate archival process** (cron, systemd timer). Rejected: a
  separate process can run when the user remembers, but cothis is a
  single-user CLI — wiring the pass into `Session.new`/`load` startup
  (ADR-0011 §1) gives the 24h throttle for free without operator
  discipline.
- **One DB per session** (filesystem-as-database). Rejected: SQLite
  indexes across sessions (`cothis history`) need one handle; 10k
  open-DB calls per listing would dominate. The current hot DB serves
  10k sessions in ~9ms.
- **JSON-only cold store** (no SQLite for archived sessions).
  Rejected: the cold path needs the same SELECT-by-id /
  has-children / block ordering contracts as hot. Reimplementing them
  on JSON is real code with new edge cases. ATTACH gives cold sessions
  the same SQL semantics with one extra `ATTACH` per transaction.
- **Automatic copy-on-resume** (promote eagerly when the user runs
  `cothis chat --resume <id>`). Rejected: defeats the point of
  archival. A `cothis history <id>` peek would lift every cold
  session back into hot. Read-in-place + promote-on-first-write
  (ADR-0011 §2, §3) keeps cold as the source of truth until the user
  actually writes.

### Consequences

- The monthly cold DB schema mirrors hot `sessions` + `blocks`
  (minus `archive_state`) — see `_ensure_cold_schema` in
  `cothis.session.archive`. No schema migration is needed when hot
  adds a column: cold's `CREATE TABLE IF NOT EXISTS` is idempotent
  and the archival `INSERT … SELECT *` requires parity, which a
  missing cold column would surface as a SQL error on first archival
  after the migration.
- ATTACH holds the hot DB's `BEGIN IMMEDIATE` lock for the duration
  of the cross-DB transaction. On a 10k-block session archival takes
  ~50ms; the consumer thread is paused for that window.

## 2. `archive/index.json` for cold lookup

A small JSON file maps `session_id → {archive_db, archived_at}`.
Without it, cold lookup would scan every monthly DB per session.
The index is rewritten in full on each save (hundreds of entries at
most — one per archived session).

### Considered

- **Per-DB manifests** (`YYYY-MM.json` next to each cold DB).
  Rejected: lookup would still walk every month's JSON; one index is
  one read.
- **SQLite index DB** (`archive/index.db`). Rejected: a JSON file
  with `_validate_archive_db` basename enforcement is simpler, and
  the index is small enough that JSON parse + dict lookup beats a
  SQLite open.

### Consequences

- The index is the source of truth for "is this session cold?".
  `Session.load` consults it on hot miss (ADR-0011 §2);
  `Session.delete` consults it for the cold path (ADR-0012 §1).
- Drift is self-healing: `delete_cold_session` and `_read_cold_session`
  both gate on `is_file()` and drop the index entry if the cold DB
  is gone. The next `run_archival_pass` won't re-archive a session
  the index still tracks.

## 3. Promote-back-on-first-write

Cold sessions stay cold until the user writes. The first
`_drain_one` after a cold load calls `promote_session`, which
ATTACHes the cold DB, INSERTs rows into hot with `updated_at=now`,
VACUUMs the cold DB, and drops the index entry. Subsequent writes
are hot-only.

### Considered

- **Lazy row on first write instead of promote** (insert a fresh
  `SessionRow`, leave archived blocks in cold). Rejected: the
  archived blocks would be invisible to the next reload; `cothis
  history` would show the session as cold-only while hot has a row.
- **Promote in a background thread.** Rejected: the consumer thread
  is the only writer; an off-thread promote would race with the
  block insert that prompted it.

### Consequences

- `updated_at` jumps from the archived value to `now`. This is the
  contract — a freshly-touched session isn't immediately re-archived
  by the next 90-day pass (ADR-0011 §3 documents the crash-safety
  of the in-memory `_cold` flag).
- The first write after a cold load pays the promote cost (ATTACH +
  cross-DB INSERT + VACUUM). Subsequent writes are hot-only.

## 4. Operational consequences

- **24h throttle.** `run_archival_pass` records `last_run` in
  `archive_state`; subsequent calls within 24h skip. The throttle
  bounds the work — at most one full pass per process per day.
  Tests that drive `run_archival_pass` directly clear the state via
  `_clear_archive_state`.
- **VACUUM reclaims space.** Both `archive_session` and
  `promote_session` end with `VACUUM` on the affected DBs.
  `run_archival_pass` batches the hot-DB VACUUM across all sessions
  moved in one pass (one VACUUM at the end, not per session).
- **No compression of block contents.** Cold DBs use the same schema
  as hot; `tool_output` and `content` columns are stored verbatim.
  The `cothis archive compress <file>` CLI (#85) produces a gzip
  sidecar for cold transport/backup, but online access is
  uncompressed SQLite. Compression would block random access and
  save at most ~3x on text-heavy transcripts — not worth the
  complexity at this scale.
