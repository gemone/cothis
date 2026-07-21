# Cold/hot archival — startup wiring

The storage primitives from #84 (ATTACH-based archival transaction,
`ArchiveIndex`, `promote_session`) ship without callers. #86 wires
them into `Session`'s startup path: every `Session.new` / `Session.load`
runs the archival pass, and `Session.load` falls through to the cold
DB on a hot miss.

## 1. Startup trigger — implicit, not a CLI flag

`Session.new` and `Session.load` both call `run_archival_pass` after
`Storage` opens. There is no `--archive` flag, no scheduler, no
opt-in. The throttle (24h, persisted in `archive_state.last_run`)
bounds the work — at most one full pass per process per day.

### Considered

- **CLI flag (`cothis archive --all`).** Rejected: a CLI command can
  run when the user remembers; idle sessions pile up between runs.
  Wiring into `Session` startup guarantees the threshold is enforced
  without operator discipline.
- **Background scheduler thread.** Rejected: introduces a second
  writer to the hot DB (the consumer thread is the only writer today).
  The single-process model means the startup pass is sufficient.

### Consequences

- First `Session.new` of the day pays the archival cost (one `SELECT`
  on `sessions.updated_at`, then one `ATTACH` per archived session).
  Subsequent calls within 24h skip via the throttle — one `SELECT`
  against `archive_state`.
- Tests that drive `run_archival_pass` directly must clear
  `archive_state.last_run`; otherwise the throttle trips on the value
  stamped by `_seed_session`'s `Session.new`.

## 2. Cold read in place — no copy on load

When `Storage.load_session` returns `None`, `Session.load` consults
`ArchiveIndex`. If the session is archived, `_read_cold_session` opens
a *separate* sqlite3 connection to the cold DB and SELECTs the
session row + blocks. The hot DB is untouched. The session is flagged
`_cold=True`; the first write triggers `promote_session`.

### Considered

- **Promote on load (eager copy back).** Rejected: defeats the point
  of archival. A `cothis history <id>` peek would lift every cold
  session back into the hot DB. Read-in-place keeps the cold DB the
  source of truth until the user actually writes.
- **ATTACH the cold DB on the Storage connection and read through
  it.** Rejected: `Storage`'s connection is the consumer thread's
  writer. ATTACHing a cold DB on it would mix read and write scopes
  on one handle. A separate connection for the cold read keeps the
  scopes clean.

### Consequences

- Orphan-truncate (`storage.delete_blocks_from_msg_idx`) and
  ancestor-chain assembly are skipped for cold sessions — both write
  to / read from the hot DB, which has no rows yet. Orphan tails
  survive in memory until the first write promotes (then a subsequent
  `load` will truncate).
- A stale index entry (cold DB file deleted out of band) is dropped
  lazily on the first `Session.load` that hits it, then `KeyError`
  propagates.

## 3. Promote-on-first-write — atomic, `updated_at=now`

`_drain_one` checks `self._cold` before computing the lazy session
row. If cold, `promote_session` runs: ATTACH cold, INSERT rows into
hot with `updated_at=now`, VACUUM, drop index entry. After promote,
the lazy-row flag is already effectively True (hot has the row), so
`_drain_one` skips the lazy write and proceeds to insert the new
blocks via `write_atomic`.

### Considered

- **Lazy-row style: insert a fresh `SessionRow` on first write instead
  of promoting.** Rejected: the archived blocks would be lost (cold
  DB still has them, but hot's `blocks` table starts empty), and the
  index entry would dangle — `cothis history` would show the session
  as cold-only even though hot has a row.
- **Promote in a background thread.** Rejected: the consumer thread
  is the only writer; running promote off-thread would race with the
  block insert that prompted it. Inline promote keeps the
  cold→hot→new-block sequence atomic from the caller's view.

### Consequences

- The first write after a cold load pays the promote cost (ATTACH +
  cross-DB INSERT + VACUUM). Subsequent writes are hot-only.
- `updated_at` jumps from the archived value to `now`. This is the
  contract: a freshly-touched session isn't immediately re-archived
  by the next 90-day pass.

## 4. Flag plumbing — `cold: bool = False` on `__init__`

The constructor gains a `cold` param. `Session.new` and `Session.fork`
leave it at the default; `Session.load` passes `cold=cold_loaded`.
The flag is read once in `_drain_one` and cleared after promote.

### Considered

- **Re-derive cold state from `ArchiveIndex` on every write.**
  Rejected: the index is JSON-backed (`archive/index.json`); reading
  it on every `_drain_one` would add filesystem cost to the hot path.
  A session-scoped flag is one bool check.
- **Subclass `ColdSession(Session)`.** Rejected: `Session` already
  has `new` / `load` / `fork` factory classmethods; a subclass would
  need its own factories and the polymorphism would buy nothing over
  a flag.

### Consequences

- `_cold` is a private attribute, not part of the public API. Tests
  in `test_session_archival_wiring.py` reach for it because the
  public observable is "hot DB gained the session row" — but the
  flag itself is internal.
