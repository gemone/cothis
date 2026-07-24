# Supervisor — worker lifecycle + always_backoff restart

Issue #216 (Textual TUI + Durable Notify Bus) puts each Agent in a
child process (the SessionWorker, #225 / PR #247) and runs a
Supervisor process that spawns + monitors those workers. This ADR
records the design of the Supervisor component for the MVP slice
(#227):

- The Supervisor owns worker spawn + restart logic.
- Restarts use ``always_backoff`` (exponential, capped at 300s).
- ``session_lifecycle`` events go on a **separate Supervisor DB** —
  not the per-session DB the worker owns.
- The TUI polls the Supervisor's bus for status badges.

## 1. Separate Supervisor DB, not the per-session DB

Each worker holds an exclusive ``FileLock(timeout=0)`` on its
session. The Supervisor cannot write to a per-session DB without
either acquiring that lock (rejected — would block the worker) or
sharing the connection (rejected — sqlite3 + ``check_same_thread``
model doesn't extend across processes cleanly).

### Considered

- **Write lifecycle events into the worker's per-session DB.**
  Rejected: the worker's ``FileLock`` blocks the Supervisor from
  acquiring it, and opening a second connection to the same DB would
  race the worker's queue consumer.
- **HTTP endpoint on the Supervisor for status.** Rejected: adds an
  HTTP server to a process whose only job is lifecycle; the notify
  bus pattern from #223 is already the right shape.
- **In-memory status dict (no persistence).** Rejected: a Supervisor
  crash would lose the entire restart history; the TUI re-attaching
  after a Supervisor restart needs the prior state.

### Decision

``~/.cothis/supervisor.db`` (configurable via the constructor). One
``NotifyBus`` on that DB; events have ``topic="session_lifecycle"``
and ``event_type`` in ``{"spawned", "crashed", "restarted"}``.

## 2. ``always_backoff`` restart policy

Restart delay is ``min(2 ** restart_count, 300s)``. Past a rolling
window of N restarts (default 5 in 600s), the session is marked
``errored`` and no further restarts are attempted until a UI action
clears the state.

### Considered

- **Fixed delay (always 30s).** Rejected: under a tight crash loop,
  burns resources at 1 restart / 30s indefinitely. Exponential
  dampens the loop without manual tuning.
- **Linear backoff.** Rejected: too slow at small counts (1, 2, 3,
  4, …) — exponential (1, 2, 4, 8, …) catches a transient failure
  quickly + slows down for persistent ones.
- **No cap.** Rejected: ``2 ** 12`` = 4096s = 68 minutes — past the
  patience of a user who's watching the badge. The 300s cap bounds
  the worst case to one attempt per 5 minutes.

### Decision

Exponential, floor 1s, ceiling 300s. Rolling-window counter (default
600s × 5 restarts) → ``errored`` state once the threshold trips.

## 3. Rolling-window threshold

The counter keeps a list of recent restart timestamps; queries
filter to entries inside the window. Old timestamps don't actively
prune (a periodic compaction would do that; the cost of keeping
them is tiny since the per-worker list is bounded by the restart
rate × window, which is small).

### Considered

- **Sliding-window with constant-time eviction.** Rejected: the
  restart rate is human-scale (one per minute at most); a list scan
  per query is sub-microsecond.
- **Counter reset on every successful turn.** Rejected: a single
  successful turn doesn't prove the worker is healthy — the bug may
  be intermittent. Time-based expiry is the right signal.

### Decision

List of timestamps + filter at query time. ``is_over_threshold``
returns True once count ≥ threshold within the window.

## 4. Subprocess-per-session, not thread-per-session

Each session runs in its own OS process. The Supervisor spawns via
``subprocess.Popen`` (real subprocess path deferred — this MVP ships
the pure lifecycle logic + the spawn / WS-handshake contract; the
real subprocess integration lands when the worker CLI entrypoint
from #225 is finalised).

### Considered

- **One process per session, threads inside for parallel turns.**
  Rejected: the LLM stream is inherently sequential per turn; a
  thread per turn adds locking complexity without throughput gain.
- **Single process with multiple Agent instances.** Rejected: a
  crash in one Agent's tool (e.g. MCP SDK segfault) would kill all
  sessions; process isolation contains the blast radius.

### Decision

Subprocess-per-session. The Supervisor owns the spawn loop; each
worker is an isolated process whose crash the Supervisor detects via
WS disconnect or process-exit signal.

## 5. WS heartbeat, not process-exit polling

The Supervisor pings each worker over its WS connection at a fixed
cadence. A missed pong past a deadline triggers the restart path.

### Considered

- **Poll ``subprocess.Poll`` periodically.** Rejected: a hung worker
  (process alive but not making progress) wouldn't be caught.
  ``ping``/``pong`` proves the event loop is responsive.
- **WS keepalive only (no app-level ping).** Rejected: library
  keepalive doesn't surface to the Supervisor's restart logic;
  app-level ``ping`` is the integration point the worker's
  ``_dispatch`` already handles.

### Decision

App-level ``ping`` from the Supervisor; missed deadline → restart.
The WS protocol (#225) already implements ``ping``/``pong``.

## 6. Out of scope for #227

- **Real subprocess spawn + WS handshake integration test.** The
  unit tests cover the pure logic (backoff, counter, lifecycle
  record, status snapshot). The integration test with a real
  throwaway worker is tracked as follow-up #250, blocked on the
  worker CLI entrypoint (itself deferred from #225).
- **TUI consumption of the status stream.** The Supervisor exposes
  ``status()`` + ``lifecycle_since(seq)``; the Textual app that
  renders badges is a later slice (#228 TUI core).
- **Worker CLI entrypoint + ``--session`` flag.** Tracked under
  #250; the Supervisor's spawn command will call
  ``python -m cothis.worker --session <id>`` once that lands.
- **Multi-host deployment.** Loopback-only is the project stance;
  the Supervisor runs on the same host as the workers and the TUI.
