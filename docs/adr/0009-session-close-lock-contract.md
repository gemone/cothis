# Session close-path lock contract

Date: 2026-07-21

Status: Accepted. Fixes the close-path race described in issue #57.
Builds on ADR-0008's poison-row semantics: the consumer's
``ProgrammingError`` after storage close is caught by ``_drain_one``'s
retry queue and dropped, preserving the kill-9 loss ceiling.

## Context

``Session.close()`` joins the daemon consumer thread with a 5 s timeout.
Before this ADR, a consumer still alive after the timeout caused close()
to (a) leave ``Storage`` open, (b) release the cross-process file lock,
(c) return. The justification was intra-process safety: closing the
SQLite connection from the main thread while the daemon consumer is
mid-``write_atomic`` would make the consumer's next write raise
``ProgrammingError``.

The inter-process consequence was missed. ``SessionLockedError``'s
advertised contract is "the other process may be mid-write and
retrying would race it" — exactly the condition the close-path created.
A second process reaching ``Session.load(session_id)`` would succeed
(the lock is released), open the same ``db_path``, and start its own
consumer. Two consumers then interleave writes to the same database
file. SQLite WAL prevents page-level corruption, but the *semantic*
ordering of block rows is lost — process B's first drain can land
between process A's residual writes, producing a conversation history
neither process intended.

## Decision

**Close storage unconditionally after the join timeouts; release the
lock only after storage close.**

### 1. Two-phase join

```python
self._consumer.join(timeout=_CLOSE_JOIN_TIMEOUT)  # 5 s — generous
if self._consumer.is_alive():
    self._consumer.join(timeout=_CLOSE_GRACE_PERIOD)  # 1 s — final beat
```

The first timeout drains dozens of backlogged writes at ~50 ms each. The
grace period lets a nearly-finished consumer complete its current
``write_atomic``. Both timeouts are module-level constants so tests can
squash them.

### 2. Always close storage

```python
try:
    self._storage.close()
finally:
    self._release_lock()
```

No longer conditional on ``consumer_alive``. If the consumer is still
draining when storage closes, its next ``write_atomic`` raises
``ProgrammingError``. ``_drain_one``'s retry queue catches it (ADR-0008);
the in-flight batch exhausts its 4 attempts in 2.6 s and is dropped per
poison-row semantics. Loss ceiling unchanged.

### 3. Lock released only after storage close

``_release_lock()`` runs in the ``finally`` block, so it always fires —
but only after ``storage.close()`` returns. This restores the
``SessionLockedError`` contract: a second acquirer reaching the same
``session_id`` is refused because the lock is still held while storage
is being closed.

## Considered

- **Option B — cooperative close flag.** Add a ``_closing`` flag the
  consumer checks before each ``write_atomic``. ``close()`` sets the
  flag, joins, then closes storage. Eliminates the mid-write window
  without a forced close. Rejected for this fix: requires the consumer
  to poll the flag between queue gets (adds latency to the common case)
  and complicates the contract ("the consumer MAY observe _closing and
  stop early"). Option A's ProgrammingError path is already covered by
  ADR-0008's retry semantics; no new state needed.
- **Option C — consumer-owned connection.** The consumer thread opens,
  owns, and closes the SQLite connection. ``close()`` signals stop +
  joins. Eliminates cross-thread connection access entirely, removing
  the "temporal partitioning" claim from the ``Session`` docstring.
  Rejected: largest structural change in this area since #42; the
  intra-process temporal partitioning (load before consumer starts;
  close joins before main thread touches the connection) is documented
  and tested, and the close-path race is now fixed without restructuring.
  A future PR could revisit if other cross-thread bugs surface.
- **Block forever on consumer join.** Rejected — a genuinely stuck
  consumer would hang the CLI indefinitely.
- **Force-kill the consumer thread.** Python offers no safe thread
  cancellation primitive.

## Consequences

- The lock contract is restored: a second acquirer reaching the same
  ``session_id`` while ``close()`` is in flight is refused.
- A stuck consumer's residual writes fail with ``ProgrammingError``
  and are dropped per ADR-0008 poison-row semantics; ``_dropped_*``
  diagnostics are surfaced via CRITICAL log.
- ``close()`` latency ceiling is now ``_CLOSE_JOIN_TIMEOUT +
  _CLOSE_GRACE_PERIOD`` (6 s) instead of 5 s. Acceptable: only paid
  once at session end, and only when the consumer is stuck.
- The intra-process "leave storage open for daemon to finish" claim
  in the old ``close()`` docstring is dropped — the consumer's
  ProgrammingError is caught and dropped, not allowed to finish.
