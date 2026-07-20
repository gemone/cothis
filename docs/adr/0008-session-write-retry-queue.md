# Retry queue for transient SQLite write failures in `Session._drain_one`

Date: 2026-07-21

Status: Accepted. Implements Option A from the grilling pass in issue
#43. Replaces the swallow-and-log behaviour that landed with the session
store in #42 (ADR-0006 PR1 follow-on decisions, item C — deferred at
the time to its own issue + grilling pass).

## Context

`Session._drain_one` is the single drain point for the write queue:
every persisted block row goes through one `Storage.write_atomic` call.
The #42 implementation caught every exception from `write_atomic`,
logged at ERROR level, and returned. The dequeued rows were gone — the
queue is FIFO and does not re-enqueue.

A single transient failure (momentary I/O hiccup, brief lock contention
beyond `busy_timeout=5000`) thus permanently lost block data with no
user-visible signal. The worst observed consequence: a failed first
drain drops the leading `user` message, leaving an assistant-first
conversation. Anthropic's Messages API rejects assistant-first
sequences with HTTP 400, so the session becomes permanently
un-resumable — the user sees an opaque API error with no indication
that the root cause was a transient write failure.

The lazy-session-row path was already defended (a failed first drain
leaves `_session_row_written = False`, so the next enqueue re-attempts
the row). Block data had no equivalent retry path.

## Decision

**Bounded retry inside `_drain_one`; poison-row drop after exhaustion;
`_stop.wait` for close-path interruptibility.**

### 1. Retry policy

`_WRITE_RETRY_BACKOFFS = (0.1, 0.5, 2.0)` — three retries with
linear-ish backoff. Covers momentary I/O hiccups (sub-100ms), brief
lock contention (sub-500ms), and short filesystem stalls (sub-2s).
Worst-case drain latency: 0.1 + 0.5 + 2.0 = 2.6 s, within `close()`'s
5 s consumer-join ceiling — close() is not blocked by a sleeping retry.

The retry loop is in-place inside `_drain_one`, not a separate
re-enqueue path. The queue stays FIFO; rows carry their pre-allocated
`seq` / `msg_idx` / `block_idx`, so retries do not reorder anything —
they re-attempt the same INSERT.

### 2. Poison-row guard

After the third retry fails (4 total attempts: 1 initial + 3 retries),
the batch is dropped: a CRITICAL log records the seq range and last
error; the consumer continues with the next queue item. Without this
guard a permanent error (corrupt schema, unreachable disk) would have
the consumer retry the same batch forever — every subsequent enqueue
would queue behind it, and `close()` would always time out.

The loss ceiling matches `kill -9`: the in-memory `Agent._messages`
still has the rows for the current process, but durability is lost for
the dropped batch.

### 3. Close-path interruptibility

Backoff uses `self._stop.wait(backoff)` rather than `time.sleep`. On
`close()` the event is set, the wait returns `True` immediately, and
the drain abandons the batch. Without this, a long backoff would
always exceed `close()`'s 5 s consumer-join — the consumer would be
left alive on every transient failure that hit the 2 s slot.

### 4. Lazy-row flag contract unchanged

`_session_row_written` is set only after `write_atomic` commits —
unchanged from #42. Under persistent failure the flag stays `False`,
so the next successful drain re-attempts the lazy row alongside its
own block rows.

## Considered

- **Load-time assistant-first guard (Option B in #43).** On `load`,
  validate first message is `role="user"`; drop a leading assistant
  message or insert a sentinel. Rejected as the primary fix: hides the
  symptom (data is still lost), doesn't recover anything, and only
  covers the assistant-first edge case (mid-conversation losses stay
  broken). May land as a separate defensive follow-up.
- **Fail-fast (Option C in #43).** Re-raise from `_drain_one`, killing
  the consumer. Rejected: a single transient error would kill the
  session. Violates the swallow-don't-crash contract documented in the
  #42 ponytail comment on the same path.
- **Persistent retry queue (survives process restart).** Would need a
  WAL of pending writes — too much for this fix. Current loss ceiling
  matches the documented `kill -9` ceiling.
- **Exponential backoff.** Rejected: the realistic transient errors
  resolve in sub-second; exponential backoff (e.g. 1s, 2s, 4s) would
  push worst-case drain latency past 7 s, breaking `close()`'s 5 s
  join.

## Consequences

- Transient failures (the issue's primary scenario) are recovered
  in-line; both messages persist as if the first attempt never
  happened.
- Persistent failures: bounded 2.6 s latency, then drop. CRITICAL log
  surfaces the loss; future telemetry can key off the message format.
- `close()` remains interruptible during backoff — no regression on
  the 5 s join ceiling.
- `_session_row_written` flag contract preserved exactly; under
  persistent failure the lazy row is re-attempted on the next
  successful drain.
- The `_dropped_block_count` field was considered as a diagnostic
  counter but dropped in review: the CRITICAL log already carries the
  loss count, and tests assert via `caplog` (AGENTS.md: "no boilerplate
  nobody asked for").
