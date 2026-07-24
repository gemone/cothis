"""Tests for ``cothis.supervisor`` (#227).

Covers:

- Backoff math (exponential, capped at 300s, deterministic per
  ``restart_count``).
- Restart counter with a rolling window; threshold → ``errored``.
- ``session_lifecycle`` events on the supervisor DB.
- Status stream surfaces ``{session_id, status, restart_count}``.

Pure-function tests use the public helpers; integration tests would
spawn a real worker subprocess but are out of scope for this file
(the issue's integration test lands when #225's CLI entrypoint is
finalised).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------
# Backoff math
# ---------------------------------------------------------------------


def test_backoff_first_restart_is_min_delay() -> None:
    """``restart_count=0`` → minimum backoff (1s)."""
    from cothis.supervisor import backoff_seconds

    assert backoff_seconds(restart_count=0) == 1.0


def test_backoff_exponential() -> None:
    """Each successive restart doubles the delay."""
    from cothis.supervisor import backoff_seconds

    assert backoff_seconds(restart_count=1) == 2.0
    assert backoff_seconds(restart_count=2) == 4.0
    assert backoff_seconds(restart_count=3) == 8.0


def test_backoff_capped_at_300s() -> None:
    """Past 8 doublings, backoff stays at the 300s ceiling."""
    from cothis.supervisor import backoff_seconds

    assert backoff_seconds(restart_count=8) == 256.0
    assert backoff_seconds(restart_count=9) == 300.0
    assert backoff_seconds(restart_count=20) == 300.0


# ---------------------------------------------------------------------
# Restart counter (rolling window + threshold)
# ---------------------------------------------------------------------


def test_restart_counter_starts_empty() -> None:
    """Fresh counter: zero restarts, status ``running``."""
    from cothis.supervisor import RestartCounter

    rc = RestartCounter(threshold=5, window_s=600)
    assert rc.count() == 0
    assert not rc.is_over_threshold()


def test_restart_counter_increments() -> None:
    """Each ``record()`` bumps the count."""
    from cothis.supervisor import RestartCounter

    rc = RestartCounter(threshold=5, window_s=600)
    rc.record()
    rc.record()
    assert rc.count() == 2


def test_restart_counter_threshold_marks_over() -> None:
    """Past N restarts in the window, ``is_over_threshold`` returns True."""
    from cothis.supervisor import RestartCounter

    rc = RestartCounter(threshold=3, window_s=600)
    for _ in range(3):
        rc.record()
    assert rc.is_over_threshold()


def test_restart_counter_rolling_window_expires_old_entries() -> None:
    """Restarts older than ``window_s`` drop out of the count."""
    from cothis.supervisor import RestartCounter

    rc = RestartCounter(threshold=5, window_s=600)
    # Seed 3 restarts, all just outside the window.
    cutoff = datetime.now(UTC) - timedelta(seconds=601)
    rc._restarts = [cutoff, cutoff, cutoff]  # type: ignore[attr-defined]
    assert rc.count() == 0
    assert not rc.is_over_threshold()


def test_restart_counter_keeps_recent_entries() -> None:
    """Recent restarts (inside the window) stay in the count."""
    from cothis.supervisor import RestartCounter

    rc = RestartCounter(threshold=5, window_s=600)
    recent = datetime.now(UTC) - timedelta(seconds=10)
    rc._restarts = [recent, recent]  # type: ignore[attr-defined]
    assert rc.count() == 2


# ---------------------------------------------------------------------
# Lifecycle events on supervisor DB
# ---------------------------------------------------------------------


def _make_supervisor_db(tmp_path: Path) -> tuple[Any, Any]:
    """Build a Supervisor backed by an in-memory SQLite DB."""
    from cothis.notify import NotifyBus
    from cothis.supervisor import _DEFAULT_THRESHOLD, _DEFAULT_WINDOW_S, Supervisor

    conn = sqlite3.connect(":memory:")
    bus = NotifyBus(conn)
    sup = Supervisor.__new__(Supervisor)  # bypass __init__ for unit test
    sup._bus = bus
    sup._workers = {}
    sup._counters = {}
    sup._threshold = _DEFAULT_THRESHOLD
    sup._window_s = _DEFAULT_WINDOW_S
    return sup, bus


def test_lifecycle_spawned_event_written(tmp_path: Path) -> None:
    """``record_lifecycle('spawned', sid)`` writes a row to the bus."""
    sup, bus = _make_supervisor_db(tmp_path)
    sup.record_lifecycle("spawned", "s1")
    events = bus.fetch_since(last_seq=0)
    assert len(events) == 1
    assert events[0].topic == "session_lifecycle"
    assert events[0].event_type == "spawned"
    assert events[0].session_id == "s1"


def test_lifecycle_restarted_event_carries_restart_count(tmp_path: Path) -> None:
    """``restarted`` events include the rolling restart_count in meta."""
    from cothis.supervisor import RestartCounter

    sup, bus = _make_supervisor_db(tmp_path)
    sup._counters["s1"] = RestartCounter(threshold=5, window_s=600)
    sup._counters["s1"].record()
    sup._counters["s1"].record()
    sup.record_lifecycle("restarted", "s1")
    events = bus.fetch_since(last_seq=0)
    assert events[0].meta["restart_count"] == 2


# ---------------------------------------------------------------------
# Status stream
# ---------------------------------------------------------------------


def test_status_stream_surfaces_session_state(tmp_path: Path) -> None:
    """``status()`` returns the current snapshot of all workers."""
    from cothis.supervisor import WorkerHandle

    sup, _ = _make_supervisor_db(tmp_path)
    sup._workers["s1"] = WorkerHandle(
        session_id="s1",
        pid=1234,
        ws_url="ws://127.0.0.1:9999/agent",
        token="tok",
        cwd="/tmp",
        status="running",
        restart_count=0,
    )
    sup._workers["s2"] = WorkerHandle(
        session_id="s2",
        pid=5678,
        ws_url="ws://127.0.0.1:8888/agent",
        token="tok2",
        cwd="/tmp",
        status="errored",
        restart_count=5,
    )
    status = sup.status()
    assert {s["session_id"] for s in status} == {"s1", "s2"}
    s2 = next(s for s in status if s["session_id"] == "s2")
    assert s2["status"] == "errored"
    assert s2["restart_count"] == 5
