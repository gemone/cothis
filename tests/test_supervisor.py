"""Tests for ``cothis.supervisor`` (#227).

Covers:

- Backoff math (exponential, capped at 300s, deterministic per
  ``restart_count``).
- Restart counter with a rolling window; threshold → ``errored``.
- ``session_lifecycle`` events on the supervisor DB.
- Status stream surfaces ``{session_id, status, restart_count}``.
- **Integration**: ``Supervisor.spawn_worker`` launches a real
  ``python -m cothis.worker`` subprocess; the worker's WS handshake,
  ``ping``/``pong``, and clean exit on SIGTERM are exercised over a
  real socket (COH-15 / GH#250).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
import websockets

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


# ---------------------------------------------------------------------
# Integration: real ``python -m cothis.worker`` subprocess (COH-15 / GH#250)
# ---------------------------------------------------------------------
#
# Spawns the actual worker CLI via ``Supervisor.spawn_worker``, connects
# over a real WebSocket with the bearer token, and verifies the lifecycle:
# handshake → ping/pong → graceful shutdown → clean process exit. This is
# the test deferred from #227 — it exercises the spawn path end-to-end
# instead of the mock-only ``SessionWorker`` tests in test_session_worker.py.


def _seed_session(home: Path, *, model: str = "openai/gpt-oss-120b") -> str:
    """Create a session whose row is persisted so the worker can load it.

    ``Session.new`` defers the ``sessions`` row until the first message
    drain; the worker calls ``Session.load`` which needs that row. We
    append one seed user message under ``flush_sync=True`` so the row
    is committed before we hand the id to the subprocess.
    """
    from cothis.session import Session

    db_path = home / "agents.db"
    sess = Session.new(db_path, cwd=home, model=model, flush_sync=True)
    sess.append_message("user", [{"type": "text", "text": "seed"}])
    sid = sess._session_id  # type: ignore[attr-defined]  # noqa: SLF001
    sess.close()
    return sid


@pytest.mark.asyncio
async def test_spawn_worker_real_subprocess_handshake_and_ping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Supervisor.spawn_worker`` launches a real worker reachable over WS.

    Regression target for the #227 deferred criterion: "integration test
    with a real throwaway worker." Verifies the full spawn path —
    subprocess bind, ready-line parse, bearer-token handshake, ping/pong —
    then tears the worker down via SIGTERM.
    """
    from cothis.supervisor import Supervisor

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("COTHIS_HOME", str(home))
    # Any non-empty key satisfies any-llm's provider construction check;
    # no network call happens until ``run_turn`` (#225 verified this).
    monkeypatch.setenv("OPENROUTER_API_KEY", "***dummy***-integration-test")
    sid = _seed_session(home)

    sup = Supervisor(home / "supervisor.db")
    try:
        handle = sup.spawn_worker(
            sid, model="openai/gpt-oss-120b", provider="openrouter", cwd=str(home)
        )
        assert handle.session_id == sid
        assert handle.ws_url.startswith("ws://127.0.0.1:")
        assert len(handle.token) >= 32
        assert handle.status == "running"

        async with websockets.connect(
            handle.ws_url,
            additional_headers={"Authorization": f"Bearer {handle.token}"},
        ) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            assert json.loads(raw) == {"type": "pong"}

        # ``spawned`` lifecycle event was recorded on the supervisor bus.
        events = sup.lifecycle_since(0)
        spawned = [e for e in events if e.event_type == "spawned"]
        assert spawned, f"expected a 'spawned' event; got {[e.event_type for e in events]}"
    finally:
        sup.stop_worker(sid)
        sup.close()

    # After stop_worker the process is gone.
    proc: Any = getattr(handle, "_proc", None)  # type: ignore[attr-defined]  # noqa: SLF001
    assert proc is not None
    assert proc.poll() is not None, "worker process did not exit after stop_worker"


@pytest.mark.asyncio
async def test_spawn_worker_records_pid_and_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``spawned`` event carries the worker PID in its meta."""
    from cothis.supervisor import Supervisor

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("COTHIS_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "***dummy***-integration-test")
    sid = _seed_session(home)

    sup = Supervisor(home / "supervisor.db")
    try:
        handle = sup.spawn_worker(
            sid, model="openai/gpt-oss-120b", provider="openrouter", cwd=str(home)
        )
        events = sup.lifecycle_since(0)
        spawned = next(e for e in events if e.event_type == "spawned")
        assert spawned.session_id == sid
        assert spawned.meta["pid"] == handle.pid
    finally:
        sup.stop_worker(sid)
        sup.close()


def test_stop_worker_exits_process_on_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stop_worker`` (SIGTERM) makes the worker subprocess exit.

    The acceptance criterion was "a real throwaway worker that exits on
    signal." This is the direct check: spawn, then terminate, then assert
    the OS reports the process gone within the wait ceiling.
    """
    from cothis.supervisor import Supervisor

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("COTHIS_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "***dummy***-integration-test")
    sid = _seed_session(home)

    sup = Supervisor(home / "supervisor.db")
    try:
        handle = sup.spawn_worker(
            sid, model="openai/gpt-oss-120b", provider="openrouter", cwd=str(home)
        )
        proc: Any = getattr(handle, "_proc", None)  # type: ignore[attr-defined]  # noqa: SLF001
        assert proc is not None and proc.poll() is None  # alive
        sup.stop_worker(sid, timeout=10.0)
        assert proc.poll() is not None, "worker did not exit within 10s of SIGTERM"
        # ``stopped`` lifecycle event recorded.
        events = sup.lifecycle_since(0)
        assert any(e.event_type == "stopped" for e in events)
    finally:
        sup.close()


def test_spawn_worker_raises_when_session_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawning for a non-existent session raises (worker exits pre-bind)."""
    from cothis.supervisor import Supervisor

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("COTHIS_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "***dummy***-integration-test")

    sup = Supervisor(home / "supervisor.db")
    try:
        # A well-formed but non-existent 32-char hex id.
        bogus_sid = "0" * 32
        with pytest.raises(RuntimeError, match="exited .* before ready line"):
            sup.spawn_worker(
                bogus_sid, model="openai/gpt-oss-120b", provider="openrouter",
                cwd=str(home),
            )
    finally:
        sup.close()
