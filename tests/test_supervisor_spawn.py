"""Integration test for ``Supervisor.spawn_worker`` (#250 path a — second half).

Drives the real end-to-end spawn → bind-handshake → ping/pong → shutdown
contract that ``Supervisor`` will use to manage workers in production.
No mocks — the supervisor spawns a real ``cothis worker`` subprocess,
reads the bind JSON line from its stdout, and the test connects a real
WS client to the resulting URI.

Companion to ``tests/test_worker_cli_integration.py`` (#250 path a,
entrypoint half): that test drives the worker side directly via
``subprocess.Popen``; this test drives the supervisor side. Together
they cover the full #250 path (a) slice — supervisor + worker
subprocess, no WS attached yet (the TUI attachment is #252).

Out of scope (tracked under #227 full follow-up):
- Crash detection (subprocess exits unexpectedly → restart with backoff)
- WS heartbeat monitoring
- Restart-on-threshold policy
- Multi-worker concurrency stress test
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import websockets

from cothis.session import Session
from cothis.supervisor import Supervisor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_spawn_worker_returns_handle_with_live_ws(tmp_path: Path) -> None:
    """AC #250 path (a): spawn_worker returns a handle whose WS is reachable.

    Verifies the bind-handshake contract end-to-end:

      1. Pre-create a session row the worker can load.
      2. Construct Supervisor with a temp DB; spawn the worker pointing
         at the sessions dir.
      3. Handle has pid, ws_url, token; the WS handshake accepts the
         bearer token.
      4. ping → pong round-trip on the spawned worker's WS.
      5. ``shutdown_worker`` reaps the subprocess cleanly.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    db_path = sessions_dir / "session.db"

    session = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    session.append_message("user", [{"type": "text", "text": "hi"}])
    sid = session.session_id
    session.close()

    supervisor_db = tmp_path / "supervisor.db"
    sup = Supervisor(supervisor_db)
    try:
        handle = sup.spawn_worker(
            sid,
            model="openai/gpt-oss-120b",
            provider="openrouter",
            cwd=tmp_path,
            sessions_dir=sessions_dir,
            extra_env={
                # Agent constructor validates the provider API key eagerly;
                # CI has no real key. The test never makes an LLM call.
                "OPENROUTER_API_KEY": "test-dummy-not-used",
            },
        )

        # Surface checks on the handle.
        assert handle.session_id == sid
        assert handle.pid > 0
        assert handle.ws_url.startswith("ws://127.0.0.1:")
        assert len(handle.token) >= 32
        assert handle.status == "running"
        # model + provider stashed for auto-restart (#250 slice C enabler)
        assert handle.model == "openai/gpt-oss-120b"
        assert handle.provider == "openrouter"

        # Real WS round-trip via the supervisor-mediated URI + token.
        async with websockets.connect(
            handle.ws_url,
            additional_headers={"Authorization": f"Bearer {handle.token}"},
        ) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            assert json.loads(raw) == {"type": "pong"}

        # status() now lists the spawned worker.
        snapshot = sup.status()
        assert any(s["session_id"] == sid for s in snapshot)

        sup.shutdown_worker(sid)
        assert sup._procs.get(sid) is None
        # Handle stays in _workers but status flips to stopped.
        assert sup._workers[sid].status == "stopped"
    finally:
        sup.close()


@pytest.mark.asyncio
async def test_spawn_worker_unknown_session_raises(tmp_path: Path) -> None:
    """Spawn against a sessions_dir with no such session → RuntimeError, no leak.

    The worker subprocess exits non-zero before binding; the supervisor's
    spawn_worker detects the missing bind line, kills the proc, and
    raises. No zombie process, no half-registered handle.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    supervisor_db = tmp_path / "supervisor.db"

    sup = Supervisor(supervisor_db)
    try:
        with pytest.raises(RuntimeError, match="exited before bind"):
            sup.spawn_worker(
                "0" * 32,
                model="openai/gpt-oss-120b",
                provider="openrouter",
                cwd=tmp_path,
                sessions_dir=sessions_dir,
                extra_env={"OPENROUTER_API_KEY": "test-dummy-not-used"},
            )
        # No partial state on failure.
        assert "0" * 32 not in sup._workers
        assert "0" * 32 not in sup._procs
    finally:
        sup.close()


def test_spawn_worker_duplicate_session_id_rejected(tmp_path: Path) -> None:
    """Calling spawn_worker twice for the same session raises ValueError.

    Synchronous guard — the supervisor does not double-spawn. The first
    successful spawn owns the slot until ``shutdown_worker`` clears it.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    db_path = sessions_dir / "session.db"

    session = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    session.append_message("user", [{"type": "text", "text": "hi"}])
    sid = session.session_id
    session.close()

    sup = Supervisor(tmp_path / "supervisor.db")
    try:
        sup.spawn_worker(
            sid,
            model="openai/gpt-oss-120b",
            provider="openrouter",
            cwd=tmp_path,
            sessions_dir=sessions_dir,
            extra_env={"OPENROUTER_API_KEY": "test-dummy-not-used"},
        )
        with pytest.raises(ValueError, match="already spawned"):
            sup.spawn_worker(
                sid,
                model="openai/gpt-oss-120b",
                provider="openrouter",
                cwd=tmp_path,
                sessions_dir=sessions_dir,
                extra_env={"OPENROUTER_API_KEY": "test-dummy-not-used"},
            )
    finally:
        sup.close()


# ---------------------------------------------------------------------
# check_worker_health (#250 crash-monitoring foundation)
# ---------------------------------------------------------------------


def test_check_worker_health_unknown_for_unspawned(tmp_path: Path) -> None:
    """AC #250: returns ``'unknown'`` for a never-spawned session."""
    sup = Supervisor(tmp_path / "supervisor.db")
    try:
        assert sup.check_worker_health("0" * 32) == "unknown"
    finally:
        sup.close()


@pytest.mark.asyncio
async def test_check_worker_health_running_then_exited(tmp_path: Path) -> None:
    """AC #250: ``'running'`` while alive, ``'exited'`` after shutdown."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    db_path = sessions_dir / "session.db"

    session = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    session.append_message("user", [{"type": "text", "text": "hi"}])
    sid = session.session_id
    session.close()

    sup = Supervisor(tmp_path / "supervisor.db")
    try:
        sup.spawn_worker(
            sid,
            model="openai/gpt-oss-120b",
            provider="openrouter",
            cwd=tmp_path,
            sessions_dir=sessions_dir,
            extra_env={"OPENROUTER_API_KEY": "test-dummy-not-used"},
        )
        assert sup.check_worker_health(sid) == "running"

        sup.shutdown_worker(sid)
        # After deliberate shutdown_worker: handle is marked "stopped"
        # (not "exited", which would be an unexpected crash).
        assert sup.check_worker_health(sid) == "stopped"
    finally:
        sup.close()


# ---------------------------------------------------------------------
# monitor_once — crash detection (#250 slice B)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitor_once_detects_crash_and_records_lifecycle(
    tmp_path: Path,
) -> None:
    """AC #250 slice B: ``monitor_once`` detects a crashed worker + records lifecycle.

    Spawns a real worker, SIGKILLs it to simulate a crash, calls
    ``monitor_once`` — verifies the crash was detected, the lifecycle
    ``crashed`` event was recorded, + the handle is marked ``errored``.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    db_path = sessions_dir / "session.db"

    session = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    session.append_message("user", [{"type": "text", "text": "hi"}])
    sid = session.session_id
    session.close()

    sup = Supervisor(tmp_path / "supervisor.db")
    try:
        sup.spawn_worker(
            sid,
            model="openai/gpt-oss-120b",
            provider="openrouter",
            cwd=tmp_path,
            sessions_dir=sessions_dir,
            extra_env={"OPENROUTER_API_KEY": "test-dummy-not-used"},
        )
        assert sup.check_worker_health(sid) == "running"

        # Simulate a crash: SIGKILL the subprocess (not graceful SIGINT).
        proc = sup._procs[sid]
        proc.kill()
        proc.wait(timeout=5)

        # monitor_once should detect the crash.
        crashed = sup.monitor_once()
        assert sid in crashed

        # Lifecycle event recorded.
        events = sup.lifecycle_since(0)
        crash_events = [e for e in events if e.event_type == "crashed"]
        assert len(crash_events) == 1
        assert crash_events[0].session_id == sid

        # Handle marked errored.
        assert sup._workers[sid].status == "errored"

        # Second call: no re-flag (proc popped by monitor_once).
        crashed_again = sup.monitor_once()
        assert crashed_again == []
    finally:
        sup.close()


# ---------------------------------------------------------------------
# _restart_worker (#250 slice C — auto-restart)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_worker_re_spawns_after_crash(tmp_path: Path) -> None:
    """AC #250 slice C: ``_restart_worker`` re-spawns a crashed worker.

    Kill the worker (SIGKILL), detect via ``monitor_once``, then call
    ``_restart_worker`` — verifies the new worker is running + the
    ``restarted`` lifecycle event is recorded.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    db_path = sessions_dir / "session.db"

    session = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    session.append_message("user", [{"type": "text", "text": "hi"}])
    sid = session.session_id
    session.close()

    sup = Supervisor(tmp_path / "supervisor.db")
    try:
        sup.spawn_worker(
            sid,
            model="openai/gpt-oss-120b",
            provider="openrouter",
            cwd=tmp_path,
            sessions_dir=sessions_dir,
            extra_env={"OPENROUTER_API_KEY": "test-dummy-not-used"},
        )

        # Crash the worker.
        proc = sup._procs[sid]
        proc.kill()
        proc.wait(timeout=5)
        sup.monitor_once()
        assert sup._workers[sid].status == "errored"

        # Restart.
        new_handle = sup._restart_worker(sid)
        assert new_handle is not None
        assert new_handle.status == "running"
        assert new_handle.model == "openai/gpt-oss-120b"
        assert new_handle.provider == "openrouter"
        assert sup.check_worker_health(sid) == "running"

        # Lifecycle: "restarted" event recorded.
        events = sup.lifecycle_since(0)
        restarted = [e for e in events if e.event_type == "restarted"]
        assert len(restarted) == 1
        assert restarted[0].session_id == sid

        # Clean up the re-spawned worker.
        sup.shutdown_worker(sid)
    finally:
        sup.close()


# ---------------------------------------------------------------------
# monitor_worker_health — continuous loop (#250 slice C)
#
# The loop orchestrates three pieces: monitor_once (detect), backoff_seconds
# (wait), _restart_worker (re-spawn). Each piece has its own unit test; this
# covers the wiring — that the loop actually calls them in order and stops
# re-spawning once the worker is healthy (monitor_once returns empty).
#
# Hermetic via monkeypatch — no real subprocess. A real-subprocess version
# would be flaky on Windows (subprocess kill timing) and add ~5s of wall
# time per iteration for backoff, which makes CI slow.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitor_worker_health_loops_detect_backoff_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #250 slice C: the loop detects a crash + restarts after backoff.

    One crash → one restart; the second iteration sees the recovered
    worker + stops re-spawning. Verifies the wiring between
    ``monitor_once`` → ``backoff_seconds`` → ``_restart_worker``.
    """
    import cothis.supervisor as sup_mod

    sup = Supervisor(tmp_path / "supervisor.db")
    try:
        sid = "session_loop_test"

        # Stub monitor_once: return sid once, then empty (worker recovered).
        monitor_calls: list[int] = []
        def fake_monitor_once() -> list[str]:
            monitor_calls.append(len(monitor_calls))
            return [sid] if len(monitor_calls) == 1 else []
        monkeypatch.setattr(sup, "monitor_once", fake_monitor_once)

        # Stub _restart_worker: record the call (no real spawn).
        restart_calls: list[str] = []
        def fake_restart(session_id: str) -> None:
            restart_calls.append(session_id)
        monkeypatch.setattr(sup, "_restart_worker", fake_restart)

        # Stub _should_restart: True so the loop proceeds to restart.
        monkeypatch.setattr(sup, "_should_restart", lambda _: True)

        # Stub backoff to 0 so the test doesn't wait.
        monkeypatch.setattr(sup_mod, "backoff_seconds", lambda count: 0)

        # Run the loop in a task; cancel after enough iterations for
        # detect + restart + one verify-healthy pass.
        task = asyncio.create_task(sup.monitor_worker_health(interval_s=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Loop ran multiple iterations.
        assert len(monitor_calls) >= 2, (
            f"loop should call monitor_once at least twice; got {monitor_calls}"
        )
        # Restart called exactly once (the single crash).
        assert restart_calls == [sid], (
            f"expected one restart of {sid}; got {restart_calls}"
        )
    finally:
        sup.close()


@pytest.mark.asyncio
async def test_monitor_worker_health_skips_when_over_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #250 slice C: the loop leaves a session ``errored`` when over restart threshold.

    Threshold guard: if ``_should_restart`` returns False (too many
    crashes in the window), the loop logs + skips — doesn't keep
    hammering restart. The session stays ``errored`` so the TUI can
    surface a diagnose action.
    """
    sup = Supervisor(tmp_path / "supervisor.db")
    try:
        sid = "session_threshold_test"

        # Stub monitor_once to keep flagging the session as crashed.
        monkeypatch.setattr(sup, "monitor_once", lambda: [sid])
        # _should_restart False: over threshold.
        monkeypatch.setattr(sup, "_should_restart", lambda _: False)

        # Stub restart so we can detect if the loop erroneously calls it.
        restart_calls: list[str] = []
        monkeypatch.setattr(
            sup, "_restart_worker", lambda s: restart_calls.append(s)
        )

        task = asyncio.create_task(sup.monitor_worker_health(interval_s=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Restart never called despite repeated crash detection.
        assert restart_calls == [], (
            f"loop should NOT restart over-threshold session; got {restart_calls}"
        )
    finally:
        sup.close()


# ---------------------------------------------------------------------
# Supervisor.close() idempotency
#
# The docstring promises "Idempotent. Safe to call from ``__exit__`` /
# ``atexit``." If the TUI's atexit hook fires twice (e.g., normal
# shutdown + Python's finalisation sweep), the second call must not
# raise — orphaned procs or a leaked DB connection would be a real
# regression in a long-running session.
# ---------------------------------------------------------------------


def test_supervisor_close_is_idempotent(tmp_path: Path) -> None:
    """AC #250: ``close()`` can be called twice without raising.

    The second call returns early via the ``_closed`` guard — ``close()``
    compacts the bus before closing the connection, and compacting on an
    already-closed connection would raise, so the guard (not sqlite's
    idempotent ``Connection.close``) is what makes the double-call safe (#411).
    """
    sup = Supervisor(tmp_path / "supervisor.db")
    sup.close()
    sup.close()  # must not raise


def test_supervisor_close_runs_conn_close_if_compact_raises(
    tmp_path: Path,
) -> None:
    """#411: a ``compact`` failure must not leak the DB connection.

    ``compact`` is a DB write that can raise on a local sqlite error. The
    teardown sets ``_closed`` first and runs ``conn.close()`` under
    ``finally``, so the connection is always closed (no leak) and a
    re-entry is a no-op that does not retry ``compact`` on the
    half-closed connection.
    """
    import sqlite3

    sup = Supervisor(tmp_path / "supervisor.db")
    # compact is a NotifyBus method (plain Python) — safe to override with a
    # raising stub that records how many times it ran.
    compact_calls: list[int] = []

    def _raising_compact(*args: object, **kwargs: object) -> None:
        compact_calls.append(1)
        raise sqlite3.OperationalError("disk I/O")

    # ``setattr`` (not direct method assignment) avoids ty's implicit-
    # shadowing check on the plain-Python NotifyBus method.
    setattr(sup._bus, "compact", _raising_compact)

    # The compact error propagates, but the finally clause still closed the
    # connection and _closed is set.
    with pytest.raises(sqlite3.OperationalError):
        sup.close()

    # Connection was closed despite compact raising (no leak).
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        sup._conn.execute("SELECT 1")
    assert sup._closed is True
    # Re-entry returns early — compact is not retried on the closed conn.
    sup.close()
    assert len(compact_calls) == 1


def test_supervisor_close_runs_conn_close_if_shutdown_worker_raises(
    tmp_path: Path,
) -> None:
    """#425: a ``shutdown_worker`` raise during close() must not leak the conn.

    ``close()`` wraps the whole teardown (shutdown loop + compact) in one
    try/finally so a ``record_lifecycle`` or ``proc.wait`` raise still runs
    ``conn.close()`` (#425).
    """
    import sqlite3

    sup = Supervisor(tmp_path / "supervisor.db")
    # A registered proc so the shutdown loop runs and calls shutdown_worker.
    sup._procs["d" * 32] = MagicMock()  # type: ignore[assignment]

    def _boom(session_id: str, *, timeout: float = 5.0) -> None:
        raise sqlite3.OperationalError("disk I/O")

    # ``setattr`` (not direct method assignment) avoids ty's implicit-
    # shadowing check.
    setattr(sup, "shutdown_worker", _boom)

    # The shutdown error propagates, but the finally still closed the conn.
    with pytest.raises(sqlite3.OperationalError):
        sup.close()

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        sup._conn.execute("SELECT 1")
    assert sup._closed is True


def test_close_compacts_old_lifecycle_events(tmp_path: Path) -> None:
    """#411: ``close()`` compacts lifecycle events older than the retention
    window, so the shared ``~/.cothis/supervisor.db``'s ``notify_events`` does
    not grow without bound across sessions (the read side is not yet wired, but
    the write side is live).
    """
    import sqlite3
    from datetime import UTC, datetime, timedelta

    from cothis.supervisor import _LIFECYCLE_RETENTION_DAYS

    db_path = tmp_path / "supervisor.db"
    sup = Supervisor(db_path)
    # An event beyond the retention window (backdated directly into the table).
    old_ts = (
        datetime.now(UTC) - timedelta(days=_LIFECYCLE_RETENTION_DAYS + 1)
    ).isoformat()
    with sup._conn:
        sup._conn.execute(
            "INSERT INTO notify_events(ts, topic, event_type, session_id, "
            "meta, payload_pointer) VALUES (?, 'session_lifecycle', "
            "'spawned', NULL, NULL, NULL)",
            (old_ts,),
        )
    # A fresh event via the live write path stays.
    sup.record_lifecycle("spawned", "s1")
    sup.close()

    # Re-open the shared DB: only the fresh event survived compaction.
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM notify_events"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1, (
        f"close() should compact events older than "
        f"{_LIFECYCLE_RETENTION_DAYS}d; got {count} rows"
    )
# ---------------------------------------------------------------------
# on_restart callback (#398 — wire monitor_worker_health into production)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitor_worker_health_invokes_on_restart_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#398: ``monitor_worker_health`` calls ``on_restart`` after a restart.

    The callback lets the TUI re-attach to the restarted worker's fresh WS.
    Pre-fix the monitor loop was never started in production, so crashed
    workers silently died with no recovery. This test verifies the callback
    contract: ``on_restart(session_id, new_handle)`` is called after
    ``_restart_worker`` succeeds.
    """
    sup = Supervisor(tmp_path / "supervisor.db")

    # Register a fake crashed worker.
    from cothis.supervisor import WorkerHandle
    fake_handle = WorkerHandle(
        session_id="a" * 32, pid=999999, ws_url="ws://old", token="old",
        status="running", model="m", provider="p", cwd=str(tmp_path),
        sessions_dir=str(tmp_path), extra_env={},
    )
    sup._workers["a" * 32] = fake_handle
    fake_proc = MagicMock()
    fake_proc.poll.return_value = 1
    sup._procs["a" * 32] = fake_proc  # type: ignore[assignment]

    # Stub spawn_worker (called by _restart_worker) to return a fresh handle.
    new_handle = WorkerHandle(
        session_id="a" * 32, pid=888888, ws_url="ws://new", token="new",
        status="running", model="m", provider="p", cwd=str(tmp_path),
        sessions_dir=str(tmp_path), extra_env={},
    )
    monkeypatch.setattr(sup, "spawn_worker", lambda *a, **kw: new_handle)
    # Zero backoff so the restart fires within the test's sleep window.
    monkeypatch.setattr("cothis.supervisor.backoff_seconds", lambda count: 0.0)

    # Install the callback.
    restarted: list[tuple[str, WorkerHandle]] = []
    sup.on_restart = lambda sid, handle: restarted.append((sid, handle))

    # Run one iteration of the monitor loop (cancel after the first cycle).
    task = asyncio.create_task(sup.monitor_worker_health(interval_s=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(restarted) == 1, (
        f"on_restart should be called once after restart; got {restarted}"
    )
    assert restarted[0][0] == "a" * 32
    assert restarted[0][1].ws_url == "ws://new"
    sup.close()


@pytest.mark.asyncio
async def test_monitor_worker_health_survives_on_restart_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#398 review: a raising ``on_restart`` must not kill the monitor loop.

    ``on_restart`` runs synchronously inside ``monitor_worker_health``'s
    per-crash loop; an unguarded raise would propagate up, abort the loop,
    and disable crash detection for every worker. Two crashed workers are
    registered so a single ``monitor_once`` returns both — if the raise
    were unguarded, the second worker's restart would never be attempted.
    """
    sup = Supervisor(tmp_path / "supervisor.db")

    from cothis.supervisor import WorkerHandle

    def _fake_handle(sid: str) -> WorkerHandle:
        return WorkerHandle(
            session_id=sid, pid=999999, ws_url="ws://old", token="old",
            status="running", model="m", provider="p", cwd=str(tmp_path),
            sessions_dir=str(tmp_path), extra_env={},
        )

    sids = ["b" * 32, "c" * 32]
    for sid in sids:
        sup._workers[sid] = _fake_handle(sid)
        fake_proc = MagicMock()
        fake_proc.poll.return_value = 1
        sup._procs[sid] = fake_proc  # type: ignore[assignment]

    monkeypatch.setattr(sup, "spawn_worker", lambda sid, **kw: _fake_handle(sid))
    monkeypatch.setattr("cothis.supervisor.backoff_seconds", lambda count: 0.0)

    calls: list[str] = []

    def _raising_cb(sid, handle):  # noqa: ANN001
        calls.append(sid)
        raise RuntimeError("boom")

    sup.on_restart = _raising_cb

    task = asyncio.create_task(sup.monitor_worker_health(interval_s=0.01))
    await asyncio.sleep(0.05)
    # If the raise propagated, the task would be done with RuntimeError.
    assert not task.done(), "monitor loop died on on_restart raise"
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Both workers re-attached despite each on_restart raising — the guard
    # kept the loop alive for the second crash.
    assert set(calls) == set(sids), (
        f"expected both crashed workers re-attached; calls={calls}"
    )
    sup.close()
