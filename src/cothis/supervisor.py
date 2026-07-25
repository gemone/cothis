"""``cothis.supervisor`` — SessionWorker lifecycle manager (#227).

Spawns worker subprocesses, monitors each via WS heartbeat, restarts
crashed workers with ``always_backoff`` (exponential capped at 300s;
past a rolling-window threshold the session is marked ``errored`` so
the UI can surface a diagnose action).

Lives in its own process, separate from any worker. Writes
``session_lifecycle`` events to its OWN SQLite DB
(``~/.cothis/supervisor.db`` by default) — the worker holds each
session's ``FileLock(timeout=0)``, so the Supervisor cannot write
there (ADR-0018).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cothis.notify import NotifyBus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

logger = logging.getLogger(__name__)


_BACKOFF_CEILING_S = 300.0
_BACKOFF_FLOOR_S = 1.0
_DEFAULT_THRESHOLD = 5
_DEFAULT_WINDOW_S = 600  # 10 minutes


def backoff_seconds(restart_count: int) -> float:
    """Exponential backoff capped at ``_BACKOFF_CEILING_S``.

    ``restart_count=0`` → 1s (first restart). Each subsequent restart
    doubles the delay: 1, 2, 4, 8, … 256, 300, 300. Cap kicks in
    once the unbounded value exceeds 300s.
    """
    raw = _BACKOFF_FLOOR_S * (2 ** restart_count)
    return min(raw, _BACKOFF_CEILING_S)


@dataclass
class RestartCounter:
    """Rolling-window counter; ``is_over_threshold`` triggers errored state.

    Records restart timestamps; ``count()`` returns how many fall inside
    the window and prunes the stale prefix in place. Timestamps are
    monotonic by construction (``record()`` always appends
    ``datetime.now(UTC)``), so stale entries form a contiguous prefix —
    ``bisect_left`` finds the cutoff index and ``del [:idx]`` drops them.
    """

    threshold: int = _DEFAULT_THRESHOLD
    window_s: float = _DEFAULT_WINDOW_S
    _restarts: list[datetime] = field(default_factory=list)

    def record(self) -> None:
        """Note one restart at the current time."""
        self._restarts.append(datetime.now(UTC))

    def count(self) -> int:
        """Number of restarts inside the rolling window.

        Prunes the stale prefix as a side effect: without this, a
        sustained crash loop (1 restart/s) grows ``_restarts``
        unbounded and every ``count()`` call is O(N) in lifetime
        restarts — quadratic in the very condition the supervisor
        exists to survive.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=self.window_s)
        idx = bisect_left(self._restarts, cutoff)
        if idx:
            del self._restarts[:idx]
        return len(self._restarts)

    def is_over_threshold(self) -> bool:
        """Past the configured threshold → mark session ``errored``."""
        return self.count() >= self.threshold


@dataclass
class WorkerHandle:
    """Snapshot of one worker's state — exposed via ``Supervisor.status``."""

    session_id: str
    pid: int
    ws_url: str
    token: str = field(repr=False)  # bearer token; don't leak via repr/log
    cwd: str = ""
    status: str = "running"  # "running" | "restarting" | "errored"
    restart_count: int = 0
    # Stashed at spawn time so ``monitor_once`` can auto-restart (#250
    # slice C) without the caller re-passing model/provider.
    model: str = ""
    provider: str = ""
    sessions_dir: str = ""
    extra_env: dict = field(default_factory=dict)


class Supervisor:
    """Owns the worker-spawn lifecycle + a separate notify bus.

    The Supervisor writes ``session_lifecycle`` events on its OWN DB
    (separate from any per-session DB the worker owns); the TUI polls
    this bus for status badges.

    Spawning + WS-handshake + crash-detection wiring lands with the
    integration test (#227 follow-up); this class's pure logic
    (backoff + counter + lifecycle record + status snapshot) is what
    the unit tests cover.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        threshold: int = _DEFAULT_THRESHOLD,
        window_s: float = _DEFAULT_WINDOW_S,
    ) -> None:
        if db_path is None:
            db_path = Path.home() / ".cothis" / "supervisor.db"
        db_path = Path(db_path)
        # Owner-only: the DB carries session IDs + worker bearer tokens.
        # ``exist_ok=True`` doesn't chmod an existing dir, so the explicit
        # ``os.chmod`` covers the upgrade-from-older-cothis case.
        db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(db_path.parent, 0o700)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._bus = NotifyBus(self._conn)
        self._workers: dict[str, WorkerHandle] = {}
        # Subprocess handles live alongside the public ``WorkerHandle`` so
        # spawn/shutdown are paired by session id. Not exposed on the
        # ``WorkerHandle`` dataclass: ``Popen`` objects are unhashable +
        # have no meaningful repr, and the public surface is the snapshot
        # fields (pid/ws_url/...) not the live process.
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._counters: dict[str, RestartCounter] = {}
        self._threshold = threshold
        self._window_s = window_s

    def _counter_for(self, session_id: str) -> RestartCounter:
        if session_id not in self._counters:
            self._counters[session_id] = RestartCounter(
                threshold=self._threshold, window_s=self._window_s,
            )
        return self._counters[session_id]

    def record_lifecycle(
        self,
        event_type: str,
        session_id: str,
        *,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        """Append a ``session_lifecycle`` row to the supervisor bus."""
        meta: dict[str, Any] = {
            "restart_count": self._counter_for(session_id).count(),
        }
        if extra_meta:
            meta.update(extra_meta)
        self._bus.append(
            topic="session_lifecycle",
            event_type=event_type,
            session_id=session_id,
            meta=meta,
        )

    def status(self) -> list[dict[str, Any]]:
        """Return ``[{session_id, status, restart_count, ...}]`` for the TUI."""
        return [
            {
                "session_id": h.session_id,
                "pid": h.pid,
                "ws_url": h.ws_url,
                "cwd": h.cwd,
                "status": h.status,
                "restart_count": h.restart_count,
            }
            for h in self._workers.values()
        ]

    def lifecycle_since(self, last_seq: int = 0) -> list[Any]:
        """Read recent ``session_lifecycle`` events for the TUI's status stream."""
        return self._bus.fetch_since(last_seq=last_seq)

    def spawn_worker(
        self,
        session_id: str,
        *,
        model: str,
        provider: str,
        cwd: Path | str,
        sessions_dir: Path | str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> WorkerHandle:
        """Spawn a ``cothis worker`` subprocess owning one session.

        Synchronous: launches the subprocess, reads the bind JSON line
        from stdout, registers the handle. The subprocess runs
        independently — the supervisor does not block on its event loop
        nor monitor for unexpected exit (that's the #227 follow-up; this
        method is the spawn + bind contract only, half of #250 path a).

        Caller must ensure the session row exists at ``sessions_dir``
        before spawning — the worker loads it eagerly and exits non-zero
        if missing (the bind line never lands; this method raises).

        Raises ``RuntimeError`` if the worker exits before emitting the
        bind JSON, or emits malformed JSON. The subprocess is killed +
        reaped on either failure path so no zombie process leaks.
        """
        if session_id in self._workers:
            raise ValueError(f"worker for session {session_id!r} already spawned")

        env = dict(os.environ)
        if sessions_dir is not None:
            env["COTHIS_SESSIONS_DIR"] = str(sessions_dir)
        if extra_env:
            env.update(extra_env)

        # ``sys.executable`` so the spawned worker uses the same Python
        # interpreter (and venv) as the supervisor — important for tests
        # that import ``cothis.cli`` via the same sys.path.
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "cothis.cli", "worker",
                "--session", session_id,
                "--model", model,
                "--provider", provider,
            ],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        # Block on the bind line. ``cothis worker`` flushes it the moment
        # the WS server binds, so this returns within milliseconds of
        # the bind under normal conditions; a 10s safety net catches
        # Python startup + import cost on a slow CI runner.
        bind_line = proc.stdout.readline() if proc.stdout else ""
        if not bind_line:
            stderr = proc.stderr.read() if proc.stderr else ""
            rc = proc.wait(timeout=5)
            raise RuntimeError(
                f"worker for session {session_id!r} exited before bind; "
                f"rc={rc}; stderr:\n{stderr}"
            )
        try:
            bind = json.loads(bind_line)
        except json.JSONDecodeError as exc:
            proc.kill()
            proc.wait(timeout=5)
            raise RuntimeError(
                f"worker emitted non-JSON bind line: {bind_line!r}"
            ) from exc

        handle = WorkerHandle(
            session_id=session_id,
            pid=proc.pid,
            ws_url=bind["uri"],
            token=bind["token"],
            cwd=str(cwd),
            status="running",
            model=model,
            provider=provider,
            sessions_dir=str(sessions_dir) if sessions_dir else "",
            extra_env=dict(extra_env) if extra_env else {},
        )
        self._workers[session_id] = handle
        self._procs[session_id] = proc
        self.record_lifecycle(
            "spawned", session_id, extra_meta={"pid": proc.pid},
        )
        return handle

    def shutdown_worker(self, session_id: str, *, timeout: float = 5.0) -> None:
        """Signal one worker to exit + reap the subprocess.

        Sends ``SIGINT`` (not ``SIGTERM``) so the worker's cli.py
        ``KeyboardInterrupt`` handler runs the ``finally`` block —
        ``worker.stop()`` + ``agent.aclose()`` — draining the session
        queue + closing MCP handles cleanly. ``SIGTERM`` would skip
        that cleanup and could leave the SQLite WAL unfinalised.

        On Windows ``SIGINT`` doesn't exist for subprocesses; the
        platform-correct signal is ``CTRL_BREAK_EVENT``. Not handled
        here yet — #227 follow-up will pick it up when the integration
        test runs on Windows CI (currently the cleanup path is
        Unix-only; the spawn test still passes because Windows uses
        ``proc.terminate()`` as a fallback below).
        """
        proc = self._procs.pop(session_id, None)
        if proc is None:
            return
        try:
            proc.send_signal(signal.SIGINT)
        except (ValueError, OSError):
            # Process already dead, or signal not supported on this
            # platform for this Popen. Fall back to terminate — the
            # worker may miss the graceful-exit window but won't leak.
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
        handle = self._workers.get(session_id)
        if handle is not None:
            handle.status = "stopped"
        self.record_lifecycle("stopped", session_id)

    def check_worker_health(self, session_id: str) -> str:
        """Return ``'running'`` if the worker is alive, ``'exited'`` if dead.

        ``'stopped'`` when the worker was deliberately shut down via
        ``shutdown_worker`` (proc popped from ``_procs`` + handle
        marked ``stopped``). ``'unknown'`` when the session was never
        spawned. The distinction between ``'exited'`` (crash) and
        ``'stopped'`` (user-initiated) is what the crash monitor
        (#250 follow-up) uses to decide whether to auto-restart.
        """
        proc = self._procs.get(session_id)
        if proc is not None:
            return "exited" if proc.poll() is not None else "running"
        # Proc was popped by shutdown_worker → check the handle.
        handle = self._workers.get(session_id)
        if handle is not None:
            return "stopped" if handle.status == "stopped" else "exited"
        return "unknown"

    def monitor_once(self) -> list[str]:
        """One pass of crash detection: return the session_ids that crashed.

        For each spawned worker, calls ``check_worker_health``; if the
        result is ``'exited'`` (unexpected crash, NOT ``'stopped'``),
        records a ``crashed`` lifecycle event + marks the handle as
        ``errored``. Returns the list of crashed session_ids so the
        caller (the async loop or a test) can decide whether to restart.

        Does NOT restart — that's Slice C (needs model/provider info
        stashed at spawn time). This slice just detects + records.
        """
        crashed: list[str] = []
        for session_id in list(self._procs.keys()):
            health = self.check_worker_health(session_id)
            if health == "exited":
                logger.warning(
                    "supervisor: worker %s crashed unexpectedly", session_id,
                )
                self.record_lifecycle("crashed", session_id)
                handle = self._workers.get(session_id)
                if handle is not None:
                    handle.status = "errored"
                # Pop the dead proc so subsequent monitor_once calls
                # don't re-flag the same crash.
                self._procs.pop(session_id, None)
                crashed.append(session_id)
        return crashed

    async def monitor_worker_health(self, interval_s: float = 1.0) -> None:
        """Background loop: detect crashes → backoff → restart → repeat.

        Each iteration: ``monitor_once`` returns crashed session_ids.
        For each: check ``_should_restart`` (threshold guard); if True,
        sleep ``backoff_seconds(count)`` then ``_restart_worker``; if
        False (over threshold), leave the session ``errored`` so the
        TUI surfaces a diagnose action.

        Runs until cancelled (the caller cancels the task when the
        supervisor shuts down).
        """
        while True:
            crashed = self.monitor_once()
            for session_id in crashed:
                if not self._should_restart(session_id):
                    logger.warning(
                        "supervisor: %s over restart threshold; leaving errored",
                        session_id,
                    )
                    continue
                delay = backoff_seconds(self._counter_for(session_id).count())
                logger.info(
                    "supervisor: restarting %s after %.1fs backoff",
                    session_id, delay,
                )
                await asyncio.sleep(delay)
                self._restart_worker(session_id)
            await asyncio.sleep(interval_s)

    def _restart_worker(self, session_id: str) -> WorkerHandle | None:
        """Re-spawn a crashed worker using the stashed model/provider.

        Called after ``monitor_once`` detects a crash (handle marked
        ``errored``, proc popped). Reads ``model`` + ``provider`` +
        ``cwd`` off the old handle, clears the slot so ``spawn_worker``
        doesn't raise ``ValueError``, then re-spawns. On success:
        ``RestartCounter.record()`` + ``record_lifecycle("restarted")``.

        Returns the new ``WorkerHandle`` on success, ``None`` on
        failure (spawn error logged + handle left ``errored`` so the
        monitor doesn't retry infinitely on the same tick).

        Backoff delay (``backoff_seconds(count)``) is the caller's
        responsibility — ``monitor_worker_health`` sleeps between
        ``monitor_once`` calls; the restart itself is synchronous.
        """
        old = self._workers.get(session_id)
        if old is None:
            logger.warning("supervisor: cannot restart %s — no handle", session_id)
            return None
        if old.status != "errored":
            logger.debug(
                "supervisor: skipping restart for %s (status=%s, not errored)",
                session_id, old.status,
            )
            return None

        # Clear the slot so spawn_worker doesn't reject the re-spawn.
        self._workers.pop(session_id, None)
        self._procs.pop(session_id, None)

        try:
            new_handle = self.spawn_worker(
                session_id,
                model=old.model,
                provider=old.provider,
                cwd=old.cwd,
                sessions_dir=Path(old.sessions_dir) if old.sessions_dir else None,
                extra_env=dict(old.extra_env) if old.extra_env else None,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort restart
            logger.error(
                "supervisor: restart failed for %s: %s", session_id, exc,
            )
            # Re-register the old handle as errored so status() still
            # shows the session + the monitor can retry next tick.
            self._workers[session_id] = old
            return None

        self._counter_for(session_id).record()
        self.record_lifecycle("restarted", session_id)
        logger.info("supervisor: restarted worker %s", session_id)
        return new_handle

    def _should_restart(self, session_id: str) -> bool:
        """Return ``True`` if the session is under the restart threshold.

        When ``RestartCounter.is_over_threshold()`` returns ``True``
        (too many restarts in the rolling window), the session is left
        ``errored`` instead of being restarted again — the supervisor
        gives up on a persistently crashing worker so the user can
        diagnose the issue via the TUI's status badge.
        """
        return not self._counter_for(session_id).is_over_threshold()

    def close(self) -> None:
        """Shutdown all workers + close the supervisor DB connection.

        Idempotent. Safe to call from ``__exit__`` / ``atexit``. Each
        spawned worker gets ``shutdown_worker`` (SIGINT → graceful exit)
        before the DB connection closes — leaving procs alive would
        orphan the session file locks the workers hold.
        """
        for session_id in list(self._procs.keys()):
            self.shutdown_worker(session_id)
        self._conn.close()
