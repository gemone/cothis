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

import logging
import os
import sqlite3
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

    Records restart timestamps; queries count the records still inside
    the window. The window slides forward at query time — old
    timestamps are not actively pruned (a periodic compaction would
    do that, but the cost of keeping stale timestamps is tiny).
    """

    threshold: int = _DEFAULT_THRESHOLD
    window_s: float = _DEFAULT_WINDOW_S
    _restarts: list[datetime] = field(default_factory=list)

    def record(self) -> None:
        """Note one restart at the current time."""
        self._restarts.append(datetime.now(UTC))

    def count(self) -> int:
        """Number of restarts inside the rolling window."""
        cutoff = datetime.now(UTC) - timedelta(seconds=self.window_s)
        return sum(1 for r in self._restarts if r >= cutoff)

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

    # -----------------------------------------------------------------
    # Subprocess spawn / stop / restart (integration slice)
    # -----------------------------------------------------------------

    def spawn_worker(
        self,
        session_id: str,
        *,
        model: str,
        provider: str,
        cwd: str | None = None,
        max_iterations: int = 30,
        env: dict[str, str] | None = None,
    ) -> WorkerHandle:
        """Spawn ``python -m cothis.worker`` for ``session_id``.

        Blocks until the worker prints its ready line (``{"uri", "token"}``),
        then records a ``spawned`` lifecycle event and returns a handle.
        The caller (TUI) gets the WS URL + bearer token from the handle.

        ``cwd`` defaults to the current working directory; ``env`` is merged
        over ``os.environ`` (the worker inherits provider API keys this way).
        Raises ``RuntimeError`` if the worker exits before printing the
        ready line or the line is unparseable.
        """
        import json as _json
        import subprocess
        import sys

        argv = [
            sys.executable, "-m", "cothis.worker",
            "--session", session_id,
            "--model", model,
            "--provider", provider,
            "--max-iterations", str(max_iterations),
        ]
        run_env = {**os.environ, **(env or {})}
        proc = subprocess.Popen(  # noqa: S603 — argv is constructed, not shell-expanded
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd or os.getcwd(),
            env=run_env,
            text=True,
        )
        # ``readline`` returns "" at EOF; None is impossible on a pipe.
        ready_raw = proc.stdout.readline() if proc.stdout else ""
        if not ready_raw:
            stderr_tail = ""
            if proc.stderr:
                # Drain stderr so the error message reaches the caller.
                stderr_tail = proc.stderr.read()[-2000:]
            proc.wait(timeout=5)
            raise RuntimeError(
                f"worker for session {session_id!r} exited (code={proc.returncode}) "
                f"before ready line; stderr tail:\n{stderr_tail}"
            )
        try:
            ready = _json.loads(ready_raw)
        except _json.JSONDecodeError as exc:
            proc.kill()
            raise RuntimeError(
                f"worker ready line not JSON: {ready_raw!r}"
            ) from exc
        handle = WorkerHandle(
            session_id=session_id,
            pid=proc.pid,
            ws_url=ready["uri"],
            token=ready["token"],
            cwd=cwd or os.getcwd(),
            status="running",
            restart_count=self._counter_for(session_id).count(),
        )
        # Stash the Popen so ``stop_worker`` / ``restart_worker`` can reach it.
        # ``object`` avoids a dataclass field for a process object.
        setattr(handle, "_proc", proc)
        self._workers[session_id] = handle
        self.record_lifecycle("spawned", session_id, extra_meta={"pid": proc.pid})
        return handle

    def stop_worker(self, session_id: str, *, timeout: float = 10.0) -> None:
        """Terminate one worker (SIGTERM → wait → SIGKILL if needed).

        Records a ``stopped`` lifecycle event. Idempotent: a no-op if the
        session isn't tracked or the process already exited.
        """
        import subprocess

        handle = self._workers.get(session_id)
        if handle is None:
            return
        proc: subprocess.Popen[Any] | None = getattr(handle, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()  # SIGTERM
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()  # SIGKILL
                proc.wait(timeout=timeout)
        handle.status = "stopped"
        self.record_lifecycle("stopped", session_id)

    def close(self) -> None:
        """Close the supervisor DB connection."""
        self._conn.close()
