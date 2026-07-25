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
