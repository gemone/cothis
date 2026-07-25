"""Integration test for the ``cothis worker`` CLI entrypoint (#250 path a).

Spawns the worker as a real subprocess (the way the Supervisor will, once
``Supervisor.spawn_worker`` lands), reads the bind JSON line from stdout,
connects a real WS client, exercises ping/pong + shutdown, and asserts
clean process exit. No mocks — the whole point is end-to-end coverage of
the bind-handshake contract that ``Supervisor`` will depend on.

The CLI entrypoint itself is small (~30 lines in cli.py); the integration
test exercises the full surface:

  - ``cothis worker --session <id>`` finds the session in the DB
    resolved by ``COTHIS_SESSIONS_DIR`` (same env-driven path the rest
    of the CLI uses).
  - stdout receives exactly one JSON line ``{"uri": ..., "token": ...}``
    once the WS server binds; the line is flushed so a line-buffered
    reader on the supervisor side sees it without delay.
  - The WS handshake requires the bearer token; without it the upgrade
    fails with HTTP 401 (covered by ``test_session_worker.py``).
  - ``shutdown`` control message ends the accept loop and the process
    exits 0.

Path (a) of issue #250 lands the entrypoint; the follow-up integration
test for ``Supervisor.spawn_worker`` will reuse this subprocess contract.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
import websockets

from cothis.session import Session

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_worker_cli_emits_bind_json_and_serves_ws(tmp_path: Path) -> None:
    """AC #250 path (a): ``cothis worker`` emits bind JSON + serves WS.

    Drives the full lifecycle the Supervisor will use:

      1. Pre-create a session row the worker can load.
      2. Spawn ``python -m cothis.cli worker --session <id>`` with
         ``COTHIS_SESSIONS_DIR`` pointed at the temp dir.
      3. Read the bind JSON line from stdout (10s deadline).
      4. Connect a real WS client with the bearer token.
      5. ping → pong round-trip.
      6. Send ``shutdown`` → assert clean process exit (5s deadline).

    A regression in any of these (JSON shape, bearer enforcement, accept
    loop wiring, shutdown propagation) shows up as a hard test failure
    with the worker's stderr attached.
    """
    db_path = tmp_path / "session.db"
    session = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    # Session row is written lazily on the first message's drain — append
    # one user message so the row lands before close() and the subprocess
    # can load it by id.
    session.append_message("user", [{"type": "text", "text": "hi"}])
    sid = session.session_id
    session.close()

    env = {
        **os.environ,
        # Same env-driven DB resolution the rest of the CLI uses; isolates
        # the test from the user's ``$COTHIS_HOME``.
        "COTHIS_SESSIONS_DIR": str(tmp_path),
        # Agent construction validates the provider API key eagerly; CI
        # runners have no real key. The test only exercises bind +
        # ping/pong + shutdown — never an actual LLM call — so a dummy
        # key satisfies the constructor guard.
        "OPENROUTER_API_KEY": "test-dummy-not-used",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "cothis.cli", "worker", "--session", sid],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        # ``stdout=PIPE`` guarantees non-None; assert narrows for typer.
        assert proc.stdout is not None
        # Read the bind JSON line. ``readline`` blocks until the worker
        # prints + flushes; the 10s deadline covers Python startup + WS
        # bind on a slow CI runner.
        bind_line = await asyncio.wait_for(
            asyncio.to_thread(proc.stdout.readline),
            timeout=15.0,
        )
        if not bind_line:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise AssertionError(
                f"worker exited without emitting bind JSON; "
                f"returncode={proc.returncode}; stderr:\n{stderr}"
            )
        bind = json.loads(bind_line)
        assert "uri" in bind, f"bind JSON missing 'uri': {bind!r}"
        assert "token" in bind, f"bind JSON missing 'token': {bind!r}"
        uri = bind["uri"]
        token = bind["token"]
        assert uri.startswith("ws://127.0.0.1:"), (
            f"bind URI must be loopback WS; got {uri!r}"
        )
        assert len(token) >= 32, (
            f"bearer token must be >=32 chars (secrets.token_urlsafe(32)); "
            f"got len={len(token)}"
        )

        async with websockets.connect(
            uri, additional_headers={"Authorization": f"Bearer {token}"}
        ) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            assert json.loads(raw) == {"type": "pong"}, (
                f"expected pong; got {raw!r}"
            )

            await ws.send(json.dumps({"type": "shutdown"}))

        rc = await asyncio.wait_for(
            asyncio.to_thread(proc.wait),
            timeout=5.0,
        )
        assert rc == 0, (
            f"worker exited with {rc} after shutdown; "
            f"stderr:\n{proc.stderr.read() if proc.stderr else ''}"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass


@pytest.mark.asyncio
async def test_worker_cli_unknown_session_exits_nonzero(tmp_path: Path) -> None:
    """``--session <bogus>`` fails fast with a Typer error + non-zero exit.

    Guards the contract that the supervisor can rely on: a failed spawn
    (bad session id, missing DB) surfaces as a non-zero exit code on the
    subprocess, not as a hang or a WS bind with no underlying session.
    """
    env = {
        **os.environ,
        "COTHIS_SESSIONS_DIR": str(tmp_path),
        # Dummy key — see happy-path test for rationale. The unknown-session
        # path exits before any LLM call, but Agent construction still
        # validates the key eagerly.
        "OPENROUTER_API_KEY": "test-dummy-not-used",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "cothis.cli", "worker",
         "--session", "0" * 32],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        rc = await asyncio.wait_for(
            asyncio.to_thread(proc.wait),
            timeout=15.0,
        )
        assert rc != 0, (
            f"worker should exit non-zero on unknown session; got rc=0; "
            f"stdout:\n{proc.stdout.read() if proc.stdout else ''}"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
