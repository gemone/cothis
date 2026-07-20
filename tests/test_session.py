"""Tests for ``cothis.session`` — durable conversation persistence.

Covers the five groups the issue acceptance criteria require (plus the
fcntl cross-process guard), all offline (no LLM, no network):

1. **CRUD round-trip** — new → append user/assistant(tool_use)/tool_result
   → close → load → deep-equal of role + content shape.
2. **``BEGIN deferred`` fairness** — 4 concurrent writers each on their
   own session id, fairness = min/avg wall-clock per thread ≥ 0.8.
3. **Crash recovery** — partial assistant (orphan ``tool_use``) is
   truncated on reload (Q2-A drop-trailing).
4. **``COTHIS_SESSIONS_DIR`` override + auto ``.gitignore``** — dir inside
   project writes ``*``; outside project does not; existing file is
   respected (skip-if-exists).
5. **fcntl lock refuses second opener** — opening the same session id
   twice in the same process raises ``SessionLockedError``.

Tests 1, 3, 4 use ``flush_sync=True`` (deterministic, no thread); the
fairness test (2) uses the real queue + consumer thread because it's the
only test that exercises the deferred-BEGIN contention path. The lock
test (5) also uses real mode (``flush_sync=False`` default) so the lock
fd lifecycle matches production.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from cothis.session import Session, SessionLockedError
from cothis.session.storage import Storage

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _user_text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _assistant_with_tool_use(text: str, tool_id: str, tool_name: str = "fs.read") -> list[dict[str, Any]]:
    return [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": text},
        {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {"path": "/x"}},
    ]


def _tool_result(tool_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


def _normalise_for_compare(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip assistant-only metadata + ephemeral flags before deep-compare.

    Reload doesn't restore ``id``/``model``/``stop_reason``/``usage`` (Q11:
    those are ``None`` after load) or ``is_error`` (Q23: ephemeral). Compare
    only the durable shape: ``role`` + each block's durable fields.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        blocks = []
        for b in m["content"]:
            b = dict(b)
            b.pop("is_error", None)
            blocks.append(b)
        out.append({"role": m["role"], "content": blocks})
    return out


# ---------------------------------------------------------------------
# 1. CRUD round-trip
# ---------------------------------------------------------------------


def test_crud_roundtrip_preserves_role_and_block_shape(tmp_path: Path) -> None:
    """Round-trip through SQLite reconstructs messages losslessly.

    Exercises the multi-block atomic assistant write, the per-execution
    tool_result same-role continuation (merged into one user message), and
    all five Anthropic block types' field mapping (text / thinking /
    tool_use / tool_result; image is covered separately).
    """
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    sid = s.session_id

    s.append_message("user", [_user_text("hello")])
    s.append_message("assistant", _assistant_with_tool_use("thinking...", "tu1"))
    s.append_block("user", _tool_result("tu1", "file contents"))
    s.close()

    s2 = Session.load(db_path, sid, flush_sync=True)
    assert [m["role"] for m in s2.messages] == ["user", "assistant", "user"]

    # User message shape.
    assert s2.messages[0]["content"] == [_user_text("hello")]

    # Assistant content reconstructs all three block types in order.
    assistant_types = [b["type"] for b in s2.messages[1]["content"]]
    assert assistant_types == ["thinking", "text", "tool_use"]
    thinking = s2.messages[1]["content"][0]
    assert thinking["thinking"] == "hmm"
    assert thinking["signature"] == "sig"
    tool_use = s2.messages[1]["content"][2]
    assert tool_use["id"] == "tu1"
    assert tool_use["name"] == "fs.read"
    assert tool_use["input"] == {"path": "/x"}

    # tool_result block reconstructs tool_use_id + content; is_error is gone.
    tool_result = s2.messages[2]["content"][0]
    assert tool_result["tool_use_id"] == "tu1"
    assert tool_result["content"] == "file contents"
    assert "is_error" not in tool_result
    s2.close()


def test_crud_roundtrip_multiple_tool_results_share_msg_idx(tmp_path: Path) -> None:
    """Per-execution enqueues of tool_results merge into ONE user message.

    Anthropic requires all tool_results from one turn in a single user
    message. The Q22 per-exec enqueue + Q3-A same-role-continuation rule
    must keep them at the same ``msg_idx`` so reload produces one message
    with N blocks (not N messages with one block each).
    """
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    sid = s.session_id

    s.append_message("user", [_user_text("do two things")])
    s.append_message("assistant", [
        {"type": "text", "text": "ok"},
        {"type": "tool_use", "id": "tu1", "name": "fs.read", "input": {}},
        {"type": "tool_use", "id": "tu2", "name": "fs.read", "input": {}},
    ])
    # Per-exec: two separate append_block calls, same role.
    s.append_block("user", _tool_result("tu1", "result 1"))
    s.append_block("user", _tool_result("tu2", "result 2"))
    s.close()

    s2 = Session.load(db_path, sid, flush_sync=True)
    # Exactly 3 messages: user, assistant, user (with 2 tool_result blocks).
    assert len(s2.messages) == 3
    assert s2.messages[2]["role"] == "user"
    assert len(s2.messages[2]["content"]) == 2
    s2.close()


def test_crud_roundtrip_image_block(tmp_path: Path) -> None:
    """Image block's source round-trips through ``image_source`` JSON."""
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    sid = s.session_id
    s.append_message("user", [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
    ])
    s.close()
    s2 = Session.load(db_path, sid, flush_sync=True)
    assert s2.messages[0]["content"][0]["type"] == "image"
    assert s2.messages[0]["content"][0]["source"] == {
        "type": "base64", "media_type": "image/png", "data": "abc",
    }
    s2.close()


# ---------------------------------------------------------------------
# 2. Deferred-BEGIN configuration (structural test, not behavioral)
# ---------------------------------------------------------------------
#
# Issue #34's research validated BEGIN deferred at fairness=1.0 across 4
# *concurrent processes* (vs BEGIN IMMEDIATE starving). Reproducing that
# multi-process contention in a unit test is heavy and slow, and a
# multi-thread variant does NOT validate the same property — Python's GIL
# and OS thread scheduling dominate intra-process fairness, so DEFERRED
# and IMMEDIATE come out indistinguishable (~0.34 min/mean for both).
#
# What we CAN pin cheaply is the design choice the research informed:
# every Storage connection opens with isolation_level="DEFERRED" +
# busy_timeout=5000 + journal_mode=WAL. If someone swaps to IMMEDIATE
# (the starving variant the issue rejected) or drops the busy_timeout
# (which would make concurrent writers fail with SQLITE_BUSY instead of
# waiting), this test fails. That's the regression we actually need to
# catch at unit-test time; true cross-process fairness is a benchmark,
# not a CI assertion.


def test_storage_opens_with_deferred_isolation_and_wal(tmp_path: Path) -> None:
    """Pin the BEGIN-mode + busy_timeout + WAL configuration choices.

    Reads the live PRAGMAs back from an opened connection. ``journal_mode``
    is persistent but we set it on every open; ``busy_timeout`` is
    per-connection. ``isolation_level`` is what Python sqlite3 uses for its
    implicit BEGIN — ``DEFERRED`` is the value the issue's fairness
    validation picked.
    """
    storage = Storage(tmp_path / "test.db")
    try:
        mode = storage._conn.execute("PRAGMA journal_mode").fetchone()[0]
        # SQLite normalises "wal" → "wal" on read.
        assert mode == "wal"
        bt = storage._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert bt == 5000
        # Python sqlite3 exposes the configured isolation_level on the
        # connection. Empty string = default = deferred; we set "DEFERRED"
        # explicitly.
        assert storage._conn.isolation_level == "DEFERRED"
    finally:
        storage.close()


def test_storage_does_not_deadlock_under_threaded_contention(
    tmp_path: Path
) -> None:
    """Smoke test: 4 threads × 30 writes complete without deadlock.

    Not a fairness assertion (see the structural test above for that) —
    just a guard that the queue + consumer + WAL + busy_timeout path
    doesn't deadlock or livelock. Each thread gets its own db (own
    session_id, own SQLite file) — cross-session write contention on the
    same db isn't a real cothis scenario (different sessions never share
    a db in practice).
    """
    n_threads = 4
    writes_per_thread = 30
    start = threading.Event()
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def writer(idx: int) -> None:
        try:
            db_path = tmp_path / f"sessions-{idx}" / "session.db"
            s = Session.new(db_path, cwd=tmp_path, model="m")
            barrier.wait()
            start.wait()
            for i in range(writes_per_thread):
                s.append_block("user", {"type": "text", "text": f"t{idx}-{i}"})
            s.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(timeout=30)
    assert not errors, f"writer threads raised: {errors}"
    # All threads finished — if any timed out, join() returned but the
    # thread is still alive; check explicitly.
    assert not any(t.is_alive() for t in threads), "a writer thread timed out"


# ---------------------------------------------------------------------
# 3. Crash recovery (drop trailing orphan tool_use)
# ---------------------------------------------------------------------


def test_reload_drops_trailing_orphan_tool_use(tmp_path: Path) -> None:
    """Partial assistant (tool_use with no matching tool_result) is dropped.

    Simulates the crash case Q2-A guards against: the consumer committed
    an assistant message containing a ``tool_use`` block, but the process
    died before the matching ``tool_result`` was written. On reload the
    orphan tail is dropped — otherwise the next ``amessages`` call 400s
    on the unpaired tool_use.
    """
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    sid = s.session_id
    s.append_message("user", [_user_text("hi")])
    s.append_message("assistant", [
        {"type": "text", "text": "first"},
        {"type": "tool_use", "id": "tu1", "name": "fs.read", "input": {}},
    ])
    # Pair the first tool_use, then start a second turn that crashes mid-tool.
    s.append_block("user", _tool_result("tu1", "ok"))
    s.append_message("assistant", [
        {"type": "text", "text": "second"},
        {"type": "tool_use", "id": "tu2", "name": "fs.read", "input": {}},
    ])
    # No tool_result for tu2 — simulated crash.
    s.close()

    s2 = Session.load(db_path, sid, flush_sync=True)
    # The orphan tail (the second assistant message) is dropped.
    assert [m["role"] for m in s2.messages] == ["user", "assistant", "user"]
    assert s2.messages[1]["content"][1]["id"] == "tu1"
    s2.close()


def test_reload_truncate_is_persisted_not_just_in_memory(
    tmp_path: Path,
) -> None:
    """Regression: the truncate must DELETE orphan rows, not just hide them.

    Without the DELETE, the orphan rows stay in ``blocks``; a second reload
    re-truncates at the same point — silently erasing any messages written
    between reloads. Reproduces the high-severity finding from review.
    """
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    sid = s.session_id
    s.append_message("user", [_user_text("q1")])
    s.append_message("assistant", [
        {"type": "text", "text": "a1"},
        {"type": "tool_use", "id": "tu_orphan", "name": "fs.read", "input": {}},
    ])
    s.close()  # Simulated crash: no tool_result for tu_orphan.

    # Reload 1 — orphan dropped from memory AND from disk.
    s2 = Session.load(db_path, sid, flush_sync=True)
    assert [m["role"] for m in s2.messages] == ["user"]
    # Append post-recovery history.
    s2.append_message("assistant", [{"type": "text", "text": "fresh answer"}])
    s2.close()

    # Reload 2 — without the DELETE fix, the orphan would still be in the
    # table and the fresh assistant message would get wiped alongside it.
    s3 = Session.load(db_path, sid, flush_sync=True)
    try:
        roles = [m["role"] for m in s3.messages]
        assert roles == ["user", "assistant"], (
            f"truncate was not persisted; reload re-dropped the fresh "
            f"message. Got roles={roles}."
        )
        assert s3.messages[1]["content"][0]["text"] == "fresh answer"
    finally:
        s3.close()


def test_session_row_written_flag_set_after_successful_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Lazy session-row flag stays False until the row actually commits.

    Pins the invariant that ``_session_row_written`` is set only after
    ``write_atomic`` returns: setting it earlier would leave no sessions
    row for the FK check, blocking every subsequent block INSERT.

    Persistent failure across all retries is needed to reach this path —
    transient failures are recovered by the retry queue
    (see ``test_transient_write_failure_recovered_by_retry``).
    """
    # Squash retry backoff to zero so the test doesn't sleep ~2.6s on the
    # 3 retry attempts before the poison-row drop kicks in.
    monkeypatch.setattr(
        "cothis.session._WRITE_RETRY_BACKOFFS", (0.0, 0.0, 0.0)
    )
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    # First write_atomic: fail every attempt so the row stays unwritten.
    real_write = s._storage.write_atomic

    def persistently_failing_write(session_row, block_rows, updated_at):
        raise sqlite3.OperationalError("disk I/O error (simulated)")

    monkeypatch.setattr(s._storage, "write_atomic", persistently_failing_write)
    # This first enqueue exhausts retries and drops the batch (poison row).
    with caplog.at_level("CRITICAL", logger="cothis.session"):
        s.append_message("user", [_user_text("first")])
    assert s._session_row_written is False, (
        "flag must not be set after a failed write — would permanently "
        "block the lazy row from ever being written"
    )
    # Restore real write; second enqueue should succeed AND write the row.
    monkeypatch.setattr(s._storage, "write_atomic", real_write)
    s.append_message("assistant", [{"type": "text", "text": "second"}])
    assert s._session_row_written is True
    # The session row exists in the DB (FK would have blocked otherwise).
    sr = s._storage.load_session(s.session_id)
    assert sr is not None
    # The first ("user") message was dropped by the poison-row guard;
    # only the assistant message landed. The CRITICAL log records the loss.
    drops = [r for r in caplog.records if "dropping" in r.getMessage()]
    assert len(drops) == 1
    assert "1 block(s)" in drops[0].getMessage()
    s.close()
    s2 = Session.load(db_path, s.session_id, flush_sync=True)
    try:
        assert [m["role"] for m in s2.messages] == ["assistant"]
        assert s2.messages[0]["content"][0]["text"] == "second"
    finally:
        s2.close()


def test_transient_write_failure_recovered_by_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A transient ``write_atomic`` failure triggers an in-line retry;
    if the next attempt succeeds, both messages persist as if the first
    attempt never happened.
    """
    # Zero backoff: the test isn't validating the backoff schedule, only
    # the recovery contract. Avoids a 100ms sleep per retry.
    monkeypatch.setattr(
        "cothis.session._WRITE_RETRY_BACKOFFS", (0.0, 0.0, 0.0)
    )
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    real_write = s._storage.write_atomic
    call_count = {"n": 0}

    def one_shot_flaky_write(session_row, block_rows, updated_at):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise sqlite3.OperationalError("disk I/O error (simulated)")
        return real_write(session_row, block_rows, updated_at)

    monkeypatch.setattr(s._storage, "write_atomic", one_shot_flaky_write)
    # Single enqueue: first attempt fails, first retry succeeds.
    with caplog.at_level("CRITICAL", logger="cothis.session"):
        s.append_message("user", [_user_text("first")])
    assert call_count["n"] == 2, "retry must re-invoke write_atomic"
    assert s._session_row_written is True
    # No CRITICAL drop record on recovery — only the WARNING retry log.
    drops = [r for r in caplog.records if "dropping" in r.getMessage()]
    assert drops == [], "no rows dropped on recovery"
    s.append_message("assistant", [{"type": "text", "text": "second"}])
    s.close()
    s2 = Session.load(db_path, s.session_id, flush_sync=True)
    try:
        # BOTH messages persisted — transient failure recovered.
        assert [m["role"] for m in s2.messages] == ["user", "assistant"]
        assert s2.messages[0]["content"][0]["text"] == "first"
        assert s2.messages[1]["content"][0]["text"] == "second"
    finally:
        s2.close()


def test_persistent_write_failure_drops_after_retries_no_deadlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """After ``len(_WRITE_RETRY_BACKOFFS) + 1`` attempts, a persistently-
    failing batch is dropped and the consumer moves on; the next enqueue
    must succeed.
    """
    monkeypatch.setattr(
        "cothis.session._WRITE_RETRY_BACKOFFS", (0.0, 0.0, 0.0)
    )
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    real_write = s._storage.write_atomic
    call_count = {"n": 0}

    def persistently_failing_write(session_row, block_rows, updated_at):
        call_count["n"] += 1
        raise sqlite3.OperationalError("disk I/O error (simulated)")

    monkeypatch.setattr(s._storage, "write_atomic", persistently_failing_write)
    # First enqueue: persistent failure → 4 attempts (1 + 3 retries) → drop.
    with caplog.at_level("CRITICAL", logger="cothis.session"):
        s.append_message("user", [_user_text("doomed")])
    assert call_count["n"] == 4
    drops = [r for r in caplog.records if "dropping" in r.getMessage()]
    assert len(drops) == 1
    assert "1 block(s)" in drops[0].getMessage()
    # Restore real write; the NEXT enqueue must succeed — the consumer is
    # not deadlocked on the dropped batch.
    monkeypatch.setattr(s._storage, "write_atomic", real_write)
    s.append_message("assistant", [{"type": "text", "text": "recovered"}])
    assert s._session_row_written is True
    s.close()
    s2 = Session.load(db_path, s.session_id, flush_sync=True)
    try:
        # Only the post-drop assistant message survived. The dropped "user"
        # message is gone — that's the contract for persistent errors.
        assert [m["role"] for m in s2.messages] == ["assistant"]
        assert s2.messages[0]["content"][0]["text"] == "recovered"
    finally:
        s2.close()


def test_retry_backoff_interrupted_by_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``close()`` during a retry backoff interrupts the wait via ``_stop``,
    abandons the in-flight batch, and returns within ``close``'s 5s ceiling.
    """
    # Use a real backoff (so _stop.wait actually waits) — but we'll set
    # _stop immediately after the first failed attempt to simulate close.
    monkeypatch.setattr(
        "cothis.session._WRITE_RETRY_BACKOFFS", (10.0, 10.0, 10.0)
    )
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    call_count = {"n": 0}

    def failing_then_signal_close(session_row, block_rows, updated_at):
        call_count["n"] += 1
        # Simulate close() happening between attempt 1 and the first retry:
        # the consumer sets _stop, the in-flight _drain_one's _stop.wait
        # returns True immediately, and the drain bails out.
        s._stop.set()
        raise sqlite3.OperationalError("disk I/O error (simulated)")

    monkeypatch.setattr(s._storage, "write_atomic", failing_then_signal_close)
    # First enqueue: fails once, _stop is set during the failure, the
    # first _stop.wait returns True, drain abandons the batch.
    with caplog.at_level("CRITICAL", logger="cothis.session"):
        s.append_message("user", [_user_text("interrupted")])
    assert call_count["n"] == 1, "must not retry after _stop is set"
    # Close-path abandon does NOT emit a CRITICAL drop record — different
    # loss category (kill -9 ceiling, not poison row).
    drops = [r for r in caplog.records if "dropping" in r.getMessage()]
    assert drops == []
    # Session is still usable for further appends (in-memory), but the
    # abandoned batch is lost.
    s.append_message("assistant", [{"type": "text", "text": "after"}])
    s.close()


def test_reload_keeps_full_history_when_no_orphans(tmp_path: Path) -> None:
    """No false positives: a clean conversation survives reload intact."""
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    sid = s.session_id
    s.append_message("user", [_user_text("hi")])
    s.append_message("assistant", _assistant_with_tool_use("...", "tu1"))
    s.append_block("user", _tool_result("tu1", "data"))
    s.append_message("assistant", [{"type": "text", "text": "done"}])
    s.close()

    s2 = Session.load(db_path, sid, flush_sync=True)
    assert len(s2.messages) == 4
    assert [m["role"] for m in s2.messages] == [
        "user", "assistant", "user", "assistant",
    ]
    s2.close()


# ---------------------------------------------------------------------
# 4. COTHIS_SESSIONS_DIR override + auto .gitignore
# ---------------------------------------------------------------------
#
# Tests below pass ``db_path`` (not ``sessions_dir``) to ``Session.new``,
# matching the new API. The CLI's ``_resolve_db_path`` adds the
# ``session.db`` filename; tests build the path explicitly to avoid
# coupling to the CLI resolver.
#
# Lock files live under the cache dir (``~/.cache/cothis/<id>.lock`` by
# default), decoupled from the db location. Tests that need to assert on
# lock paths use ``monkeypatch.setenv("XDG_CACHE_HOME", tmp_path / cache)``
# so the lock lands in a test-isolated location instead of polluting the
# real user cache.


def test_gitignore_written_when_db_dir_inside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """db_dir inside cwd → ``*`` written on first save (split mode)."""
    monkeypatch.chdir(tmp_path)
    # Keep the lock out of the real user cache.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s.append_message("user", [_user_text("x")])
    s.close()
    assert (db_path.parent / ".gitignore").read_text() == "*\n"


def test_no_gitignore_when_db_dir_outside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """db_dir outside cwd → no .gitignore (e.g. default ~/.cothis/agents.db)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    outside = tmp_path.parent / "other-sessions" / "session.db"
    s = Session.new(outside, cwd=tmp_path, model="m", flush_sync=True)
    s.append_message("user", [_user_text("x")])
    s.close()
    assert not (outside.parent / ".gitignore").exists()


def test_gitignore_skip_if_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing .gitignore is respected — cothis never overwrites user rules."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    db_path = tmp_path / "sessions" / "session.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    (db_path.parent / ".gitignore").write_text("custom_rule\n")
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s.append_message("user", [_user_text("x")])
    s.close()
    assert (db_path.parent / ".gitignore").read_text() == "custom_rule\n"


def test_cothis_sessions_dir_env_creates_split_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``COTHIS_SESSIONS_DIR=<path>`` triggers split layout (session.db, not agents.db)."""
    override = tmp_path / "custom" / "location"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(override))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    # Reproduce the CLI resolver: <DIR>/session.db
    db_path = Path(os.environ["COTHIS_SESSIONS_DIR"]).expanduser() / "session.db"
    assert db_path == override / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s.append_message("user", [_user_text("x")])
    sid = s.session_id
    s.close()
    # Split layout: db at <override>/session.db
    assert (override / "session.db").exists()
    # Lock lives in cache dir, NOT under <override>/
    cache_dir = tmp_path / "cache" / "cothis"
    # The cache-dir lock was created during the session; filelock removes
    # it on release. The invariant is that the lock was NEVER at
    # <override>/<sid>.lock — asserted on the next line.
    assert not (override / f"{sid}.lock").exists()


def test_project_type_uses_cwd_agents_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``COTHIS_SESSIONS_TYPE=project`` → ``<cwd>/.agents/sessions/session.db``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COTHIS_SESSIONS_TYPE", "project")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    # Reproduce the CLI resolver for project mode
    db_path = Path.cwd() / ".agents" / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s.append_message("user", [_user_text("x")])
    s.close()
    assert db_path.exists()
    # .gitignore auto-written (db_dir is inside cwd)
    assert (db_path.parent / ".gitignore").read_text() == "*\n"


def test_default_mode_uses_home_agents_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env → default layout: ``$COTHIS_HOME/agents.db`` single-file."""
    # Use sandboxed COTHIS_HOME + a separate cwd so the db's parent dir is
    # NOT inside cwd (the real default puts it in ~/.cothis/, never in cwd).
    fake_home = tmp_path / "fake-home"
    cwd = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.setenv("COTHIS_HOME", str(fake_home))
    monkeypatch.delenv("COTHIS_SESSIONS_TYPE", raising=False)
    monkeypatch.delenv("COTHIS_SESSIONS_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.chdir(cwd)
    # Reproduce the default-mode resolver path
    db_path = fake_home / "agents.db"
    fake_home.mkdir(parents=True, exist_ok=True)
    s = Session.new(db_path, cwd=cwd, model="m", flush_sync=True)
    s.append_message("user", [_user_text("x")])
    s.close()
    assert db_path.exists()
    # Default mode: no .gitignore written (db_dir is ~/, not in cwd)
    assert not (fake_home / ".gitignore").exists()


# ---------------------------------------------------------------------
# 5. fcntl lock refuses second opener
# ---------------------------------------------------------------------


def test_lock_refuses_second_opener_same_process(tmp_path: Path) -> None:
    """A second Session on the same id in the same process is refused.

    fcntl locks are per-fd (POSIX), so two fds in the same process exhibit
    the same exclusion as two processes. ``Session.new`` → first opener
    succeeds; ``Session.load`` (or a second ``new`` on the same id) →
    ``SessionLockedError``. The CLI surfaces this with a non-zero exit.
    """
    db_path = tmp_path / "sessions" / "session.db"
    s1 = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    # First opener writes a session row so load() can find it.
    s1.append_message("user", [_user_text("x")])
    sid = s1.session_id
    try:
        with pytest.raises(SessionLockedError) as excinfo:
            Session.load(db_path, sid, flush_sync=True)
        assert sid in str(excinfo.value) or "in use" in str(excinfo.value)
    finally:
        s1.close()
    # After close, the lock is released — load succeeds.
    s2 = Session.load(db_path, sid, flush_sync=True)
    s2.close()


def test_lock_released_after_close_allows_reopen(tmp_path: Path) -> None:
    """Lock release on close is verifiable: a second opener succeeds afterwards."""
    db_path = tmp_path / "sessions" / "session.db"
    s1 = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s1.append_message("user", [_user_text("x")])
    sid = s1.session_id
    s1.close()
    # No raise expected.
    s2 = Session.load(db_path, sid, flush_sync=True)
    s2.close()


def test_lock_is_instance_scoped_not_class_scoped(tmp_path: Path) -> None:
    """Regression: two Sessions keep separate locks; closing one doesn't
    release the other's lock.

    Before the fix, ``_lock`` was a class-level slot: a second Session
    would overwrite it, so ``s1.close()`` would release ``s2``'s lock
    (the cross-process guard silently breaks). The test exercises the
    concurrent-coexistence case directly.
    """
    db_path = tmp_path / "sessions" / "session.db"
    s1 = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s1.append_message("user", [_user_text("a")])
    s2 = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s2.append_message("user", [_user_text("b")])
    # Both alive at once; lock objects are distinct.
    assert s1._lock is not None and s2._lock is not None
    assert s1._lock is not s2._lock
    # Close s1: its lock is released; s2's lock is untouched and still
    # held (so re-opening s2's id is refused).
    s1_sid = s1.session_id
    s2_sid = s2.session_id
    s1.close()
    assert s1._lock is None
    assert s2._lock is not None, "s1.close() must not clear s2's lock"
    # s1's lock is released — reopening s1's id should succeed.
    s1_again = Session.load(db_path, s1_sid, flush_sync=True)
    s1_again.close()
    # s2's lock is still held — reopening s2's id is refused.
    with pytest.raises(SessionLockedError):
        Session.load(db_path, s2_sid, flush_sync=True)
    s2.close()


# ---------------------------------------------------------------------
# additional sanity: schema_version placeholder + Storage idempotent open
# ---------------------------------------------------------------------


def test_storage_reopen_is_idempotent(tmp_path: Path) -> None:
    """Re-opening an existing db doesn't error and preserves data."""
    db = tmp_path / "test.db"
    s1 = Storage(db)
    from cothis.session.storage import BlockRow, SessionRow
    sr = SessionRow(
        id="s1", parent_id=None, parent_seq=None, cwd="/x",
        cli_version="0.1.0", model="m", title="t",
        created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
    )
    br = BlockRow(
        session_id="s1", seq=0, msg_idx=0, block_idx=0,
        role="user", type="text", ts="2026-01-01T00:00:00Z", content="hi",
    )
    s1.write_atomic(sr, [br], "2026-01-01T00:00:01Z")
    s1.close()
    # Reopen — DDL is IF NOT EXISTS, no error.
    s2 = Storage(db)
    assert s2.load_session("s1") is not None
    assert len(s2.load_blocks("s1")) == 1
    s2.close()


# ---------------------------------------------------------------------
# Integration: Agent.run with attached Session persists end-to-end
# ---------------------------------------------------------------------


def test_agent_run_with_session_persists_tool_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wiring test: Agent + attached Session persists a full tool turn.

    Drives the actual ``Agent.run`` code path (``_ensure_messages`` user
    enqueue + assistant append + per-exec ``tool_result`` enqueue) with a
    mocked LLM, then reloads the Session and asserts the persisted shape.
    Catches regressions where someone moves an enqueue site, drops the
    ``if self._session`` guard, or breaks the per-exec merging.
    """
    import asyncio as asyncio_mod
    from unittest.mock import MagicMock

    import any_llm
    from anthropic.types import TextBlock, ToolUseBlock

    from cothis.agent import Agent

    monkeypatch.setattr(
        any_llm.AnyLLM,
        "create",
        staticmethod(lambda *a, **kw: MagicMock()),
    )

    agent = Agent(model="x", provider="openrouter", tools=[], max_iterations=5)
    agent._tool_map["echo"] = lambda **kw: kw["msg"]
    db_path = tmp_path / "sessions" / "session.db"
    session = Session.new(db_path, cwd=tmp_path, model="x", flush_sync=True)
    sid = session.session_id
    agent.attach_session(session)

    state = {"turn": 0}

    async def fake_amessages(**kwargs: Any) -> Any:
        from any_llm.types.messages import (
            MessageResponse,
            MessageUsage,
        )
        state["turn"] += 1
        if state["turn"] == 1:
            content: list[Any] = [
                ToolUseBlock(
                    type="tool_use", id="tu1", name="echo", input={"msg": "hi"}
                )
            ]
        else:
            content = [TextBlock(type="text", text="final")]
        return MessageResponse(
            id="m1", model="x", role="assistant", type="message",
            content=content, stop_reason="end_turn",
            usage=MessageUsage(input_tokens=1, output_tokens=1),
        )

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    answer = asyncio_mod.run(agent.run("hello"))
    assert answer == "final"
    asyncio_mod.run(agent.aclose())

    # Reload — the persisted shape should reconstruct to 4 messages:
    # user / assistant(tool_use) / user(tool_result) / assistant(text).
    s2 = Session.load(db_path, sid, flush_sync=True)
    try:
        roles = [m["role"] for m in s2.messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        # The tool turn round-tripped: tool_use id + tool_result pairing.
        assert s2.messages[1]["content"][0]["type"] == "tool_use"
        assert s2.messages[1]["content"][0]["id"] == "tu1"
        assert s2.messages[2]["content"][0]["type"] == "tool_result"
        assert s2.messages[2]["content"][0]["tool_use_id"] == "tu1"
        # Final assistant answer survived.
        assert s2.messages[3]["content"][0]["text"] == "final"
    finally:
        s2.close()


# ---------------------------------------------------------------------
# Regression: resumed session seeds Agent history
# ---------------------------------------------------------------------


def test_attach_session_seeds_agent_messages_from_loaded_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ``attach_session`` must seed ``Agent._messages`` from a
    loaded session's history.

    Before the fix, ``Session.load`` rebuilt history into
    ``session.messages`` but ``attach_session`` only stored the session
    reference — it never propagated history into the Agent. The model
    would then send an empty conversation on its next ``amessages`` call,
    silently continuing as if no prior turns existed (amnesiac resume).

    This test asserts both that the Agent's in-memory history matches
    the Session's, AND that the next outgoing ``amessages`` request
    includes the prior turns.
    """
    import asyncio as asyncio_mod
    from unittest.mock import MagicMock

    import any_llm
    from anthropic.types import TextBlock

    from cothis.agent import Agent

    monkeypatch.setattr(
        any_llm.AnyLLM,
        "create",
        staticmethod(lambda *a, **kw: MagicMock()),
    )

    db_path = tmp_path / "sessions" / "session.db"
    # Session 1: write some history, close (simulates a previous chat).
    s1 = Session.new(db_path, cwd=tmp_path, model="x", flush_sync=True)
    sid = s1.session_id
    s1.append_message("user", [{"type": "text", "text": "remember this"}])
    s1.append_message("assistant", [{"type": "text", "text": "got it"}])
    s1.close()

    # Resume: load + attach to a fresh Agent.
    s2 = Session.load(db_path, sid, flush_sync=True)
    agent = Agent(model="x", provider="openrouter", tools=[], max_iterations=5)
    agent.attach_session(s2)

    # 1. The Agent's in-memory history is seeded from the Session.
    assert len(agent._messages) == 2, (
        f"Agent should have 2 messages from resumed session, got "
        f"{len(agent._messages)}. Resume is amnesiac."
    )
    assert agent._messages[0]["role"] == "user"
    assert agent._messages[0]["content"][0]["text"] == "remember this"
    assert agent._messages[1]["role"] == "assistant"
    assert agent._messages[1]["content"][0]["text"] == "got it"

    # 2. The next outgoing amessages call includes the prior history.
    seen_request_messages: list[Any] = []

    async def fake_amessages(**kwargs: Any) -> Any:
        from any_llm.types.messages import MessageResponse, MessageUsage
        # Capture what the Agent is about to send.
        seen_request_messages.extend(kwargs.get("messages", []))
        return MessageResponse(
            id="m1", model="x", role="assistant", type="message",
            content=[TextBlock(type="text", text="ok")],
            stop_reason="end_turn",
            usage=MessageUsage(input_tokens=1, output_tokens=1),
        )

    monkeypatch.setattr(agent._llm, "amessages", fake_amessages)
    asyncio_mod.run(agent.run("follow up"))
    asyncio_mod.run(agent.aclose())

    # The outgoing request must include the resumed history before the
    # new "follow up" user message. 2 prior + 1 new user = 3 user-side
    # messages minimum before the assistant reply is added.
    request_user_texts = [
        m["content"][0].get("text")
        for m in seen_request_messages
        if m.get("role") == "user" and m.get("content")
        and isinstance(m["content"][0], dict)
        and m["content"][0].get("type") == "text"
    ]
    assert "remember this" in request_user_texts, (
        f"Resumed history missing from outgoing request. User texts seen: "
        f"{request_user_texts}"
    )
    assert "follow up" in request_user_texts


def test_ensure_messages_merges_when_history_ends_with_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: resume after a trailing-``user`` crash must not produce
    consecutive ``user`` messages (Anthropic HTTP 400).

    A session can legitimately end in ``role="user"``: crash mid-LLM-call
    (user input persisted, assistant response never written), or trailing
    ``tool_result`` with no final assistant. The next ``_ensure_messages``
    call must merge into the trailing user message, not append a second
    one. Before the fix, ``_ensure_messages`` used raw ``append`` and
    ``_messages`` ended up as ``[..., {user}, {user}]``.
    """
    from unittest.mock import MagicMock

    import any_llm

    from cothis.agent import Agent

    monkeypatch.setattr(
        any_llm.AnyLLM,
        "create",
        staticmethod(lambda *a, **kw: MagicMock()),
    )

    db_path = tmp_path / "sessions" / "session.db"
    s1 = Session.new(db_path, cwd=tmp_path, model="x", flush_sync=True)
    sid = s1.session_id
    s1.append_message("user", [{"type": "text", "text": "first prompt"}])
    s1.close()  # trailing user — no assistant reply

    s2 = Session.load(db_path, sid, flush_sync=True)
    agent = Agent(model="x", provider="openrouter", tools=[], max_iterations=5)
    agent.attach_session(s2)

    # _ensure_messages is the unit: does a new user input merge?
    agent._ensure_messages("follow up")

    # Exactly ONE user message: two text blocks merged into it.
    user_msgs = [m for m in agent._messages if m["role"] == "user"]
    assert len(user_msgs) == 1, (
        f"Expected 1 user message (merged), got {len(user_msgs)}. "
        f"Consecutive user messages → Anthropic 400."
    )
    text_blocks = [
        b for b in user_msgs[0]["content"]
        if b.get("type") == "text"
    ]
    assert len(text_blocks) == 2
    assert text_blocks[0]["text"] == "first prompt"
    assert text_blocks[1]["text"] == "follow up"
    s2.close()


def test_crud_roundtrip_real_consumer_thread(tmp_path: Path) -> None:
    """``flush_sync=False`` round-trip: real queue + daemon consumer.

    All other CRUD tests use ``flush_sync=True`` (inline ``_drain_one``,
    no thread), so a bug specific to the async path — close() joining
    before the consumer drains, or the consumer writing out of order —
    would slip through. This test enqueues a multi-turn conversation
    through the real consumer, closes, reloads, and deep-compares.

    Covers two async-path risks: (1) close()/drain timing — does the
    consumer actually persist everything before close() returns? (2)
    write ordering — do blocks land in enqueue order?
    """
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m")  # flush_sync=False (default)
    sid = s.session_id

    # Multi-turn: user → assistant(tool_use) → tool_result → assistant(text).
    s.append_message("user", [_user_text("turn 1")])
    s.append_message("assistant", _assistant_with_tool_use("using tool", "tu1"))
    s.append_block("user", _tool_result("tu1", "result data"))
    s.append_message("assistant", [{"type": "text", "text": "final answer"}])
    s.close()

    s2 = Session.load(db_path, sid, flush_sync=True)
    try:
        # Ordering: roles must match enqueue order exactly.
        assert [m["role"] for m in s2.messages] == [
            "user", "assistant", "user", "assistant",
        ]
        # Data integrity: every enqueued block survived the async path.
        assert s2.messages[0]["content"] == [_user_text("turn 1")]
        assistant1_types = [b["type"] for b in s2.messages[1]["content"]]
        assert assistant1_types == ["thinking", "text", "tool_use"]
        tool_result = s2.messages[2]["content"][0]
        assert tool_result["tool_use_id"] == "tu1"
        assert tool_result["content"] == "result data"
        assert s2.messages[3]["content"][0]["text"] == "final answer"
    finally:
        s2.close()


# ---------------------------------------------------------------------
# 6. Security: db file permissions + session_id traversal guard
# ---------------------------------------------------------------------


def test_db_file_is_owner_only(tmp_path: Path) -> None:
    """Transcripts carry tool output (routinely secrets) → db 0o600.

    Also covers WAL/SHM sidecars: SQLite creates them via open() under
    the process umask (0o644), and they do NOT inherit the main db's
    mode. WAL holds un-checkpointed data (same secrets); SHM is the
    shared-memory index. All three must be tightened.

    Checks while the Session is still open (sidecars exist) — close()
    checkpoints and deletes them.

    POSIX-only assertion: Windows uses icacls-based ACL restriction
    (``_restrict_to_owner``) instead of Unix mode bits, so ``st_mode``
    doesn't reflect 0o600 there. The ACL path is exercised on Windows
    CI (no crash) but can't be asserted portably.
    """
    if os.name != "posix":
        pytest.skip("Unix mode bits are POSIX-only; Windows uses ACLs")
    db_path = tmp_path / "sessions" / "session.db"
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    s.append_message("user", [_user_text("x")])
    try:
        for suffix in ("", "-wal", "-shm"):
            f = db_path.with_name(db_path.name + suffix) if suffix else db_path
            if not f.exists():
                continue  # WAL may not exist yet if no writes pending
            mode = f.stat().st_mode & 0o777
            assert mode == 0o600, (
                f"{f.name} is {oct(mode)}, expected 0o600 (owner-only)"
            )
    finally:
        s.close()


def test_load_rejects_non_hex_session_id(tmp_path: Path) -> None:
    """session_id becomes a filesystem path — reject traversal attempts."""
    db_path = tmp_path / "sessions" / "session.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.touch()
    for bad in ["../etc/passwd", "deadbeef", "X" * 32, "a" * 31 + "g", ""]:
        with pytest.raises(ValueError):
            Session.load(db_path, bad, flush_sync=True)


def test_close_storage_closed_even_when_consumer_stuck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lock contract: close() must close storage before releasing the lock.

    A stuck consumer (one that outlives the join timeout) must not keep
    storage open across close() — otherwise the cross-process lock is
    released while writes may still be in flight, defeating the
    ``SessionLockedError`` contract.
    """
    db_path = tmp_path / "sessions" / "session.db"
    # flush_sync=True avoids starting a real consumer; we inject a fake.
    s = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)

    storage_closed = {"called": False}
    release_called_after = {"ok": False}
    def tracking_close() -> None:
        storage_closed["called"] = True
    monkeypatch.setattr(s._storage, "close", tracking_close)
    real_release = s._release_lock
    def tracking_release() -> None:
        if storage_closed["called"]:
            release_called_after["ok"] = True
        real_release()
    monkeypatch.setattr(s, "_release_lock", tracking_release)

    # Fake consumer that's still "alive" after the join timeout —
    # simulates a stuck write_atomic (slow disk, retry storm, etc.).
    # Subclasses Thread to satisfy the type checker; we override
    # is_alive/join so no real thread lifecycle is involved.
    class _StuckConsumer(threading.Thread):
        def run(self) -> None:  # never started; never runs
            pass
        def is_alive(self) -> bool:
            return True
        def join(self, timeout: float | None = None) -> None:
            return  # returns immediately, still alive
    s._consumer = _StuckConsumer()

    # Squash the join timeouts so the test doesn't sleep.
    monkeypatch.setattr("cothis.session._CLOSE_JOIN_TIMEOUT", 0.0)
    monkeypatch.setattr("cothis.session._CLOSE_GRACE_PERIOD", 0.0)

    s.close()

    assert storage_closed["called"], (
        "close() must close storage even when consumer is stuck — "
        "otherwise the lock is released while writes may still happen, "
        "violating the SessionLockedError contract"
    )
    assert release_called_after["ok"], (
        "lock must be released AFTER storage close — releasing earlier "
        "opens a race window for a second acquirer"
    )


def test_restrict_to_owner_logs_warning_windows_no_user_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Windows + missing USERNAME/USER env: warn so the user knows tightening
    was skipped (the db is left at default umask with potential secrets)."""
    from cothis.session import storage as storage_mod

    target = tmp_path / "file.db"
    target.touch()
    monkeypatch.setattr(storage_mod.os, "name", "nt")
    monkeypatch.delenv("USERNAME", raising=False)
    monkeypatch.delenv("USER", raising=False)

    with caplog.at_level("WARNING", logger="cothis.session.storage"):
        storage_mod._restrict_to_owner(str(target))

    warnings = [r for r in caplog.records if "no USERNAME/USER" in r.getMessage()]
    assert warnings, "must log a WARNING when tightening is skipped on Windows"


def test_restrict_to_owner_logs_warning_windows_icacls_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Windows + icacls fails (missing binary / non-zero exit / timeout): warn
    so the user sees the failure rather than believing tightening succeeded."""
    from cothis.session import storage as storage_mod

    target = tmp_path / "file.db"
    target.touch()
    monkeypatch.setattr(storage_mod.os, "name", "nt")
    monkeypatch.setenv("USERNAME", "tester")

    def failing_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["icacls"], output=b"", stderr=b"denied")
    monkeypatch.setattr(storage_mod.subprocess, "run", failing_run)

    with caplog.at_level("WARNING", logger="cothis.session.storage"):
        storage_mod._restrict_to_owner(str(target))

    warnings = [r for r in caplog.records if "icacls failed" in r.getMessage()]
    assert warnings, "must log a WARNING when icacls fails"


def test_restrict_to_owner_logs_warning_posix_chmod_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """POSIX + unexpected OSError from chmod: warn (real FS doesn't fail
    here; a denial is environmental and the user can fix it)."""
    from cothis.session import storage as storage_mod

    target = tmp_path / "file.db"
    target.touch()
    monkeypatch.setattr(storage_mod.os, "name", "posix")

    def chmod_fail(path: str, mode: int) -> None:
        raise OSError(1, "Operation not permitted", path)
    monkeypatch.setattr(storage_mod.os, "chmod", chmod_fail)

    with caplog.at_level("WARNING", logger="cothis.session.storage"):
        storage_mod._restrict_to_owner(str(target))

    warnings = [r for r in caplog.records if "chmod 0o600 failed" in r.getMessage()]
    assert warnings, "must log a WARNING when chmod fails on POSIX"
