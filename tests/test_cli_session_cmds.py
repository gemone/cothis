"""Tests for the ``cothis history`` / ``cothis delete`` CLI commands.

These are integration tests against a real (temp) SQLite db: they
construct sessions via the ``Session`` API, then invoke the CLI through
typer's testing runner to verify listing, fork, and delete behaviour.
``chat --resume`` is covered indirectly through ``Session.load``'s
visibility filter (tested in ``test_session.py``) — the CLI plumbing
is one line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from cothis.cli import app
from cothis.session import Session
from cothis.session.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path


def _make_session(
    db_path: Path,
    cwd: Path,
    *,
    model: str = "m",
    texts: list[str] | None = None,
) -> str:
    """Create a session, append ``texts`` as user/assistant alternation, return id.

    Consecutive same-role messages are merged by ``Session.append_message``
    (Anthropic alternation invariant), so the caller must alternate roles
    to get distinct message rows. This helper alternates automatically:
    even-indexed ``texts`` become user, odd-indexed become assistant.
    """
    s = Session.new(db_path, cwd=cwd, model=model, flush_sync=True)
    sid = s.session_id
    for i, t in enumerate(texts or []):
        role = "user" if i % 2 == 0 else "assistant"
        s.append_message(role, [{"type": "text", "text": t}])
    s.close()
    return sid


def test_history_lists_visible_sessions(tmp_path: Path, monkeypatch: Any) -> None:
    """``cothis history`` shows sessions whose cwd is current-or-ancestor."""
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    sid = _make_session(db_path, tmp_path, texts=["hello world"])

    runner = CliRunner()
    result = runner.invoke(app, ["history"])
    assert result.exit_code == 0, result.output
    assert sid in result.output
    assert "hello world" in result.output


def test_history_hides_sessions_in_other_cwd(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Sessions scoped to a sibling project are not visible."""
    db_path = tmp_path / "session.db"
    project_a = tmp_path / "a"
    project_a.mkdir()
    project_b = tmp_path / "b"
    project_b.mkdir()
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))

    sid_a = _make_session(db_path, project_a, texts=["a-content"])
    sid_b = _make_session(db_path, project_b, texts=["b-content"])

    monkeypatch.chdir(project_a)
    runner = CliRunner()
    result = runner.invoke(app, ["history"])
    assert result.exit_code == 0
    assert sid_a in result.output
    assert sid_b not in result.output


def test_history_no_db_is_clean_no_error(tmp_path: Path, monkeypatch: Any) -> None:
    """No db file → ``no sessions database yet``, exit 0 (not an error)."""
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["history"])
    assert result.exit_code == 0
    assert "no sessions database" in result.output


def test_history_unknown_id_errors_cleanly(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``cothis history <unknown>`` exits non-zero with a clear message."""
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _make_session(db_path, tmp_path, texts=["seed"])  # ensure db exists

    runner = CliRunner()
    result = runner.invoke(app, ["history", "a" * 32])
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "usage" in result.output.lower()


def test_history_with_id_lists_messages(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``cothis history <id>`` prints each message with its role and a preview."""
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    sid = _make_session(db_path, tmp_path, texts=["first turn", "second turn"])

    runner = CliRunner()
    # Choose 'q' to skip the picker prompt.
    result = runner.invoke(app, ["history", sid], input="q\n")
    assert result.exit_code == 0, result.output
    assert "first turn" in result.output
    assert "second turn" in result.output


def test_history_with_id_fork_picker_creates_fork(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Choosing a message index forks at that point and reports the new id."""
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    sid = _make_session(db_path, tmp_path, texts=["msg-zero", "msg-one", "msg-two"])

    runner = CliRunner()
    result = runner.invoke(app, ["history", sid], input="1\n")
    assert result.exit_code == 0, result.output
    assert "forked as" in result.output
    fork_line = [ln for ln in result.output.splitlines() if "forked as" in ln][0]
    fork_id = fork_line.split("as")[1].split(";")[0].strip()
    fork_id = fork_id.replace("[cyan]", "").replace("[/cyan]", "").strip()
    assert len(fork_id) == 32, f"expected 32-hex fork id, got {fork_id!r}"

    # The fork records the parent link on the first persisted message.
    # The CLI's fork does NOT auto-write a message (lazy row strategy),
    # so we add one here to flush the link.
    forked = Session.load(db_path, fork_id, cwd=tmp_path, flush_sync=True)
    forked.append_message("user", [{"type": "text", "text": "fork msg"}])
    forked.close()

    storage = Storage(db_path)
    try:
        sr = storage.load_session(fork_id)
        assert sr is not None
        assert sr.parent_id == sid
    finally:
        storage.close()


def test_delete_removes_leaf_session(tmp_path: Path, monkeypatch: Any) -> None:
    """``cothis delete <id>`` removes a leaf session from the db."""
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    sid = _make_session(db_path, tmp_path, texts=["hi"])

    runner = CliRunner()
    result = runner.invoke(app, ["delete", sid])
    assert result.exit_code == 0, result.output
    assert "deleted" in result.output

    storage = Storage(db_path)
    try:
        assert storage.load_session(sid) is None
    finally:
        storage.close()


def test_delete_refuses_non_leaf(tmp_path: Path, monkeypatch: Any) -> None:
    """``cothis delete`` on a parent with a living fork refuses cleanly."""
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    parent_id = _make_session(db_path, tmp_path, texts=["parent"])

    ps = Storage(db_path)
    try:
        cap = max(r.seq for r in ps.load_blocks(parent_id))
    finally:
        ps.close()
    forked = Session.fork(
        db_path, parent_id, cap, cwd=tmp_path, model="m", flush_sync=True
    )
    forked.append_message("user", [{"type": "text", "text": "fork msg"}])
    forked.close()

    runner = CliRunner()
    result = runner.invoke(app, ["delete", parent_id])
    assert result.exit_code != 0
    assert "child" in result.output.lower() or "leaf" in result.output.lower()

    # Parent is still present.
    storage = Storage(db_path)
    try:
        assert storage.load_session(parent_id) is not None
    finally:
        storage.close()


def test_delete_unknown_id_errors_cleanly(tmp_path: Path, monkeypatch: Any) -> None:
    """``cothis delete <unknown>`` exits non-zero with a not-found message."""
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _make_session(db_path, tmp_path, texts=["seed"])

    runner = CliRunner()
    result = runner.invoke(app, ["delete", "a" * 32])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
