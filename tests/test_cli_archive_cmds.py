"""Tests for the ``cothis archive`` CLI subcommands (#85 follow-up #110).

Integration tests against a real (temp) SQLite db: drive sessions
through ``Session``, then invoke the CLI via ``typer.testing.CliRunner``
to verify the four archive subcommands behave per #85's contract:

- ``cothis archive`` (default ``all``) runs ``run_archival_pass`` and
  reports the count (or "no sessions to archive").
- ``cothis archive <id>`` archives one session, printing confirmation.
  Idempotent on re-run (INSERT OR REPLACE).
- ``cothis archive restore <id>`` promotes the session back; not-found
  in index → ``BadParameter`` + exit 1.
- ``cothis archive compress <file>`` gzips the named file under
  ``archive_dir``; path-escape (``..``) and missing ``.db`` suffix are
  rejected with ``BadParameter``.
"""

from __future__ import annotations

import gzip
import sqlite3
from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from cothis.cli import app
from cothis.session import Session
from cothis.session.archive import ArchiveIndex
from cothis.session.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


runner = CliRunner()


def _user_text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _output(result: Any) -> str:
    """CliRunner result with stdout + stderr concatenated.

    ``typer.BadParameter`` messages are wrapped by rich's panel across
    lines (so exact-phrase matches fail); this helper exists so tests
    can match shorter fragments without restating the concat.
    """
    return result.stdout + (result.stderr or "")


def _seed_session(
    db_path: Path, cwd: Path, *, texts: list[str], model: str = "m",
) -> str:
    s = Session.new(db_path, cwd=cwd, model=model, flush_sync=True)
    sid = s.session_id
    for i, t in enumerate(texts):
        role = "user" if i % 2 == 0 else "assistant"
        s.append_message(role, [_user_text(t)])
    s.close()
    return sid


def _set_updated_at(db_path: Path, sid: str, updated_at: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE sessions SET updated_at=? WHERE id=?", (updated_at, sid)
        )
        conn.commit()
    finally:
        conn.close()


def _clear_archive_state(db_path: Path) -> None:
    """Drop ``archive_state.last_run`` so the next pass isn't throttled."""
    if not db_path.is_file():
        return
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM archive_state WHERE key='last_run'")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------
# cothis archive (default "all")
# ---------------------------------------------------------------------


def test_archive_all_no_idle_sessions_reports_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "session.db"
    _seed_session(db_path, tmp_path, texts=["fresh"])
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _clear_archive_state(db_path)

    result = runner.invoke(app, ["archive"])
    assert result.exit_code == 0
    assert "no sessions to archive" in result.stdout


def test_archive_all_moves_idle_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "session.db"
    old_sid = _seed_session(db_path, tmp_path, texts=["old"])
    _set_updated_at(db_path, old_sid, "2026-04-13T00:00:00+00:00")
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _clear_archive_state(db_path)

    result = runner.invoke(app, ["archive"])
    assert result.exit_code == 0
    assert "archived 1 session" in result.stdout

    hot = Storage(db_path)
    try:
        assert hot.load_session(old_sid) is None
    finally:
        hot.close()
    idx = ArchiveIndex(db_path.parent / "archive" / "index.json")
    assert idx.get(old_sid) is not None


# ---------------------------------------------------------------------
# cothis archive <id>
# ---------------------------------------------------------------------


def test_archive_one_session_by_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["alpha"])
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["archive", sid])
    assert result.exit_code == 0
    assert f"archived session {sid}" in result.stdout

    hot = Storage(db_path)
    try:
        assert hot.load_session(sid) is None
    finally:
        hot.close()
    idx = ArchiveIndex(db_path.parent / "archive" / "index.json")
    assert idx.get(sid) is not None


def test_archive_unknown_id_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cothis archive <unknown-id>`` raises BadParameter, not false success.

    Previously the CLI unconditionally printed ``archived session <id>``
    even when ``archive_session`` returned ``None`` because the id wasn't
    in the hot DB. #121 — surface the no-op as an error so a typo doesn't
    masquerade as success.
    """
    db_path = tmp_path / "session.db"
    _seed_session(db_path, tmp_path, texts=["seed"])  # init hot db
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    missing_sid = "0" * 32
    result = runner.invoke(app, ["archive", missing_sid])
    assert result.exit_code != 0
    combined = _output(result)
    assert "not found" in combined
    # Suggests the likely-fix paths (typo, missed restore, wrong scope).
    assert "history" in combined or "restore" in combined


def test_archive_one_session_is_idempotent_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-archiving a session that's already in cold is safe (INSERT OR REPLACE).

    Uses ``cothis archive restore`` (``promote_session``) to move the
    row back to hot — the cold DB still has its copy (promote is a
    copy, not a move), so the second ``archive`` re-runs the INSERT
    OR REPLACE path naturally without hand-rolled SQL.
    """
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["once"])
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    first = runner.invoke(app, ["archive", sid])
    assert first.exit_code == 0

    # Restore (promote) — hot row back, cold still has its copy.
    restore = runner.invoke(app, ["archive", "restore", sid])
    assert restore.exit_code == 0

    # Re-archive: INSERT OR REPLACE into cold hits the existing row.
    second = runner.invoke(app, ["archive", sid])
    assert second.exit_code == 0
    assert f"archived session {sid}" in second.stdout


# ---------------------------------------------------------------------
# cothis archive restore <id>
# ---------------------------------------------------------------------


def test_archive_restore_brings_session_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["archived"])
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    # Seed-then-archive via the CLI itself.
    archive_result = runner.invoke(app, ["archive", sid])
    assert archive_result.exit_code == 0

    restore = runner.invoke(app, ["archive", "restore", sid])
    assert restore.exit_code == 0
    assert f"restored session {sid}" in restore.stdout

    hot = Storage(db_path)
    try:
        assert hot.load_session(sid) is not None
    finally:
        hot.close()
    idx = ArchiveIndex(db_path.parent / "archive" / "index.json")
    assert idx.get(sid) is None


def test_archive_restore_unknown_id_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "session.db"
    _seed_session(db_path, tmp_path, texts=["seed"])
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    missing_sid = "0" * 32
    result = runner.invoke(app, ["archive", "restore", missing_sid])
    assert result.exit_code != 0
    # typer.BadParameter writes to stderr; the panel wraps long
    # messages across lines, so match the key fragment.
    combined = _output(result)
    assert "not found" in combined
    assert "archive index" in combined


def test_archive_restore_without_target_is_bad_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "session.db"
    _seed_session(db_path, tmp_path, texts=["seed"])
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["archive", "restore"])
    assert result.exit_code != 0
    combined = _output(result)
    assert "requires a session id" in combined


# ---------------------------------------------------------------------
# cothis archive compress <file>
# ---------------------------------------------------------------------


def test_archive_compress_gzips_cold_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "session.db"
    sid = _seed_session(db_path, tmp_path, texts=["cold"])
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["archive", sid])  # move to cold
    archive_dir = db_path.parent / "archive"
    cold_files = list(archive_dir.glob("*.db"))
    assert len(cold_files) == 1
    cold_name = cold_files[0].name

    result = runner.invoke(app, ["archive", "compress", cold_name])
    assert result.exit_code == 0
    assert "compressed to" in result.stdout

    gz = archive_dir / f"{cold_name}.gz"
    assert gz.is_file()
    # Verify it's a real gzip and contents are a SQLite DB.
    with gzip.open(gz, "rb") as f:
        header = f.read(16)
    assert header.startswith(b"SQLite format 3")


def test_archive_compress_rejects_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "session.db"
    _seed_session(db_path, tmp_path, texts=["seed"])
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    # Drop a real .db file outside archive_dir to prove the reject
    # is by path analysis, not by file existence.
    outside = tmp_path / "outside.db"
    outside.write_bytes(b"SQLite format 3\x00")

    result = runner.invoke(app, ["archive", "compress", "../outside.db"])
    assert result.exit_code != 0
    combined = _output(result)
    assert "must be inside" in combined


def test_archive_compress_rejects_non_db_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "session.db"
    _seed_session(db_path, tmp_path, texts=["seed"])
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["archive", "compress", "README.md"])
    assert result.exit_code != 0
    combined = _output(result)
    assert "must end in .db" in combined


def test_archive_compress_missing_file_is_bad_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "session.db"
    _seed_session(db_path, tmp_path, texts=["seed"])
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["archive", "compress", "2025-01.db"])
    assert result.exit_code != 0
    combined = _output(result)
    assert "no such file" in combined
