"""Tests for the ``cothis search`` CLI command.

Integration tests against a real (temp) SQLite db: build sessions via the
``Session`` API, then invoke ``cothis search <query>`` through typer's
testing runner to verify the happy path, no-results path, missing-db path,
and ``--limit`` truncation. Mirrors ``test_cli_session_cmds.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from cothis.cli import app
from cothis.session import Session

if TYPE_CHECKING:
    from pathlib import Path


def _make_session(
    db_path: Path,
    cwd: Path,
    *,
    model: str = "m",
    texts: list[str] | None = None,
) -> str:
    """Create a session, append ``texts`` as user/assistant alternation, return id."""
    s = Session.new(db_path, cwd=cwd, model=model, flush_sync=True)
    sid = s.session_id
    for i, t in enumerate(texts or []):
        role = "user" if i % 2 == 0 else "assistant"
        s.append_message(role, [{"type": "text", "text": t}])
    s.close()
    return sid


def test_search_prints_matching_session_and_snippet(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``cothis search <term>`` prints the matching session id + snippet."""
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    sid = _make_session(db_path, tmp_path, texts=["deploy the alpha service"])

    runner = CliRunner()
    result = runner.invoke(app, ["search", "alpha"])
    assert result.exit_code == 0, result.output
    assert sid in result.output
    # The snippet highlights the matched term.
    assert "alpha" in result.output


def test_search_no_match_prints_no_matches(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A query that matches nothing prints ``no matches`` and exits 0."""
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _make_session(db_path, tmp_path, texts=["real content here"])

    runner = CliRunner()
    result = runner.invoke(app, ["search", "zzzznomatch"])
    assert result.exit_code == 0, result.output
    assert "no matches" in result.output


def test_search_no_db_is_clean_no_error(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """No db file → ``no sessions database yet``, exit 0 (not an error)."""
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["search", "anything"])
    assert result.exit_code == 0, result.output
    assert "no sessions database" in result.output


def test_search_limit_truncates_output(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``--limit`` (``-n``) caps the number of result lines printed."""
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    # Five sessions, all containing the shared term — five hits unbounded.
    for i in range(5):
        _make_session(db_path, tmp_path, texts=[f"common term number {i}"])

    runner = CliRunner()
    full = runner.invoke(app, ["search", "common"])
    assert full.exit_code == 0, full.output
    # Each hit prints one line; ``no matches`` adds none. Count non-empty
    # result lines that carry a session id (32-hex prefix).
    full_hit_lines = [
        ln for ln in full.output.splitlines()
        if ln and ln.split()[0].isascii() and len(ln.split()[0]) == 32
    ]
    assert len(full_hit_lines) == 5

    capped = runner.invoke(app, ["search", "common", "-n", "2"])
    assert capped.exit_code == 0, capped.output
    capped_hit_lines = [
        ln for ln in capped.output.splitlines()
        if ln and ln.split()[0].isascii() and len(ln.split()[0]) == 32
    ]
    assert len(capped_hit_lines) == 2


def test_search_malformed_query_prints_clean_message_not_traceback(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A malformed FTS5 query prints ``invalid query`` + exits 0, not a traceback.

    An unterminated quote (a likely paste error) makes ``storage.search``
    raise ``sqlite3.OperationalError``; the CLI catches it and prints a
    dim one-liner instead of letting a Python traceback surface.
    """
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("COTHIS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _make_session(db_path, tmp_path, texts=["real content here"])

    runner = CliRunner()
    result = runner.invoke(app, ["search", '"unterminated'])
    assert result.exit_code == 0, result.output
    assert "invalid query" in result.output
    # No Python traceback leaks to the user-facing surface.
    assert "Traceback" not in result.output
