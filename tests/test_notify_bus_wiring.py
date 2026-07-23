"""Tests for ``Agent._execute_tool`` notify-bus emission (#224).

The bus is gated by ``COTHIS_NOTIFY_BUS``. When unset, ``Agent._bus``
is ``None`` and tool behavior is byte-for-byte identical to today.
When set, every tool dispatch emits ``started`` + (``completed`` or
``failed``) rows into ``notify_events`` via the session storage
connection (#223).

Direct-call tests drive ``_execute_tool`` without going through the
LLM — the bus emission is synchronous w.r.t. the dispatch, so no
streaming plumbing is needed.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _make_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any]:
    """Build an Agent + Session pair for testing.

    Agent uses a mocked AnyLLM (no real provider calls). Session uses
    a temp db under ``tmp_path``; ``flush_sync=True`` so writes are
    visible immediately.
    """
    import any_llm

    from cothis.agent import Agent
    from cothis.session import Session

    monkeypatch.setattr(
        any_llm.AnyLLM,
        "create",
        staticmethod(lambda *a, **kw: MagicMock()),
    )
    agent = Agent(
        model="x", provider="openrouter", tools=[], max_iterations=5,
    )
    db_path = tmp_path / "sessions" / "session.db"
    session = Session.new(db_path, cwd=tmp_path, model="x", flush_sync=True)
    agent.attach_session(session)
    return agent, session


def _notify_rows(session: Any) -> list[dict[str, Any]]:
    """Read all notify_events rows for the session, ordered by seq.

    Returns ``[]`` when the table doesn't exist (bus was never
    initialised under a flag-off scenario).
    """
    conn = session._storage._conn  # type: ignore[union-attr]
    try:
        cur = conn.execute(
            "SELECT seq, topic, event_type, session_id, meta, payload_pointer "
            "FROM notify_events ORDER BY seq"
        )
    except Exception:  # noqa: BLE001 — sqlite3.OperationalError on missing table
        return []
    rows = cur.fetchall()
    import json
    return [
        {
            "seq": r[0],
            "topic": r[1],
            "event_type": r[2],
            "session_id": r[3],
            "meta": json.loads(r[4]) if r[4] else None,
            "payload_pointer": r[5],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------
# Feature-flag off — zero diff
# ---------------------------------------------------------------------


def test_execute_tool_no_bus_when_flag_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """COTHIS_NOTIFY_BUS unset → Agent._bus is None → no notify_events."""
    monkeypatch.delenv("COTHIS_NOTIFY_BUS", raising=False)

    def echo(**kw: Any) -> str:
        return "ok"

    agent, session = _make_agent(tmp_path, monkeypatch)
    agent._tool_map["echo"] = echo

    is_error, output = asyncio.run(
        agent._execute_tool(
            {"type": "tool_use", "id": "tu_1", "name": "echo", "input": {}}
        )
    )
    assert not is_error
    assert output == "ok"
    assert agent._bus is None
    rows = _notify_rows(session)
    # notify_events table may not even exist when bus was never created.
    # If it does exist (Session.new could theoretically init it), it's empty.
    assert rows == []


# ---------------------------------------------------------------------
# Feature-flag on — happy path
# ---------------------------------------------------------------------


def test_execute_tool_emits_started_and_completed_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag set + successful tool → started + completed rows."""

    def echo(**kw: Any) -> str:
        return "ok"

    monkeypatch.setenv("COTHIS_NOTIFY_BUS", "1")
    agent, session = _make_agent(tmp_path, monkeypatch)
    agent._tool_map["echo"] = echo

    is_error, _ = asyncio.run(
        agent._execute_tool(
            {"type": "tool_use", "id": "tu_happy", "name": "echo", "input": {}}
        )
    )
    assert not is_error
    assert agent._bus is not None

    rows = _notify_rows(session)
    assert len(rows) == 2
    assert rows[0]["event_type"] == "started"
    assert rows[1]["event_type"] == "completed"

    for r in rows:
        assert r["topic"] == "tool_call"
        assert r["session_id"] == session.session_id
        assert r["meta"]["tool"] == "echo"
        assert r["meta"]["call_id"] == "tu_happy"

    # completed row carries duration + ok=True
    assert "duration_ms" in rows[1]["meta"]
    assert rows[1]["meta"]["ok"] is True
    # payload_pointer references the session + tool call
    assert rows[1]["payload_pointer"] == (
        f"session:{session.session_id}:tool:tu_happy"
    )


# ---------------------------------------------------------------------
# Feature-flag on — failure path
# ---------------------------------------------------------------------


def test_execute_tool_emits_started_and_failed_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag set + raising tool → started + failed rows, no completed."""

    def boom(**kw: Any) -> Any:
        raise RuntimeError("kaboom")

    monkeypatch.setenv("COTHIS_NOTIFY_BUS", "1")
    agent, session = _make_agent(tmp_path, monkeypatch)
    agent._tool_map["boom"] = boom

    is_error, output = asyncio.run(
        agent._execute_tool(
            {"type": "tool_use", "id": "tu_fail", "name": "boom", "input": {}}
        )
    )
    assert is_error
    assert "kaboom" in output

    rows = _notify_rows(session)
    types = [r["event_type"] for r in rows]
    assert types == ["started", "failed"], types
    assert rows[-1]["meta"]["ok"] is False
    assert rows[-1]["meta"]["tool"] == "boom"
    assert rows[-1]["meta"]["call_id"] == "tu_fail"


# ---------------------------------------------------------------------
# tool_call_id matches the tool_use["id"] the model sent
# ---------------------------------------------------------------------


def test_execute_tool_meta_call_id_matches_tool_use_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever id the model sends must land in meta.call_id verbatim."""

    def echo(**kw: Any) -> str:
        return "ok"

    monkeypatch.setenv("COTHIS_NOTIFY_BUS", "1")
    agent, session = _make_agent(tmp_path, monkeypatch)
    agent._tool_map["echo"] = echo

    asyncio.run(
        agent._execute_tool(
            {
                "type": "tool_use",
                "id": "toolu_abcDEF123",
                "name": "echo",
                "input": {},
            }
        )
    )
    rows = _notify_rows(session)
    for r in rows:
        assert r["meta"]["call_id"] == "toolu_abcDEF123"


# ---------------------------------------------------------------------
# Payload pointer format
# ---------------------------------------------------------------------


def test_execute_tool_payload_pointer_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """payload_pointer uses session:<sid>:tool:<call_id> addressing."""

    def echo(**kw: Any) -> str:
        return "result bytes"

    monkeypatch.setenv("COTHIS_NOTIFY_BUS", "1")
    agent, session = _make_agent(tmp_path, monkeypatch)
    agent._tool_map["echo"] = echo

    asyncio.run(
        agent._execute_tool(
            {"type": "tool_use", "id": "tu_pp", "name": "echo", "input": {}}
        )
    )
    rows = _notify_rows(session)
    completed = next(r for r in rows if r["event_type"] == "completed")
    assert completed["payload_pointer"] == (
        f"session:{session.session_id}:tool:tu_pp"
    )


# ---------------------------------------------------------------------
# Events fire for async tools too (covers MCP-style tools)
# ---------------------------------------------------------------------


def test_execute_tool_emits_for_async_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async tool bodies also get started/completed events."""

    async def aecho(**kw: Any) -> str:
        return "async ok"

    monkeypatch.setenv("COTHIS_NOTIFY_BUS", "1")
    agent, session = _make_agent(tmp_path, monkeypatch)
    agent._tool_map["aecho"] = aecho

    is_error, _ = asyncio.run(
        agent._execute_tool(
            {"type": "tool_use", "id": "tu_async", "name": "aecho", "input": {}}
        )
    )
    assert not is_error
    rows = _notify_rows(session)
    assert [r["event_type"] for r in rows] == ["started", "completed"]
