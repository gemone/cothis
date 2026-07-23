"""Tests for #188: synthetic pre-activation pair is stripped from request.

The synthetic ``load_skill`` tool_use crashes thinking-mode providers
(DeepSeek, OpenAI o-series) because it lacks ``reasoning_content``.
The fix: strip synthetic pairs from ``_request_messages`` and deliver
the skill body via the system prompt instead. The pair stays in
``_messages`` for persistence + resume rebuild (#71).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from cothis.agent import (
    _request_messages,
    _synthetic_skill_pair,
)

if TYPE_CHECKING:
    from pathlib import Path


_PREACT_PREFIX = "preact_"


def _make_skill(name: str) -> Any:
    from pathlib import Path as _Path

    from cothis.skills import Skill
    return Skill(
        name=name, description=f"{name} d", body=f"{name} b",
        source=_Path(f"/tmp/{name}/SKILL.md"), deactivation="delete",
    )


# ---------------------------------------------------------------------
# _request_messages strips synthetic pairs
# ---------------------------------------------------------------------


def test_request_messages_strips_synthetic_tool_use() -> None:
    """Assistant message with only a ``preact_`` tool_use → skipped."""
    tu, tr = _synthetic_skill_pair("python", "body")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [tu]},
        {"role": "user", "content": [tr]},
    ]
    out = _request_messages(messages)
    # Only the first user message survives.
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert out[0]["content"][0]["text"] == "hi"


def test_request_messages_strips_synthetic_tool_result() -> None:
    """User message with only a ``preact_``-referenced tool_result → skipped.

    The two surviving user messages are consecutive (the synthetic
    assistant + user pair between them was stripped). The merge pass
    (#205) combines them into one message — Anthropic's alternation
    invariant is preserved.
    """
    tu, tr = _synthetic_skill_pair("python", "body")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [tu]},
        {"role": "user", "content": [tr]},
        {"role": "user", "content": [{"type": "text", "text": "follow-up"}]},
    ]
    out = _request_messages(messages)
    # Consecutive user messages merged into one.
    assert len(out) == 1
    texts = [b["text"] for b in out[0]["content"] if isinstance(b, dict) and b.get("type") == "text"]
    assert "hi" in texts
    assert "follow-up" in texts


def test_request_messages_keeps_real_tool_use() -> None:
    """Non-preact tool_use messages are kept (regression check)."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_real1", "name": "fs_read", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_real1", "content": "data"},
            ],
        },
    ]
    out = _request_messages(messages)
    assert len(out) == 3  # nothing stripped


def test_request_messages_keeps_mixed_message_with_non_preact_block() -> None:
    """Assistant with a real block + a preact tool_use → kept (real wins)."""
    tu, _ = _synthetic_skill_pair("python", "body")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "real thinking"},
                tu,
            ],
        },
    ]
    out = _request_messages(messages)
    # The real text block keeps the message alive.
    assert len(out) == 2


# ---------------------------------------------------------------------
# Agent._run_preactivation augments system prompt
# ---------------------------------------------------------------------


def _patched_agent(preactivate: list[str]) -> Any:
    from types import SimpleNamespace
    return SimpleNamespace(
        _messages=[],
        _session=None,
        system="persona",
        preactivate_skills=list(preactivate),
        _preactivation_done=False,
    )


def test_run_preactivation_appends_skill_body_to_system() -> None:
    """Skill body lands in the system prompt (via ``self.system`` list)."""
    from cothis.agent import Agent
    agent = _patched_agent(preactivate=["python"])
    agent._messages.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    with patch("cothis.skills.discover_skills", return_value=[_make_skill("python")]):
        Agent._run_preactivation(agent)
    # system is now a list including the skill body text.
    assert isinstance(agent.system, list)
    texts = [b.get("text", "") for b in agent.system if isinstance(b, dict)]
    assert any("python b" in t for t in texts)
    assert any("<skill_content" in t for t in texts)


def test_run_preactivation_still_injects_pair_into_messages() -> None:
    """The synthetic pair is in ``_messages`` (for persistence/resume)."""
    from cothis.agent import Agent
    agent = _patched_agent(preactivate=["python"])
    agent._messages.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    with patch("cothis.skills.discover_skills", return_value=[_make_skill("python")]):
        Agent._run_preactivation(agent)
    # user + assistant(tool_use) + user(tool_result) = 3 messages.
    assert len(agent._messages) == 3


def test_request_messages_uses_augmented_system() -> None:
    """End-to-end: after preactivation, the request has the skill body in system
    (not in any tool_use shape that would crash DeepSeek)."""
    from cothis.agent import Agent, _system_param
    agent = _patched_agent(preactivate=["python"])
    agent._messages.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    with patch("cothis.skills.discover_skills", return_value=[_make_skill("python")]):
        Agent._run_preactivation(agent)

    # System prompt augmented.
    system_param = _system_param(agent.system)
    assert system_param is not None
    system_texts = [b.get("text", "") for b in system_param if isinstance(b, dict)]
    assert any("python b" in t for t in system_texts)

    # Request messages don't include the synthetic pair.
    request = _request_messages(agent._messages)
    types_in_request = [
        b.get("type")
        for m in request
        for b in m["content"]
        if isinstance(b, dict)
    ]
    assert "tool_use" not in types_in_request
    assert "tool_result" not in types_in_request


# ---------------------------------------------------------------------
# Resume coherence — synthetic pair still in storage for #71 rebuild
# ---------------------------------------------------------------------


def test_synthetic_pair_persisted_for_resume(tmp_path: Path) -> None:
    """Even with the system-prompt path, the pair is persisted so resume
    rebuild (#71) re-activates the skill."""
    from cothis.agent import Agent
    from cothis.session import Session
    from cothis.session.storage import Storage

    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    agent = _patched_agent(preactivate=["python"])
    agent._session = s
    agent._messages.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    with patch("cothis.skills.discover_skills", return_value=[_make_skill("python")]):
        Agent._run_preactivation(agent)
    sid = s._session_id
    s.close()

    storage = Storage(tmp_path / "db.db")
    rows = storage.load_blocks(sid)
    storage.close()
    # Tool_use + tool_result rows persisted.
    types = {r.type for r in rows}
    assert "tool_use" in types
    assert "tool_result" in types
    skills = {r.skill for r in rows if r.skill}
    assert "python" in skills
