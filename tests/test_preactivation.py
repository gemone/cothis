"""Tests for ``--skill`` pre-activation via synthetic load pair (#73).

Each pre-activated skill is materialised as a synthetic
``load_skill`` tool_use + tool_result pair, tagged
``_cothis_skill=name``, inserted after the first user message and
before the first LLM call. Resume rebuild (#71) treats the pair as a
normal load; ``deactivate_skill`` archives it like any other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from cothis.agent import _synthetic_skill_pair

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------
# Pair construction (pure function)
# ---------------------------------------------------------------------


def test_synthetic_pair_shape_for_load_skill() -> None:
    """Pair is (tool_use, tool_result); both tagged ``_cothis_skill``."""
    tu, tr = _synthetic_skill_pair("python", "<skill_content>body</skill_content>")
    assert tu["type"] == "tool_use"
    assert tu["name"] == "load_skill"
    assert tu["input"] == {"name": "python"}
    assert tu["_cothis_skill"] == "python"
    assert "id" in tu

    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == tu["id"]
    assert tr["content"] == "<skill_content>body</skill_content>"
    assert tr["_cothis_skill"] == "python"


def test_synthetic_pair_ids_match() -> None:
    """``tool_result.tool_use_id`` references the ``tool_use.id``."""
    tu, tr = _synthetic_skill_pair("python", "body")
    assert tr["tool_use_id"] == tu["id"]


def test_synthetic_pair_unique_ids_per_call() -> None:
    """Two pairs don't share ids."""
    tu1, _ = _synthetic_skill_pair("python", "b1")
    tu2, _ = _synthetic_skill_pair("python", "b2")
    assert tu1["id"] != tu2["id"]


def test_synthetic_pair_handles_empty_body() -> None:
    """Empty body still produces a valid pair."""
    tu, tr = _synthetic_skill_pair("x", "")
    assert tu["input"] == {"name": "x"}
    assert tr["content"] == ""


# ---------------------------------------------------------------------
# Agent._run_preactivation integration
# ---------------------------------------------------------------------


def _make_skill(name: str) -> Any:
    """Build a minimal Skill for tests."""
    from pathlib import Path as _Path

    from cothis.skills import Skill
    return Skill(
        name=name,
        description=f"{name} description",
        body=f"{name} body",
        source=_Path(f"/tmp/{name}/SKILL.md"),
        deactivation="delete",
    )


def _patched_agent(preactivate: list[str]) -> Any:
    """Build a minimal Agent-like object exposing only what
    ``_run_preactivation`` needs. Avoids pydantic's ``__init__``.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        _messages=[],
        _session=None,
        system="persona",
        preactivate_skills=list(preactivate),
        _preactivation_done=False,
    )


def test_run_preactivation_injects_after_first_user_message() -> None:
    """One pre-activated skill → synthetic pair appended after user msg."""
    from cothis.agent import Agent
    agent = _patched_agent(preactivate=["python"])
    agent._messages.append({"role": "user", "content": [{"type": "text", "text": "hello"}]})
    with patch("cothis.skills.discover_skills", return_value=[_make_skill("python")]):
        Agent._run_preactivation(agent)
    msgs = agent._messages
    assert len(msgs) == 3
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert agent._preactivation_done is True


def test_run_preactivation_alternation_for_multiple_skills() -> None:
    """Multiple skills → user/assistant/user/assistant/user."""
    from cothis.agent import Agent
    agent = _patched_agent(preactivate=["python", "bash"])
    agent._messages.append({"role": "user", "content": [{"type": "text", "text": "hello"}]})
    with patch(
        "cothis.skills.discover_skills",
        return_value=[_make_skill("python"), _make_skill("bash")],
    ):
        Agent._run_preactivation(agent)
    roles = [m["role"] for m in agent._messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


def test_run_preactivation_idempotent() -> None:
    """Second call is a no-op (the ``_preactivation_done`` flag short-circuits)."""
    from cothis.agent import Agent
    agent = _patched_agent(preactivate=["python"])
    agent._messages.append({"role": "user", "content": [{"type": "text", "text": "hello"}]})
    with patch("cothis.skills.discover_skills", return_value=[_make_skill("python")]):
        Agent._run_preactivation(agent)
    len_after_first = len(agent._messages)
    Agent._run_preactivation(agent)
    assert len(agent._messages) == len_after_first


def test_run_preactivation_empty_list_is_noop() -> None:
    """Empty ``preactivate_skills`` → no-op."""
    from cothis.agent import Agent
    agent = _patched_agent(preactivate=[])
    agent._messages.append({"role": "user", "content": [{"type": "text", "text": "hello"}]})
    Agent._run_preactivation(agent)
    assert len(agent._messages) == 1
    assert agent._preactivation_done is True


def test_run_preactivation_unknown_skill_raises() -> None:
    """Unknown skill name fails fast."""
    from cothis.agent import Agent
    agent = _patched_agent(preactivate=["nonexistent"])
    agent._messages.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    with patch("cothis.skills.discover_skills", return_value=[]):
        with pytest.raises(ValueError, match="nonexistent"):
            Agent._run_preactivation(agent)


def test_run_preactivation_activates_in_session(tmp_path: Path) -> None:
    """Pre-activation adds the skill to ``Session.active_skills``."""
    from cothis.agent import Agent
    from cothis.session import Session
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    agent = _patched_agent(preactivate=["python"])
    agent._session = s
    agent._messages.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    with patch("cothis.skills.discover_skills", return_value=[_make_skill("python")]):
        Agent._run_preactivation(agent)
    assert "python" in s.active_skills
    s.close()


def test_run_preactivation_persists_pair_to_storage(tmp_path: Path) -> None:
    """Synthetic pair is persisted so resume sees it."""
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
    skills_seen = {r.skill for r in rows if r.skill is not None}
    assert "python" in skills_seen


# ---------------------------------------------------------------------
# Resume coherence (end-to-end)
# ---------------------------------------------------------------------


def test_resume_rebuild_treats_synthetic_pair_as_normal_load(
    tmp_path: Path,
) -> None:
    """End-to-end: preactivate → close → resume → skill still active."""
    from cothis.agent import Agent
    from cothis.session import Session

    # Place SKILL.md so discover_skills finds python on reload.
    skill_dir = tmp_path / ".agents" / "skills" / "python"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: python\ndescription: d\n---\nbody\n", encoding="utf-8",
    )

    db_path = tmp_path / "db.db"
    s1 = Session.new(db_path, cwd=tmp_path, model="m", flush_sync=True)
    agent = _patched_agent(preactivate=["python"])
    agent._session = s1
    agent._messages.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    with patch("cothis.skills.discover_skills", return_value=[_make_skill("python")]):
        Agent._run_preactivation(agent)
    s1.close()

    # discover_skills on resume reads from tmp_path/.agents/skills (set
    # up above) so the real discover call works.
    s2 = Session.load(db_path, s1._session_id, flush_sync=True, cwd=tmp_path)
    assert "python" in s2.active_skills
    s2.close()
