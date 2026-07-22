"""Tests for persist-time skill tagging (#164).

Covers the three pieces reviewer flagged as untested on PR #163:

* ``_request_messages`` strips ``_cothis_*`` private keys before send.
* Tagged ``tool_use`` + ``tool_result`` round-trip through ``BlockRow.skill``.
* ``_skill_for_block`` returns ``None`` for non-``skill_marker`` tools.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from cothis.agent import _request_messages
from cothis.session import _block_to_row
from cothis.session.storage import BlockRow, SessionRow, Storage


def test_request_messages_strips_cothis_private_keys() -> None:
    """``_cothis_*`` keys are internal tags; must not leak to the model."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok",
                 "_cothis_skill": "python"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "t2", "name": "load_skill",
                 "input": {"name": "python"}, "_cothis_skill": "python"},
            ],
        },
    ]
    out = _request_messages(messages)
    for m in out:
        for b in m["content"]:
            if isinstance(b, dict):
                assert not any(k.startswith("_cothis_") for k in b), (
                    f"private key leaked to request: {b}"
                )
    # The non-private fields survive.
    assert out[0]["content"][0]["tool_use_id"] == "t1"
    assert out[1]["content"][1]["input"] == {"name": "python"}


def test_request_messages_passes_through_non_dict_blocks() -> None:
    """String content blocks (legacy) pass through untouched."""
    messages = [{"role": "user", "content": ["plain string"]}]
    out = _request_messages(messages)
    assert out[0]["content"] == ["plain string"]


def test_block_to_row_reads_cothis_skill_for_tool_use(tmp_path: Any) -> None:
    """Tagged ``tool_use`` block persists skill name on ``BlockRow.skill``."""
    block = {
        "type": "tool_use",
        "id": "tu1",
        "name": "load_skill",
        "input": {"name": "python"},
        "_cothis_skill": "python",
    }
    row = _block_to_row(
        "s1", seq=0, msg_idx=0, block_idx=0,
        role="assistant", ts="2026-01-01T00:00:00Z",
        block=block,
    )
    assert row.skill == "python"
    assert row.tool_name == "load_skill"


def test_block_to_row_reads_cothis_skill_for_tool_result(tmp_path: Any) -> None:
    """Tagged ``tool_result`` block persists skill name on ``BlockRow.skill``."""
    block = {
        "type": "tool_result",
        "tool_use_id": "tu1",
        "content": "loaded",
        "_cothis_skill": "python",
    }
    row = _block_to_row(
        "s1", seq=1, msg_idx=1, block_idx=0,
        role="user", ts="2026-01-01T00:00:01Z",
        block=block,
    )
    assert row.skill == "python"
    assert row.tool_use_id == "tu1"


def test_block_to_row_skill_none_when_no_tag() -> None:
    """Untagged block → ``BlockRow.skill`` is None (default)."""
    block = {"type": "text", "text": "hi"}
    row = _block_to_row(
        "s1", seq=0, msg_idx=0, block_idx=0,
        role="user", ts="2026-01-01T00:00:00Z",
        block=block,
    )
    assert row.skill is None


def test_tagged_blocks_round_trip_through_storage(tmp_path: Any) -> None:
    """Tagged tool_use + tool_result persist + reload with skill intact."""
    db = tmp_path / "test.db"  # type: ignore[union-attr]
    storage = Storage(db)

    sr = SessionRow(
        id="s1", parent_id=None, parent_seq=None, cwd="/x",
        cli_version="0.1.0", model="m", title="t",
        created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
    )
    rows = [
        BlockRow(
            session_id="s1", seq=0, msg_idx=0, block_idx=0,
            role="assistant", type="tool_use", ts="2026-01-01T00:00:00Z",
            content=None, signature=None, tool_id="tu1", tool_name="load_skill",
            tool_input='{"name": "python"}', tool_use_id=None,
            tool_output=None, image_source=None, skill="python",
        ),
        BlockRow(
            session_id="s1", seq=1, msg_idx=1, block_idx=0,
            role="user", type="tool_result", ts="2026-01-01T00:00:01Z",
            content="loaded", signature=None, tool_id=None, tool_name=None,
            tool_input=None, tool_use_id="tu1",
            tool_output="loaded", image_source=None, skill="python",
        ),
    ]
    storage.write_atomic(sr, rows, "2026-01-01T00:00:02Z")
    reloaded = storage.load_blocks("s1")
    assert len(reloaded) == 2
    assert reloaded[0].skill == "python"
    assert reloaded[0].tool_name == "load_skill"
    assert reloaded[1].skill == "python"
    assert reloaded[1].tool_use_id == "tu1"
    storage.close()


def test_skill_for_block_returns_none_for_non_marker_tool() -> None:
    """Non-``skill_marker`` tools (e.g. ``fs.read``) do not get tagged.

    Builds a minimal Agent-like object with just the ``_tool_map`` +
    ``_skill_for_block`` bound method, so we don't need a full Agent
    (which would need a live LLM client).
    """
    from cothis.agent import Agent

    fake_tool = SimpleNamespace(_skill_marker=False)
    agent = Agent.__new__(Agent)  # bypass __init__
    agent._tool_map = {"fs_read": fake_tool}  # type: ignore[attr-defined]

    block = {
        "type": "tool_use", "id": "t1", "name": "fs_read",
        "input": {"path": "/x"},
    }
    assert agent._skill_for_block(block) is None


def test_skill_for_block_returns_name_for_marker_tool() -> None:
    """``skill_marker`` tools with a ``name`` arg return that name."""
    from cothis.agent import Agent

    fake_tool = SimpleNamespace(_skill_marker=True)
    agent = Agent.__new__(Agent)
    agent._tool_map = {"load_skill": fake_tool}  # type: ignore[attr-defined]

    block = {
        "type": "tool_use", "id": "t1", "name": "load_skill",
        "input": {"name": "python"},
    }
    assert agent._skill_for_block(block) == "python"


def test_skill_for_block_none_when_marker_tool_lacks_name_arg() -> None:
    """A ``skill_marker`` tool invoked without a ``name`` arg → None."""
    from cothis.agent import Agent

    fake_tool = SimpleNamespace(_skill_marker=True)
    agent = Agent.__new__(Agent)
    agent._tool_map = {"load_skill": fake_tool}  # type: ignore[attr-defined]

    block = {
        "type": "tool_use", "id": "t1", "name": "load_skill",
        "input": {},
    }
    assert agent._skill_for_block(block) is None


def test_skill_for_block_none_for_non_tool_use() -> None:
    """Text blocks return None regardless of tool_map contents."""
    from cothis.agent import Agent

    agent = Agent.__new__(Agent)
    agent._tool_map = {}  # type: ignore[attr-defined]
    assert agent._skill_for_block({"type": "text", "text": "hi"}) is None


def test_tag_skill_blocks_tags_marker_tool_use_only() -> None:
    """``_tag_skill_blocks`` tags ``skill_marker`` tool_use, skips others."""
    from cothis.agent import Agent

    marker_tool = SimpleNamespace(_skill_marker=True)
    plain_tool = SimpleNamespace(_skill_marker=False)
    agent = Agent.__new__(Agent)
    agent._tool_map = {  # type: ignore[attr-defined]
        "load_skill": marker_tool, "fs_read": plain_tool,
    }

    content = [
        {"type": "text", "text": "thinking..."},
        {"type": "tool_use", "id": "t1", "name": "load_skill",
         "input": {"name": "python"}},
        {"type": "tool_use", "id": "t2", "name": "fs_read",
         "input": {"path": "/x"}},
    ]
    agent._tag_skill_blocks(content)
    assert content[1]["_cothis_skill"] == "python"
    assert "_cothis_skill" not in content[2]
    assert "_cothis_skill" not in content[0]
