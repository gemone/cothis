"""Tests for context-assembly skip of archived blocks (#169, read side).

The projection layer (``_request_messages``) filters out blocks whose
``_cothis_state == 'archived'``. The marker is set by:

* Half A (#167) — at enqueue time, on future writes for archived skills.
* Half B (#168) — queued UPDATE on historical/in-flight SQLite rows.
* ``_deactivate_skill`` itself — also walks the in-memory message
  mirror so the next request sees them filtered without a resume.

On resume, ``_row_to_block`` re-hydrates ``_cothis_state`` from the
row's ``state`` column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cothis.agent import _request_messages
from cothis.session import Session, _row_to_block
from cothis.session.storage import BlockRow

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------
# _request_messages filters archived blocks
# ---------------------------------------------------------------------


def test_request_messages_skips_archived_block() -> None:
    """A single archived block is filtered out of the projection."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "keep"},
            {"type": "text", "text": "hide", "_cothis_state": "archived"},
        ],
    }]
    out = _request_messages(messages)
    texts = [b["text"] for b in out[0]["content"]]
    assert "keep" in texts
    assert "hide" not in texts


def test_request_messages_skips_archived_tool_pair() -> None:
    """Archived ``tool_use`` + ``tool_result`` both skipped — no orphan.

    The paired-skip invariant: filtering is per-block, but Half A + B
    ensure both members of the pair carry ``_cothis_state='archived'``.
    """
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling tool"},
                {
                    "type": "tool_use", "id": "t1", "name": "load_skill",
                    "input": {"name": "python"},
                    "_cothis_skill": "python", "_cothis_state": "archived",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result", "tool_use_id": "t1", "content": "ok",
                    "_cothis_skill": "python", "_cothis_state": "archived",
                },
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "follow-up"}]},
    ]
    out = _request_messages(messages)
    # No archived blocks remain in projection.
    for m in out:
        for b in m["content"]:
            if isinstance(b, dict):
                assert b.get("_cothis_state") != "archived"
                assert not b.get("type") == "tool_use" or b.get("name") != "load_skill"
    # The follow-up text is preserved.
    follow_ups = [
        b for m in out if m["role"] == "user"
        for b in m["content"]
        if isinstance(b, dict) and b.get("text") == "follow-up"
    ]
    assert len(follow_ups) == 1


def test_request_messages_skips_multiple_archived_skills() -> None:
    """Multiple archived skills: all their blocks skipped."""
    messages = [{
        "role": "assistant",
        "content": [
            {
                "type": "tool_use", "id": "a", "name": "load_skill",
                "input": {"name": "python"},
                "_cothis_skill": "python", "_cothis_state": "archived",
            },
            {
                "type": "tool_use", "id": "b", "name": "load_skill",
                "input": {"name": "bash"},
                "_cothis_skill": "bash", "_cothis_state": "archived",
            },
            {"type": "text", "text": "kept text"},
        ],
    }]
    out = _request_messages(messages)
    text_blocks = [
        b for b in out[0]["content"]
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    assert len(text_blocks) == 1
    assert text_blocks[0]["text"] == "kept text"


def test_request_messages_keeps_non_archived_skill_blocks() -> None:
    """Active (not archived) skill blocks are kept in projection."""
    messages = [{
        "role": "assistant",
        "content": [
            {
                "type": "tool_use", "id": "a", "name": "load_skill",
                "input": {"name": "python"}, "_cothis_skill": "python",
            },
        ],
    }]
    out = _request_messages(messages)
    assert len(out[0]["content"]) == 1
    assert out[0]["content"][0]["type"] == "tool_use"


# ---------------------------------------------------------------------
# _deactivate_skill mutates in-memory messages
# ---------------------------------------------------------------------


def test_deactivate_marks_in_memory_messages_for_archived_skill(
    tmp_path: Path,
) -> None:
    """``_deactivate_skill`` walks ``messages`` and marks matching blocks.

    Without this, the next request would still show pre-deactivate
    blocks (the queued UPDATE only touches SQLite; the in-memory mirror
    is stale until resume).
    """
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    s.append_message("assistant", [{
        "type": "tool_use", "id": "t1", "name": "load_skill",
        "input": {"name": "python"}, "_cothis_skill": "python",
    }])
    # Sanity: pre-deactivate, block has no _cothis_state.
    assert s.messages[0]["content"][0].get("_cothis_state") is None

    s._deactivate_skill("python")
    # In-memory mirror updated.
    assert s.messages[0]["content"][0]["_cothis_state"] == "archived"
    s.close()


def test_deactivate_in_memory_mutation_covers_all_matching_blocks(
    tmp_path: Path,
) -> None:
    """Multiple matching blocks across messages all get marked."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    s.append_message("assistant", [{
        "type": "tool_use", "id": "t1", "name": "load_skill",
        "input": {"name": "python"}, "_cothis_skill": "python",
    }])
    s.append_block("user", {
        "type": "tool_result", "tool_use_id": "t1", "content": "ok",
        "_cothis_skill": "python",
    })
    s._deactivate_skill("python")

    # Both blocks marked.
    assert s.messages[0]["content"][0]["_cothis_state"] == "archived"
    # The merged user message has the tool_result.
    for m in s.messages:
        for b in m["content"]:
            if isinstance(b, dict) and b.get("_cothis_skill") == "python":
                assert b["_cothis_state"] == "archived"
    s.close()


def test_deactivate_in_memory_mutation_skill_specific(
    tmp_path: Path,
) -> None:
    """Only the named skill's blocks are marked; other skills untouched."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    # Put both skill blocks in one assistant message (Anthropic alternation
    # rule would merge two consecutive assistant messages anyway).
    s.append_message("assistant", [
        {
            "type": "tool_use", "id": "t1", "name": "load_skill",
            "input": {"name": "python"}, "_cothis_skill": "python",
        },
        {
            "type": "tool_use", "id": "t2", "name": "load_skill",
            "input": {"name": "bash"}, "_cothis_skill": "bash",
        },
    ])
    s._deactivate_skill("python")

    py_block = s.messages[0]["content"][0]
    bash_block = s.messages[0]["content"][1]
    assert py_block["_cothis_state"] == "archived"
    assert bash_block.get("_cothis_state") is None
    s.close()


# ---------------------------------------------------------------------
# _row_to_block rehydrates _cothis_state from row.state
# ---------------------------------------------------------------------


def test_row_to_block_sets_cothis_state_when_state_not_none() -> None:
    """Resume: archived rows rehydrate ``_cothis_state`` on the block."""
    row = BlockRow(
        session_id="s1", seq=0, msg_idx=0, block_idx=0,
        role="assistant", type="tool_use", ts="2026-01-01T00:00:00Z",
        content=None, signature=None, tool_id="t1", tool_name="load_skill",
        tool_input='{"name": "python"}', tool_use_id=None,
        tool_output=None, image_source=None, skill="python", state="archived",
    )
    block = _row_to_block(row)
    assert block.get("_cothis_state") == "archived"


def test_row_to_block_omits_cothis_state_when_state_none() -> None:
    """Active rows: no ``_cothis_state`` marker."""
    row = BlockRow(
        session_id="s1", seq=0, msg_idx=0, block_idx=0,
        role="user", type="text", ts="2026-01-01T00:00:00Z",
        content="hi", signature=None, tool_id=None, tool_name=None,
        tool_input=None, tool_use_id=None,
        tool_output=None, image_source=None, skill=None, state=None,
    )
    block = _row_to_block(row)
    assert "_cothis_state" not in block


# ---------------------------------------------------------------------
# Re-activation epoch: old blocks stay archived, new blocks visible
# ---------------------------------------------------------------------


def test_re_activate_does_not_retroactively_un_archive(tmp_path: Path) -> None:
    """load → deactivate → load: first epoch archived, second epoch visible."""
    s = Session.new(
        tmp_path / "db.db", cwd=tmp_path, model="m", flush_sync=True,
    )
    # Epoch 1: activate + write a python-tagged block.
    s._activate_skill("python")
    s.append_message("assistant", [{
        "type": "tool_use", "id": "t1", "name": "load_skill",
        "input": {"name": "python"}, "_cothis_skill": "python",
    }])
    # Deactivate python. Epoch 1's block gets archived.
    s._deactivate_skill("python")
    # Epoch 2: a new python-tagged block (Half A would mark it on enqueue,
    # but we want to verify the SKIP behaviour). For this test we use
    # a non-skill block so it's clearly visible.
    s.append_message("user", [{"type": "text", "text": "new question"}])

    out = _request_messages(s.messages)
    # Epoch 1 block (archived) is skipped.
    types_names = [
        (b.get("type"), b.get("name"))
        for m in out for b in m["content"] if isinstance(b, dict)
    ]
    assert ("tool_use", "load_skill") not in types_names
    # Epoch 2 text is visible.
    texts = [
        b["text"]
        for m in out for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    assert "new question" in texts
    s.close()


# ---------------------------------------------------------------------
# Agent._request_messages call site uses session.archived state
# ---------------------------------------------------------------------


def test_request_messages_composes_with_active_skills_footer() -> None:
    """Skipped archived block + footer rendering don't interfere (#72)."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use", "id": "t1", "name": "load_skill",
                    "input": {"name": "python"},
                    "_cothis_skill": "python", "_cothis_state": "archived",
                },
            ],
        },
    ]
    out = _request_messages(
        messages, active_skills=frozenset({"python"}),
    )
    # Footer appended (latest user text message).
    footer_blocks = [
        b for b in out[0]["content"]
        if isinstance(b, dict) and b.get("type") == "text"
        and "<active_skills>" in b.get("text", "")
    ]
    assert len(footer_blocks) == 1
    # Archived tool_use skipped.
    types = [
        b.get("type") for b in out[1]["content"]
        if isinstance(b, dict)
    ]
    assert "tool_use" not in types
