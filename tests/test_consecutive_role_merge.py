"""Tests for consecutive same-role message merge after archival (#205).

When ``_request_messages`` filters out an entire user message (all
blocks archived), the surrounding assistant messages become
consecutive. Anthropic rejects this with HTTP 400. The fix: merge
consecutive same-role messages into one.
"""

from __future__ import annotations

from cothis.agent import _request_messages


def test_consecutive_assistant_merged_after_user_skip() -> None:
    """User message fully archived → surrounding assistants merge."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll use foo"},
                {
                    "type": "tool_use", "id": "t1", "name": "foo",
                    "input": {}, "_cothis_skill": "foo", "_cothis_state": "archived",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result", "tool_use_id": "t1", "content": "ok",
                    "_cothis_skill": "foo", "_cothis_state": "archived",
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done!"}]},
    ]
    out = _request_messages(messages)
    roles = [m["role"] for m in out]
    # No consecutive same-role messages.
    for i in range(1, len(roles)):
        assert roles[i] != roles[i - 1], f"consecutive {roles[i]} at index {i}"
    # Both assistant texts survived (merged into one message).
    all_texts = [
        b.get("text") for m in out for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    assert "I'll use foo" in all_texts
    assert "Done!" in all_texts


def test_consecutive_user_merged_when_assistant_fully_archived() -> None:
    """Assistant message fully archived → surrounding users merge."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use", "id": "t1", "name": "foo",
                    "input": {}, "_cothis_skill": "foo", "_cothis_state": "archived",
                },
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]
    out = _request_messages(messages)
    roles = [m["role"] for m in out]
    for i in range(1, len(roles)):
        assert roles[i] != roles[i - 1], f"consecutive {roles[i]} at index {i}"
    all_texts = [
        b.get("text") for m in out for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    assert "first" in all_texts
    assert "second" in all_texts


def test_three_consecutive_same_role_all_merge() -> None:
    """Multiple messages of the same role merge into one."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "b"},
                {
                    "type": "tool_use", "id": "t1", "name": "foo",
                    "input": {}, "_cothis_skill": "foo", "_cothis_state": "archived",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result", "tool_use_id": "t1", "content": "ok",
                    "_cothis_skill": "foo", "_cothis_state": "archived",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "c"},
                {
                    "type": "tool_use", "id": "t2", "name": "bar",
                    "input": {}, "_cothis_skill": "bar", "_cothis_state": "archived",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result", "tool_use_id": "t2", "content": "ok",
                    "_cothis_skill": "bar", "_cothis_state": "archived",
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "d"}]},
    ]
    out = _request_messages(messages)
    roles = [m["role"] for m in out]
    for i in range(1, len(roles)):
        assert roles[i] != roles[i - 1], f"consecutive {roles[i]} at index {i}"


def test_no_merge_needed_when_alternation_preserved() -> None:
    """Normal conversation — no merge needed, output unchanged."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        {"role": "user", "content": [{"type": "text", "text": "bye"}]},
    ]
    out = _request_messages(messages)
    assert len(out) == 3
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
