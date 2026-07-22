"""Tests for ``<active_skills>`` footer projection (#72).

When any skill is active, the latest user message in the projected
request gets a ``<active_skills>`` text block appended; never written
to the session store, never modifies historical user messages, never
appears when no skill is active.
"""

from __future__ import annotations

from cothis.agent import _request_messages

FOOTER_MARKER = "<active_skills>"


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def test_no_footer_when_active_skills_empty() -> None:
    """Empty ``active_skills`` → footer omitted."""
    messages = [_user("hello")]
    out = _request_messages(messages, active_skills=frozenset())
    assert len(out) == 1
    assert FOOTER_MARKER not in str(out[0]["content"])


def test_no_footer_when_active_skills_none() -> None:
    """``active_skills=None`` (no session) → footer omitted."""
    messages = [_user("hello")]
    out = _request_messages(messages, active_skills=None)
    assert len(out) == 1
    assert FOOTER_MARKER not in str(out[0]["content"])


def test_no_footer_when_active_skills_not_passed() -> None:
    """Backward-compat: ``active_skills`` defaults to None → footer omitted."""
    messages = [_user("hello")]
    out = _request_messages(messages)
    assert FOOTER_MARKER not in str(out[0]["content"])


def test_footer_appended_to_latest_user_message() -> None:
    """Single active skill → footer appended to the latest user message."""
    messages = [_user("hello")]
    out = _request_messages(messages, active_skills=frozenset({"python"}))
    assert len(out[0]["content"]) == 2
    footer = out[0]["content"][1]
    assert footer["type"] == "text"
    assert FOOTER_MARKER in footer["text"]
    assert "python" in footer["text"]


def test_footer_lists_all_active_skills() -> None:
    """Multiple active skills → footer names all of them."""
    messages = [_user("hello")]
    out = _request_messages(
        messages, active_skills=frozenset({"python", "debug"}),
    )
    footer_text = out[0]["content"][-1]["text"]
    assert "python" in footer_text
    assert "debug" in footer_text


def test_footer_only_on_latest_user_message() -> None:
    """Historical user messages are unchanged; only latest gets footer."""
    messages = [
        _user("old"), _assistant("resp"), _user("new"),
    ]
    out = _request_messages(messages, active_skills=frozenset({"python"}))
    # Historical user message untouched.
    assert len(out[0]["content"]) == 1
    assert out[0]["content"][0]["text"] == "old"
    # Latest user message gets the footer.
    assert len(out[2]["content"]) == 2
    assert FOOTER_MARKER in out[2]["content"][1]["text"]


def test_footer_mentions_deactivate_skill() -> None:
    """Footer reminds the model about ``deactivate_skill``."""
    messages = [_user("hi")]
    out = _request_messages(messages, active_skills=frozenset({"python"}))
    footer_text = out[0]["content"][-1]["text"]
    assert "deactivate_skill" in footer_text


def test_no_footer_when_no_user_messages() -> None:
    """No user message in the projection → no crash, no footer appended."""
    messages = [_assistant("just assistant")]
    out = _request_messages(messages, active_skills=frozenset({"python"}))
    assert len(out) == 1
    assert len(out[0]["content"]) == 1  # unchanged
    assert FOOTER_MARKER not in str(out[0]["content"])


def test_footer_not_added_to_string_content() -> None:
    """Legacy string content on latest user msg: footer still appended as block.

    String content is normalised to a list so the footer can be appended
    as a sibling text block.
    """
    messages = [{"role": "user", "content": "plain string"}]
    out = _request_messages(messages, active_skills=frozenset({"python"}))
    # Content should be a list of 2 blocks now.
    assert isinstance(out[0]["content"], list)
    assert len(out[0]["content"]) == 2
    assert FOOTER_MARKER in out[0]["content"][1]["text"]


def test_input_messages_not_mutated() -> None:
    """Footer is projection-only — original messages list is untouched."""
    messages = [_user("hello")]
    original_len = len(messages[0]["content"])
    _request_messages(messages, active_skills=frozenset({"python"}))
    assert len(messages[0]["content"]) == original_len


def test_footer_disappears_when_skill_deactivated() -> None:
    """Round-trip: footer present with active skill, gone without."""
    messages = [_user("hello")]
    with_footer = _request_messages(
        messages, active_skills=frozenset({"python"}),
    )
    without_footer = _request_messages(
        messages, active_skills=frozenset(),
    )
    assert FOOTER_MARKER in str(with_footer[0]["content"])
    assert FOOTER_MARKER not in str(without_footer[0]["content"])


def test_footer_appended_after_other_blocks_in_latest_user() -> None:
    """Latest user message with mixed blocks → footer is appended last.

    A user message with both text and ``tool_result`` blocks is a valid
    Anthropic message shape; the footer appends to it as a sibling
    text block.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "q"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "r"},
        ],
    }]
    out = _request_messages(messages, active_skills=frozenset({"python"}))
    assert len(out[0]["content"]) == 3
    assert out[0]["content"][-1]["type"] == "text"
    assert FOOTER_MARKER in out[0]["content"][-1]["text"]


def test_footer_skipped_when_latest_user_is_tool_result_only() -> None:
    """Post-tool-call state: latest user has only ``tool_result`` blocks.

    Appending a text block to a tool-result-only user message would
    violate Anthropic's tool-flow shape. The footer walks back to the
    latest user-typed text message instead.
    """
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "please help"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            ],
        },
    ]
    out = _request_messages(messages, active_skills=frozenset({"python"}))
    # Latest user (tool_result-only) is untouched.
    assert len(out[2]["content"]) == 1
    assert out[2]["content"][0]["type"] == "tool_result"
    # Footer went on the earlier user-typed text message.
    assert len(out[0]["content"]) == 2
    assert out[0]["content"][-1]["type"] == "text"
    assert FOOTER_MARKER in out[0]["content"][-1]["text"]


def test_footer_skipped_when_all_user_messages_are_tool_result_only() -> None:
    """No user-typed text message to attach to → footer omitted entirely."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            ],
        },
    ]
    out = _request_messages(messages, active_skills=frozenset({"python"}))
    assert len(out[0]["content"]) == 1
    assert FOOTER_MARKER not in str(out[0]["content"])


def test_footer_walks_past_multiple_tool_result_user_messages() -> None:
    """Multi-tool turn: latest user is tool_result-only, walk past earlier ones too."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "a", "name": "x", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "a", "content": "1"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "b", "name": "y", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "b", "content": "2"}],
        },
    ]
    out = _request_messages(messages, active_skills=frozenset({"python"}))
    # Both tool_result-only user messages untouched.
    assert len(out[2]["content"]) == 1
    assert len(out[4]["content"]) == 1
    # Footer on the original user-typed text message.
    assert len(out[0]["content"]) == 2
    assert FOOTER_MARKER in out[0]["content"][-1]["text"]
