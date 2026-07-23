"""Anthropic Messages API projection contract tests (#218).

Pins the hard invariants the projection layer
(``cothis.agent._request_messages`` + ``_assemble_system``) must
honour before any byte hits the wire. Offline + profile-parameterised
— no real provider calls.

Each contract has a passing fixture (real-world shape the projection
actually emits) and a failing fixture (a mutation that would break
the contract if the projection regressed). Mutations are applied to
the *projection output*, not the input — the suite's concern is
"whatever ``_request_messages`` returns must satisfy these 7 rules",
regardless of how the input was shaped.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from cothis.agent import _request_messages

# ---------------------------------------------------------------------
# Profiles — offline stand-ins for provider requirement classes
# ---------------------------------------------------------------------


class Profile:
    """Provider requirement profiles (string sentinels, not real providers)."""

    CLAUDE = "claude"  # no reasoning_content required
    DEEPSEEK_REASONING = "deepseek_reasoning"  # reasoning_content on every tool_call turn
    O_SERIES = "o_series"  # same


REASONING_PROFILES = {Profile.DEEPSEEK_REASONING, Profile.O_SERIES}


# ---------------------------------------------------------------------
# Validators — one per contract. Return list of error strings; [] = ok.
# ---------------------------------------------------------------------


def _role_alternation(messages: list[dict[str, Any]]) -> list[str]:
    """Contract 1 — user/assistant/user/...; no two consecutive same role."""
    errors: list[str] = []
    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i - 1]["role"]:
            errors.append(
                f"messages {i - 1},{i} both role={messages[i]['role']}"
            )
    return errors


def _first_is_user(messages: list[dict[str, Any]]) -> list[str]:
    """Contract 2 — first message is user (Anthropic API rejects assistant-first)."""
    if not messages:
        return ["empty message list"]
    if messages[0]["role"] != "user":
        return [f"first role={messages[0]['role']}, expected 'user'"]
    return []


def _tool_pairing(messages: list[dict[str, Any]]) -> list[str]:
    """Contract 3 — every tool_use has a matching tool_result; no orphans."""
    errors: list[str] = []
    pending: dict[str, int] = {}
    for i, m in enumerate(messages):
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                tid = b.get("id")
                if not tid:
                    errors.append(f"msg {i}: tool_use missing 'id'")
                elif tid in pending:
                    errors.append(f"tool_use {tid!r} at msg {i}: duplicate")
                else:
                    pending[tid] = i
            elif b.get("type") == "tool_result":
                tid = b.get("tool_use_id")
                if not tid:
                    errors.append(f"msg {i}: tool_result missing 'tool_use_id'")
                elif tid not in pending:
                    errors.append(
                        f"tool_result {tid!r} at msg {i}: no preceding tool_use"
                    )
                else:
                    del pending[tid]
    for tid, idx in pending.items():
        errors.append(f"tool_use {tid!r} at msg {idx}: no matching tool_result")
    return errors


def _reasoning_for_profile(
    messages: list[dict[str, Any]],
    profile: str,
) -> list[str]:
    """Contract 4 — assistant msgs with tool_use carry reasoning_content under
    reasoning profiles (DeepSeek, o-series). Non-reasoning profiles skip."""
    if profile not in REASONING_PROFILES:
        return []
    errors: list[str] = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        has_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in content
        )
        has_reasoning = any(
            isinstance(b, dict) and b.get("type") == "thinking"
            for b in content
        )
        if has_tool_use and not has_reasoning:
            errors.append(
                f"msg {i}: assistant tool_use without thinking (profile={profile})"
            )
    return errors


def _tool_result_only_no_text(messages: list[dict[str, Any]]) -> list[str]:
    """Contract 5 — a tool_result-only user message carries no top-level text
    blocks. Footer-append corruption (#72 review) would violate this."""
    errors: list[str] = []
    for i, m in enumerate(messages):
        if m["role"] != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list) or not content:
            continue
        is_tool_result_only = all(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
        if not is_tool_result_only:
            continue
        for b in content:
            assert isinstance(b, dict)
            inner = b.get("content")
            if isinstance(inner, list) and any(
                isinstance(x, dict) and x.get("type") == "text" for x in inner
            ):
                errors.append(
                    f"msg {i}: tool_result carries inner text block "
                    f"(would corrupt tool-flow shape)"
                )
    return errors


def _system_blocks_shape(system_blocks: list[dict[str, Any]]) -> list[str]:
    """Contract 6 — system blocks are type=text with optional cache_control."""
    errors: list[str] = []
    for i, b in enumerate(system_blocks):
        if not isinstance(b, dict):
            errors.append(f"system[{i}]: not a dict")
            continue
        if b.get("type") != "text":
            errors.append(f"system[{i}]: type={b.get('type')!r}, expected 'text'")
            continue
        if not isinstance(b.get("text"), str):
            errors.append(f"system[{i}]: 'text' missing or non-string")
        cc = b.get("cache_control")
        if cc is not None and (
            not isinstance(cc, dict) or cc.get("type") not in {"ephemeral"}
        ):
            errors.append(
                f"system[{i}]: cache_control={cc!r}, expected {{type: ephemeral}}"
            )
    return errors


def _block_shape(messages: list[dict[str, Any]]) -> list[str]:
    """Contract 7 — every content block carries its required fields by type."""
    errors: list[str] = []
    for i, m in enumerate(messages):
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for j, b in enumerate(content):
            if not isinstance(b, dict):
                errors.append(f"msg {i}[{j}]: not a dict")
                continue
            t = b.get("type")
            if t == "text" and not isinstance(b.get("text"), str):
                errors.append(f"msg {i}[{j}]: text block missing 'text'")
            elif t == "tool_use":
                for k in ("id", "name", "input"):
                    if k not in b:
                        errors.append(f"msg {i}[{j}]: tool_use missing {k!r}")
                inp = b.get("input")
                if inp is not None and not isinstance(inp, dict):
                    errors.append(f"msg {i}[{j}]: tool_use input not a dict")
            elif t == "tool_result":
                for k in ("tool_use_id", "content"):
                    if k not in b:
                        errors.append(f"msg {i}[{j}]: tool_result missing {k!r}")
            elif t == "thinking":
                if not isinstance(b.get("thinking"), str):
                    errors.append(f"msg {i}[{j}]: thinking block missing 'thinking'")
            elif t is None:
                errors.append(f"msg {i}[{j}]: block missing 'type'")
    return errors


# ---------------------------------------------------------------------
# Helpers for building fixtures
# ---------------------------------------------------------------------


def _text_user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _text_assistant(text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    }


def _tool_use_assistant(
    tool_id: str,
    name: str = "fs.read",
    thinking: bool = False,
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": "planning the call"})
    blocks.append(
        {"type": "tool_use", "id": tool_id, "name": name, "input": {"path": "a"}}
    )
    return {"role": "assistant", "content": blocks}


def _tool_result_user(tool_id: str, output: str = "ok") -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": output,
            }
        ],
    }


def _archived_tool_use_assistant(tool_id: str) -> dict[str, Any]:
    """Tool-use block with the private archived marker (#169)."""
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": "load_skill",
                "input": {"name": "x"},
                "_cothis_state": "archived",
                "_cothis_skill": "x",
            }
        ],
    }


def _archived_tool_result_user(tool_id: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": "loaded",
                "_cothis_state": "archived",
                "_cothis_skill": "x",
            }
        ],
    }


def _synthetic_preactivation_pair(skill_name: str) -> list[dict[str, Any]]:
    """#73/#188 — tool_use + tool_result with the preact_ prefix + no reasoning."""
    tid = f"preact_{skill_name}"
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tid,
                    "name": "load_skill",
                    "input": {"name": skill_name},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": f"<skill>{skill_name}</skill>",
                }
            ],
        },
    ]


# ---------------------------------------------------------------------
# Fixture set — input shapes drawn from the issue's list
# ---------------------------------------------------------------------


def _archival_skip_input() -> list[dict[str, Any]]:
    """Message list where #169's skip path empties a message."""
    return [
        _text_user("hi"),
        _text_assistant("hello"),
        _archived_tool_use_assistant("toolu_1"),
        _archived_tool_result_user("toolu_1"),
        _text_user("next"),
    ]


def _sentinel_inserted_input() -> list[dict[str, Any]]:
    """Assistant-first list — #149 sentinel covers this at rebuild;
    the contract covers it post-projection."""
    return [
        _text_assistant("I'm first"),
    ]


def _orphan_tool_use_input() -> list[dict[str, Any]]:
    """Tool_use with no matching tool_result."""
    return [
        _text_user("run it"),
        _tool_use_assistant("toolu_orphan_1"),
    ]


def _orphan_tool_result_input() -> list[dict[str, Any]]:
    """Tool_result with no matching tool_use."""
    return [
        _text_user("go"),
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_orphan_2",
                    "content": "ghost",
                }
            ],
        },
    ]


def _synthetic_preactivation_input() -> list[dict[str, Any]]:
    """A persisted pre-activation pair followed by real conversation."""
    return [
        *_synthetic_preactivation_pair("git-commit"),
        _text_user("now do the real work"),
    ]


def _footer_target_walk_input() -> list[dict[str, Any]]:
    """tool_result-only trailing user messages + a real text user msg earlier."""
    return [
        _text_user("start"),
        _tool_use_assistant("toolu_a", thinking=True),
        _tool_result_user("toolu_a"),
        _text_user("based on that, continue"),
        _tool_use_assistant("toolu_b", thinking=True),
        _tool_result_user("toolu_b"),
    ]


def _mixed_blocks_input() -> list[dict[str, Any]]:
    """text + tool_use + thinking in one assistant message."""
    return [
        _text_user("go"),
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "pondering"},
                {"type": "text", "text": "I'll read the file"},
                {
                    "type": "tool_use",
                    "id": "toolu_mix",
                    "name": "fs.read",
                    "input": {"path": "x.py"},
                },
            ],
        },
        _tool_result_user("toolu_mix"),
    ]


# Named fixtures — split into:
#   CLEAN: shapes the projection must handle (pass all 7 contracts)
#   BROKEN: shapes that violate one specific contract — used for
#           per-contract negative tests, not the positive sweep.
CLEAN_FIXTURES: list[tuple[str, list[dict[str, Any]]]] = [
    ("archival_skip", _archival_skip_input()),
    ("synthetic_preactivation", _synthetic_preactivation_input()),
    ("footer_target_walk", _footer_target_walk_input()),
    ("mixed_blocks", _mixed_blocks_input()),
    ("clean_two_turn", [_text_user("hi"), _text_assistant("hello")]),
]


def _project(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run ``_request_messages`` on a copy of the input."""
    return _request_messages(deepcopy(messages))


# ---------------------------------------------------------------------
# Positive sweep — every CLEAN fixture passes every contract.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,input_messages",
    CLEAN_FIXTURES,
    ids=[name for name, _ in CLEAN_FIXTURES],
)
def test_projection_role_alternation(
    name: str, input_messages: list[dict[str, Any]]
) -> None:
    out = _project(input_messages)
    assert _role_alternation(out) == [], _role_alternation(out)


@pytest.mark.parametrize(
    "name,input_messages",
    CLEAN_FIXTURES,
    ids=[name for name, _ in CLEAN_FIXTURES],
)
def test_projection_first_is_user(
    name: str, input_messages: list[dict[str, Any]]
) -> None:
    out = _project(input_messages)
    assert out, "projection must not return empty for non-empty input"
    assert _first_is_user(out) == [], _first_is_user(out)


@pytest.mark.parametrize(
    "name,input_messages",
    CLEAN_FIXTURES,
    ids=[name for name, _ in CLEAN_FIXTURES],
)
def test_projection_tool_pairing(
    name: str, input_messages: list[dict[str, Any]]
) -> None:
    out = _project(input_messages)
    assert _tool_pairing(out) == [], _tool_pairing(out)


@pytest.mark.parametrize(
    "name,input_messages",
    CLEAN_FIXTURES,
    ids=[name for name, _ in CLEAN_FIXTURES],
)
@pytest.mark.parametrize(
    "profile",
    [Profile.CLAUDE, Profile.DEEPSEEK_REASONING, Profile.O_SERIES],
    ids=["claude", "deepseek_reasoning", "o_series"],
)
def test_projection_reasoning_content(
    name: str,
    input_messages: list[dict[str, Any]],
    profile: str,
) -> None:
    out = _project(input_messages)
    errors = _reasoning_for_profile(out, profile)
    assert errors == [], errors


@pytest.mark.parametrize(
    "name,input_messages",
    CLEAN_FIXTURES,
    ids=[name for name, _ in CLEAN_FIXTURES],
)
def test_projection_tool_result_only_no_text(
    name: str, input_messages: list[dict[str, Any]]
) -> None:
    out = _project(input_messages)
    assert _tool_result_only_no_text(out) == [], _tool_result_only_no_text(out)


@pytest.mark.parametrize(
    "name,input_messages",
    CLEAN_FIXTURES,
    ids=[name for name, _ in CLEAN_FIXTURES],
)
def test_projection_block_shape(
    name: str, input_messages: list[dict[str, Any]]
) -> None:
    out = _project(input_messages)
    assert _block_shape(out) == [], _block_shape(out)


def test_system_blocks_shape_via_assemble() -> None:
    """Contract 6 — ``_assemble_system`` output is all text blocks."""
    from cothis.agent import _assemble_system

    blocks = _assemble_system("you are an agent")
    errors = _system_blocks_shape(blocks)
    assert errors == [], errors


# ---------------------------------------------------------------------
# Active-skills footer honours tool_result-only user messages (#72)
# ---------------------------------------------------------------------


def test_footer_not_appended_to_tool_result_only_trailing_user() -> None:
    """The walk must skip trailing tool_result-only user messages; the
    footer lands on the latest user message with a non-tool_result block."""
    msg = _footer_target_walk_input()
    out = _project(msg)
    out_with_footer = _request_messages(
        deepcopy(msg), active_skills=frozenset({"git-commit"})
    )
    assert len(out_with_footer) == len(out)
    # Trailing message is tool_result-only → footer did not land there.
    last = out_with_footer[-1]
    assert all(b.get("type") == "tool_result" for b in last["content"])
    # Some earlier user message gained the footer text.
    footer_found = any(
        any(
            isinstance(b, dict)
            and b.get("type") == "text"
            and "<active_skills>" in b.get("text", "")
            for b in m["content"]
        )
        for m in out_with_footer
    )
    assert footer_found, "footer was not appended anywhere"


# ---------------------------------------------------------------------
# Negative variants — each fixture (or constructed shape) violates
# exactly one contract; the validator must catch it.
# ---------------------------------------------------------------------


def test_negative_consecutive_same_role_caught() -> None:
    """Contract 1 negative — two same-role messages back-to-back."""
    out = _project([_text_user("hi"), _text_assistant("hello")])
    out.append(_text_assistant("same role"))
    assert _role_alternation(out)


def test_negative_assistant_first_caught() -> None:
    """Contract 2 negative — assistant-first message list."""
    out = _project(_sentinel_inserted_input())
    # Projection doesn't synthesise a sentinel user (#149 covers that at
    # rebuild), so the assistant-first shape persists to the wire.
    assert _first_is_user(out) != []


def test_negative_orphan_tool_use_caught() -> None:
    """Contract 3 negative — tool_use with no matching tool_result."""
    out = _project(_orphan_tool_use_input())
    assert _tool_pairing(out) != []


def test_negative_orphan_tool_result_caught() -> None:
    """Contract 3 negative — tool_result with no preceding tool_use."""
    assert _tool_pairing([_text_user("x"), _archived_tool_result_user("ghost")]) != []


def test_negative_missing_reasoning_under_reasoning_profile() -> None:
    """Contract 4 negative — assistant tool_use without thinking under
    DeepSeek/o-series."""
    msg = _tool_use_assistant("toolu_x", thinking=False)
    errors = _reasoning_for_profile([msg], Profile.DEEPSEEK_REASONING)
    assert errors != []


def test_negative_footer_injected_into_tool_result_only_caught() -> None:
    """Contract 5 negative — a tool_result-only user message whose inner
    content carries a text block would corrupt the tool-flow shape."""
    msg = _tool_result_user("toolu_y")
    msg["content"].append(
        {
            "type": "tool_result",
            "tool_use_id": "toolu_y",
            "content": [{"type": "text", "text": "<active_skills>…</active_skills>"}],
        }
    )
    assert _tool_result_only_no_text([msg]) != []


def test_negative_non_text_system_block_caught() -> None:
    """Contract 6 negative — a non-text block in system."""
    bad_blocks = [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "x"}]
    assert _system_blocks_shape(bad_blocks) != []


def test_negative_malformed_tool_use_block_caught() -> None:
    """Contract 7 negative — tool_use missing 'input'."""
    bad = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t", "name": "n"}],
        }
    ]
    assert _block_shape(bad) != []
