"""Tests for ``cothis.agent`` pure helpers.

These helpers are the silent-breakage surface of the chat streaming path:

- ``_assemble_tool_calls`` — by-index merge of streamed ``ChoiceDeltaToolCall``
  fragments into ``SimpleNamespace`` tool calls. If the merge or the
  arguments-string concatenation drifts, chat still *runs* but the agent
  sees malformed tool arguments and starts emitting errors like
  "could not parse tool arguments".
- ``_safe_parse_args`` — best-effort JSON parse with a ``{"_raw": raw}``
  fallback. A regression here either crashes mid-stream (unhandled JSON
  error) or hides what the provider actually sent.

Fragment data is inline (no fixtures on disk): the sequence mirrors what
OpenRouter produced for a real ``add(a=2, b=3)`` tool call, captured once.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from cothis.agent import Agent, MaxIterationsError, _safe_parse_args

if TYPE_CHECKING:
    from cothis.tools import Tool


def _frag(
    index: int,
    *,
    id_: str | None = None,
    name: str | None = None,
    args: str | None = None,
) -> SimpleNamespace:
    """Build a minimal object shaped like ``ChoiceDeltaToolCall``."""
    return SimpleNamespace(
        index=index,
        id=id_,
        function=SimpleNamespace(name=name, arguments=args),
    )


# Real fragment sequence from OpenRouter streaming a ``add(a=2, b=3)`` call.
# First fragment carries id + name; the rest carry arguments-string shards
# that must concatenate into valid JSON.
FRAGMENTS = [
    _frag(0, id_="call-xyz", name="add", args=""),
    _frag(0, args=""),
    _frag(0, args='{"'),
    _frag(0, args="a"),
    _frag(0, args='":'),
    _frag(0, args="2"),
    _frag(0, args="," + '"'),
    _frag(0, args="b"),
    _frag(0, args='":'),
    _frag(0, args="3"),
    _frag(0, args="}"),
]


def _bare_agent() -> Agent:
    """An Agent instance without running ``model_post_init``.

    We only exercise private helpers that don't touch AnyLLM or the network,
    so skip pydantic's init to avoid needing a real provider.
    """
    return Agent.__new__(Agent)


def test_assemble_single_call_merges_fragments() -> None:
    agent = _bare_agent()
    assembled = agent._assemble_tool_calls(FRAGMENTS)

    assert len(assembled) == 1
    call = assembled[0]
    assert call.id == "call-xyz"
    assert call.function.name == "add"
    assert call.function.arguments == '{"a":2,"b":3}'


def test_assemble_parallel_calls_sorted_by_index() -> None:
    # Fragments arrive in arbitrary order; output must be sorted by index.
    parallel = [
        _frag(1, id_="c1", name="fs.read", args='{"path":"a"}'),
        _frag(0, id_="c0", name="fs.write", args='{"path":"b","content":"x"}'),
    ]
    assembled = _bare_agent()._assemble_tool_calls(parallel)
    assert [c.id for c in assembled] == ["c0", "c1"]


def test_assemble_arguments_concatenate_across_shards() -> None:
    # Long arguments split across many shards must concatenate exactly.
    parts = ['{"a', '":1,"b"', ":2}"]
    shards = [_frag(0, id_="x", name="fs.write", args=part) for part in parts]
    assembled = _bare_agent()._assemble_tool_calls(shards)
    assert len(assembled) == 1
    assert assembled[0].function.arguments == '{"a":1,"b":2}'


def test_safe_parse_args_valid_json() -> None:
    assert _safe_parse_args('{"path":"/x"}') == {"path": "/x"}


def test_safe_parse_args_none_returns_empty() -> None:
    # None → {} (not {"_raw": None}) so display formats no-arg calls cleanly.
    assert _safe_parse_args(None) == {}


def test_safe_parse_args_empty_string_returns_empty() -> None:
    assert _safe_parse_args("") == {}


def test_safe_parse_args_malformed_falls_back_to_raw() -> None:
    assert _safe_parse_args("{trailing,") == {"_raw": "{trailing,"}


def test_safe_parse_args_non_dict_falls_back_to_raw() -> None:
    # Lists / numbers aren't valid tool args; surface them raw.
    assert _safe_parse_args("[1,2,3]") == {"_raw": "[1,2,3]"}
    assert _safe_parse_args("42") == {"_raw": "42"}


def test_agent_rejects_non_tool() -> None:
    # ``tools`` field validation rejects anything not satisfying the Tool
    # protocol. Validation runs during __init__ before model_post_init, so
    # AnyLLM.create is never reached (no provider/network needed). ``cast``
    # satisfies ty at the call site without a per-line type-ignore — the value
    # is deliberately wrong at runtime.
    not_a_tool = cast("Tool", object())
    with pytest.raises(ValidationError):
        Agent(model="x", provider="mistral", tools=[not_a_tool])


def test_run_retries_on_empty_message_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty provider message (no content, no tool_calls) loops, doesn't exit.

    Regression check for the silent-blank-exit bug: ``run`` used to return
    ``message.content or ""`` on any no-tool-call turn, so a provider emitting
    an empty message made ``ask`` print a blank line and exit. The fix: only
    return when content is non-empty; otherwise loop. This test fakes a
    provider that emits one empty turn then a real answer, and asserts both
    that ``run`` retries and that it ultimately returns the content.
    """
    import asyncio
    from unittest.mock import MagicMock

    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[], max_iterations=5)

    state = {"turn": 0}

    async def fake_acompletion(**kwargs: Any) -> Any:
        state["turn"] += 1
        resp = MagicMock()
        msg = MagicMock()
        if state["turn"] == 1:
            msg.content = None  # provider emits nothing
            msg.tool_calls = None
        else:
            msg.content = "recovered"
            msg.tool_calls = None
        resp.choices = [MagicMock(message=msg)]
        return resp

    monkeypatch.setattr(agent._llm, "acompletion", fake_acompletion)
    result = asyncio.run(agent.run("hi"))
    assert result == "recovered"
    assert state["turn"] == 2  # retried once after the empty turn


def test_run_raises_on_persistent_empty_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent empty messages exhaust ``max_iterations`` (loud, not silent).

    Without this, a provider that always emits empty messages would loop
    forever or silently exit. The fix drives to ``MaxIterationsError``.
    """
    import asyncio
    from unittest.mock import MagicMock

    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[], max_iterations=3)

    async def fake_acompletion(**kwargs: Any) -> Any:
        resp = MagicMock()
        msg = MagicMock()
        msg.content = None
        msg.tool_calls = None
        resp.choices = [MagicMock(message=msg)]
        return resp

    monkeypatch.setattr(agent._llm, "acompletion", fake_acompletion)
    with pytest.raises(MaxIterationsError, match="no content and no tool calls"):
        asyncio.run(agent.run("hi"))


@pytest.mark.asyncio
async def test_execute_str_result_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool returning ``str`` gets passed through to the tool message as-is.

    Str output (file contents, confirmations, stdout, error strings) is the
    common case; it must not be re-encoded.
    """
    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])
    agent._tool_map["str_tool"] = lambda **kw: "plain text"

    tc = MagicMock()
    tc.id = "c1"
    tc.function.name = "str_tool"
    tc.function.arguments = "{}"
    assert await agent._execute(tc) == "plain text"


@pytest.mark.asyncio
async def test_execute_dict_result_serialised_as_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool returning ``dict`` is serialised as JSON, not ``str(dict)``.

    Why: ``str(dict)`` uses Python repr quotes (``{'k': 'v'}``), which LLMs
    parse unreliably. JSON (``{"k": "v"}``) is the model-native shape.
    """
    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])
    payload = {"name": "src", "type": "dir", "children": ["a.py", "b.py"]}
    agent._tool_map["dict_tool"] = lambda **kw: payload

    tc = MagicMock()
    tc.id = "c1"
    tc.function.name = "dict_tool"
    tc.function.arguments = "{}"
    rendered = await agent._execute(tc)
    # Round-trips through json.loads → model sees accurate structure.
    assert json.loads(rendered) == payload
    # Uses JSON double-quoted form, not Python repr single-quoted.
    assert '"name": "src"' in rendered
    assert "'name'" not in rendered


@pytest.mark.asyncio
async def test_execute_list_result_serialised_as_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool returning ``list`` is serialised as JSON (same rationale as dict)."""
    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])
    payload = [{"name": "x", "type": "f"}, {"name": "y", "type": "d"}]
    agent._tool_map["list_tool"] = lambda **kw: payload

    tc = MagicMock()
    tc.id = "c1"
    tc.function.name = "list_tool"
    tc.function.arguments = "{}"
    rendered = await agent._execute(tc)
    assert json.loads(rendered) == payload


@pytest.mark.asyncio
async def test_execute_non_str_non_collection_falls_back_to_str(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool returning an int (or other non-collection) falls back to ``str()``."""
    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])
    agent._tool_map["int_tool"] = lambda **kw: 42

    tc = MagicMock()
    tc.id = "c1"
    tc.function.name = "int_tool"
    tc.function.arguments = "{}"
    assert await agent._execute(tc) == "42"


@pytest.mark.asyncio
async def test_execute_async_tool_coroutine_awaited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool returning a coroutine has its result awaited (ADR-0004).

    Sync tools return plain values; the ``isawaitable`` check skips the
    await. Async tools (MCP) return coroutines; the await activates. This
    test proves the async path works — without it, a regression removing
    the ``isawaitable`` bridge would silently break MCP tools while all
    sync-tool tests still pass.
    """
    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])

    async def async_echo(**kw: Any) -> str:
        return "from coroutine"

    agent._tool_map["async_tool"] = async_echo

    tc = MagicMock()
    tc.id = "c1"
    tc.function.name = "async_tool"
    tc.function.arguments = "{}"
    assert await agent._execute(tc) == "from coroutine"
