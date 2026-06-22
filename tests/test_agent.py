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

from types import SimpleNamespace

from cothis.agent import Agent, _safe_parse_args


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
