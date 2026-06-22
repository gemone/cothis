"""Tests for ``cothis.cli`` formatting helpers.

``_format_tool_call`` is the only pure function in the CLI module today;
it's worth locking down because its output format is what users read to
debug multi-step agent turns, and the ``repr`` convention (strings quoted,
numbers not) is a deliberate choice.
"""

from __future__ import annotations

from cothis.agent import ToolCallEvent
from cothis.cli import _format_tool_call


def test_format_string_argument_quoted() -> None:
    event = ToolCallEvent(name="fs.read", arguments={"path": "/tmp/x"})
    assert _format_tool_call(event) == "calling fs.read(path='/tmp/x')"


def test_format_multiple_arguments() -> None:
    event = ToolCallEvent(
        name="fs.write",
        arguments={"path": "/tmp/out.txt", "content": "hello"},
    )
    out = _format_tool_call(event)
    # dict iteration order is insertion order; assert both pieces present.
    assert "path='/tmp/out.txt'" in out
    assert "content='hello'" in out
    assert out.startswith("calling fs.write(")
    assert out.endswith(")")


def test_format_integer_argument_not_quoted() -> None:
    event = ToolCallEvent(name="add", arguments={"a": 2, "b": 3})
    out = _format_tool_call(event)
    assert "a=2" in out
    assert "b=3" in out
    # repr distinguishes: 2 not '2'
    assert "a='2'" not in out


def test_format_no_arguments() -> None:
    event = ToolCallEvent(name="noop", arguments={})
    assert _format_tool_call(event) == "calling noop()"


def test_format_string_with_special_chars_repr_escaped() -> None:
    # repr keeps newlines / quotes visible, preventing garbled display.
    event = ToolCallEvent(
        name="fs.write",
        arguments={"content": 'line1\nline2 "quoted"'},
    )
    out = _format_tool_call(event)
    assert "content='line1\\nline2 \"quoted\"'" in out
