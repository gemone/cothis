"""Tests for the ``@tool`` decorator and its docstring-parsing path.

The decorator is the schema-fidelity surface for Python tools: it parses a
Google-style docstring (``griffe``) + ``inspect.signature`` and builds an
OpenAI-format schema carrying per-arg descriptions. If the parsing or schema
construction drifts, the LLM silently sees generic descriptions (the same
lossy behaviour as ``any-llm``'s ``callable_to_tool``) — defeating the
decorator's reason for existing.

Covers:

- **Summary extraction**: first docstring paragraph → tool ``description``.
- **Per-arg descriptions**: ``Args:`` section → ``properties[name].description``.
- **Type mapping**: annotation → JSON-Schema type (with ``string`` fallback).
- **Required vs optional**: presence of a default → ``required`` list.
- **Multi-line descriptions**: indentation/newlines collapsed to single spaces.
- **Namespaced names**: ``@tool("fs.read")`` positional / keyword name
  override flows through to the schema's ``name`` field.
- **Edge cases**: no docstring, no ``Args:`` section, ``*args``/``**kwargs``
  dropped from the schema.
"""

from __future__ import annotations

from typing import Any

from cothis.tools import Tool, _parse_docstring, tool


def test_summary_from_first_paragraph() -> None:
    """The docstring's first paragraph becomes the tool description.

    Only the first paragraph (before the first blank line) is kept —
    extended notes and parameter-behaviour prose stay out of the schema's
    top-level ``description``. Per-arg detail lives in each parameter's own
    ``description`` via the ``Args:`` section.
    """

    @tool
    def greet(name: str) -> str:
        """Greet someone by name.

        Longer explanation that is NOT part of the summary.
        """

        return f"Hello, {name}"

    schema = greet.__cothis_schema__
    assert schema is not None
    desc = schema["function"]["description"]
    assert desc == "Greet someone by name."
    assert "Longer explanation" not in desc


def test_per_arg_description_from_args_section() -> None:
    """The ``Args:`` section's per-line descriptions reach the schema."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: The first operand.
            b: The second operand.
        """

        return a + b

    props = add.__cothis_schema__["function"]["parameters"]["properties"]
    assert props["a"]["description"] == "The first operand."
    assert props["b"]["description"] == "The second operand."


def test_type_mapping_from_annotation() -> None:
    """Python annotations map to JSON-Schema types."""

    @tool
    def typed(n: int, s: str, f: float, b: bool, items: list) -> str:
        """Typed.

        Args:
            n: an int.
            s: a str.
            f: a float.
            b: a bool.
            items: a list.
        """
        return ""

    props = typed.__cothis_schema__["function"]["parameters"]["properties"]
    assert props["n"]["type"] == "integer"
    assert props["s"]["type"] == "string"
    assert props["f"]["type"] == "number"
    assert props["b"]["type"] == "boolean"
    assert props["items"]["type"] == "array"


def test_unknown_annotation_falls_back_to_string() -> None:
    """An annotation not in the type map defaults to ``string``."""

    @tool
    def custom(x: Any) -> str:
        """Custom.

        Args:
            x: something.
        """
        return ""

    props = custom.__cothis_schema__["function"]["parameters"]["properties"]
    assert props["x"]["type"] == "string"


def test_required_vs_optional_from_defaults() -> None:
    """Args with defaults are optional; args without are required."""

    @tool
    def f(req: str, opt: str = "x") -> str:
        """F.

        Args:
            req: required.
            opt: optional.
        """
        return ""

    required = f.__cothis_schema__["function"]["parameters"]["required"]
    assert required == ["req"]


def test_multiline_description_collapsed() -> None:
    """Multi-line arg descriptions are collapsed to single-line."""

    @tool
    def read(path: str) -> str:
        """Read a file.

        Args:
            path: Path to the file to read. Relative paths are resolved
                against the current working directory.
                eg. "src/main.py".
        """
        return ""

    desc = read.__cothis_schema__["function"]["parameters"]["properties"]["path"][
        "description"
    ]
    assert "Relative paths are resolved" in desc
    # No stray newlines in the collapsed description.
    assert "\n" not in desc


def test_tool_with_positional_name() -> None:
    """``@tool("fs.read")`` sets the name from the positional arg.

    The schema's ``name`` field carries the namespaced name, and
    ``__name__`` is rewritten so ``Agent._tool_map`` keys correctly.
    """

    @tool("fs.read")
    def read(path: str) -> str:
        """Read.

        Args:
            path: where.
        """
        return ""

    assert read.__name__ == "fs.read"
    assert read.__cothis_schema__["function"]["name"] == "fs.read"


def test_tool_with_keyword_name_and_description() -> None:
    """``@tool(name=..., description=...)`` overrides both fields."""

    @tool(name="custom.tool", description="Override description.")
    def f(x: str) -> str:
        """Original docstring (ignored when description= given)."""
        return x

    assert f.__name__ == "custom.tool"
    schema = f.__cothis_schema__
    assert schema["function"]["name"] == "custom.tool"
    assert schema["function"]["description"] == "Override description."


def test_tool_no_parens_uses_dunder_name() -> None:
    """``@tool`` (no parens) uses ``__name__`` as the tool name."""

    @tool
    def bare(x: str) -> str:
        """Bare."""
        return x

    assert bare.__cothis_schema__["function"]["name"] == "bare"


def test_no_docstring_yields_default_description() -> None:
    """A function without a docstring falls back to a derived description."""

    @tool
    def bare(x: str) -> str:
        return ""

    schema = bare.__cothis_schema__
    # No docstring → description falls back. The fallback names the function.
    assert "bare" in schema["function"]["description"]


def test_no_args_section_yields_no_descriptions() -> None:
    """A docstring without ``Args:`` produces no per-arg descriptions."""

    @tool
    def f(x: str) -> str:
        """Does a thing, but doesn't document its arg."""

        return x

    props = f.__cothis_schema__["function"]["parameters"]["properties"]
    assert "description" not in props["x"]


def test_var_args_dropped_from_schema() -> None:
    """``*args`` and ``**kwargs`` are excluded from the schema."""

    @tool
    def f(a: str, *args: Any, **kwargs: Any) -> str:
        """F.

        Args:
            a: a real arg.
        """
        return ""

    props = f.__cothis_schema__["function"]["parameters"]["properties"]
    assert set(props) == {"a"}


def test_parse_docstring_helper_directly() -> None:
    """``_parse_docstring`` returns (first-paragraph summary, {arg: description})."""
    summary, args = _parse_docstring(
        """Summary line.

        More detail (excluded from summary).

        Args:
            x: the x value.
            y: the y value.
        """
    )
    assert summary == "Summary line."
    assert args == {"x": "the x value.", "y": "the y value."}


def test_parse_docstring_empty() -> None:
    """An empty/None docstring yields empty summary + empty args."""
    assert _parse_docstring(None) == ("", {})
    assert _parse_docstring("") == ("", {})


def test_tool_returns_same_callable_type() -> None:
    """``@tool`` returns the function itself (not a wrapper), with attributes."""

    @tool
    def f(x: str) -> str:
        """F.

        Args:
            x: arg.
        """
        return x

    # Still callable directly, returns what the body returns.
    assert f(x="hello") == "hello"
    # Satisfies the Tool protocol (has __name__ + __call__).
    assert isinstance(f, Tool)


def test_griffe_warnings_do_not_leak(caplog: Any) -> None:
    """griffe's "no type annotation" log warnings are silenced.

    If the silencing breaks, these warnings pollute stderr at import time
    for every ``@tool``-decorated function. ``caplog`` captures log
    records; we assert none came from griffe.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="griffe"):
        _parse_docstring(
            """Summary.

            Args:
                x: the x value.
            """
        )
    griffe_warnings = [r for r in caplog.records if r.name == "griffe"]
    assert griffe_warnings == []


def test_load_gitignore_returns_none_when_absent(tmp_path: Any) -> None:
    """``_load_gitignore`` returns None when the directory has no .gitignore.

    Callers use None to skip the matching pass entirely, so this contract
    must hold — a stray PathSpec on a no-gitignore dir would silently filter
    nothing, but returning a real object when None is documented would
    mislead callers into thinking they have patterns to apply.
    """
    from cothis.tools import _load_gitignore

    assert _load_gitignore(tmp_path) is None


def test_load_gitignore_parses_patterns(tmp_path: Any) -> None:
    """``_load_gitignore`` returns a PathSpec matching .gitignore lines."""
    from cothis.tools import _load_gitignore

    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")
    spec = _load_gitignore(tmp_path)
    assert spec is not None
    assert spec.match_file("debug.log")
    assert spec.match_file("build/output.txt")
    assert not spec.match_file("src/main.py")


def test_read_start_line_beyond_eof_returns_actionable_error(
    tmp_path: Any,
) -> None:
    """``fs.read`` with ``start_line`` past EOF returns an error, not empty.

    Regression check: the model can't know file length ahead of time, so
    an out-of-range ``start_line`` is common. Returning empty gave it
    nothing to act on. The error names the requested line and actual length.
    """
    from cothis.tools import read

    f = tmp_path / "small.txt"
    f.write_text("line1\nline2\nline3\n")
    result = read(path=str(f), start_line=100, end_line=200)
    assert result.startswith("Error: start_line 100 is beyond EOF")
    assert "3 lines" in result  # names the actual length


def test_dir_returns_structured_entries(tmp_path: Any) -> None:
    """``fs.dir`` returns a list of ``{"name", "type"}`` dicts, not text.

    Structured output lets ``_execute`` serialise it as JSON (the model-native
    shape) instead of a bespoke text format the model has to parse.
    """
    from cothis.tools import dir

    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("hi")
    result = dir(path=str(tmp_path))
    assert isinstance(result, list)
    by_name = {e["name"]: e["type"] for e in result}
    assert by_name == {"src": "dir", "README.md": "file"}


def test_dir_nonexistent_returns_error_string(tmp_path: Any) -> None:
    """``fs.dir`` on a missing path returns an ``"Error: ..."`` str.

    Error paths stay as strings (not structured) — ``_execute`` passes them
    through unchanged so the model sees an actionable message.
    """
    from cothis.tools import dir

    result = dir(path=str(tmp_path / "nonexistent"))
    assert isinstance(result, str)
    assert result.startswith("Error: no such directory")


def test_dir_recursive_includes_nested_paths(tmp_path: Any) -> None:
    """Recursive listing yields entries with nested relative paths."""
    from cothis.tools import dir

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("")
    result = dir(path=str(tmp_path), recursive=True)
    names = {e["name"] for e in result}
    assert "pkg" in names
    assert "pkg/mod.py" in names
