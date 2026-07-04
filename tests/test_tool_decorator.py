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

from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import Field

from cothis.tools import Tool, ToolDef, tool
from cothis.tools.core import _parse_docstring


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
    """ " "The ``Args:`` section's per-line descriptions reach the schema."""

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
    """ " "Python annotations map to JSON-Schema types."""

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
    """ " "An annotation not in the type map defaults to ``string``."""

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
    """ " "Args with defaults are optional; args without are required."""

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
    """ " "Multi-line arg descriptions are collapsed to single-line."""

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
    """ " "``@tool(name=..., description=...)`` overrides both fields."""

    @tool(name="custom.tool", description="Override description.")
    def f(x: str) -> str:
        """Original docstring (ignored when description= given)."""
        return x

    assert f.__name__ == "custom.tool"
    schema = f.__cothis_schema__
    assert schema["function"]["name"] == "custom.tool"
    assert schema["function"]["description"] == "Override description."


def test_tool_no_parens_uses_dunder_name() -> None:
    """ " "``@tool`` (no parens) uses ``__name__`` as the tool name."""

    @tool
    def bare(x: str) -> str:
        """Bare."""
        return x

    assert bare.__cothis_schema__["function"]["name"] == "bare"


def test_no_docstring_yields_default_description() -> None:
    """ " "A function without a docstring falls back to a derived description."""

    @tool
    def bare(x: str) -> str:
        return ""

    schema = bare.__cothis_schema__
    # No docstring → description falls back. The fallback names the function.
    assert "bare" in schema["function"]["description"]


def test_no_args_section_yields_no_descriptions() -> None:
    """ " "A docstring without ``Args:`` produces no per-arg descriptions."""

    @tool
    def f(x: str) -> str:
        """Does a thing, but doesn't document its arg."""

        return x

    props = f.__cothis_schema__["function"]["parameters"]["properties"]
    assert "description" not in props["x"]


def test_var_args_dropped_from_schema() -> None:
    """ " "``*args`` and ``**kwargs`` are excluded from the schema."""

    @tool
    def f(a: str, *args: Any, **kwargs: Any) -> str:
        """F.

        Args:
            a: a real arg.
        """
        return ""

    props = f.__cothis_schema__["function"]["parameters"]["properties"]
    assert set(props) == {"a"}


def test_annotated_field_constraints_reach_schema() -> None:
    """``Annotated[T, Field(...)]`` constraints flow through to the JSON Schema (story 4).

    pydantic's ``TypeAdapter`` honours ``ge`` / ``le`` / ``min_length`` etc.,
    so the model sees the valid range and picks in-range arguments.
    """

    @tool
    def set_score(score: Annotated[int, Field(ge=0, le=100)]) -> str:
        """Set the score.

        Args:
            score: A value 0-100.
        """
        return str(score)

    props = set_score.__cothis_schema__["function"]["parameters"]["properties"]
    score_prop = props["score"]
    assert score_prop["type"] == "integer"
    assert score_prop["minimum"] == 0
    assert score_prop["maximum"] == 100
    assert score_prop["description"] == "A value 0-100."


def test_basic_type_schema_unchanged() -> None:
    """A plain ``int`` annotation still produces ``{"type": "integer"}`` — no regression."""

    @tool
    def echo(n: int) -> str:
        """Echo.

        Args:
            n: a number.
        """
        return str(n)

    props = echo.__cothis_schema__["function"]["parameters"]["properties"]
    assert props["n"] == {"type": "integer", "description": "a number."}


def test_parse_docstring_helper_directly() -> None:
    """ " "``_parse_docstring`` returns (first-paragraph summary, {arg: description})."""
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
    """ " "An empty/None docstring yields empty summary + empty args."""
    assert _parse_docstring(None) == ("", {})
    assert _parse_docstring("") == ("", {})


def test_tool_returns_same_callable_type() -> None:
    """ " "``@tool`` returns the function itself (not a wrapper), with attributes."""

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
    from cothis.tools.builtins import _load_gitignore

    assert _load_gitignore(tmp_path) is None


def test_load_gitignore_parses_patterns(tmp_path: Any) -> None:
    """ " "``_load_gitignore`` returns a PathSpec matching .gitignore lines."""
    from cothis.tools.builtins import _load_gitignore

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
    from cothis.tools.builtins import _list_dir

    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("hi")
    result = _list_dir(path=str(tmp_path))
    assert isinstance(result, list)
    by_name = {e["name"]: e["type"] for e in result}
    assert by_name == {"src": "dir", "README.md": "file"}


def test_dir_nonexistent_returns_error_string(tmp_path: Any) -> None:
    """``fs.dir`` on a missing path returns an ``"Error: ..."`` str.

    Error paths stay as strings (not structured) — ``_execute`` passes them
    through unchanged so the model sees an actionable message.
    """
    from cothis.tools.builtins import _list_dir

    result = _list_dir(path=str(tmp_path / "nonexistent"))
    assert isinstance(result, str)
    assert result.startswith("Error: no such directory")


def test_dir_recursive_includes_nested_paths(tmp_path: Any) -> None:
    """ " "Recursive listing yields entries with nested relative paths."""
    from cothis.tools.builtins import _list_dir

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("")
    result = _list_dir(path=str(tmp_path), recursive=True)
    names = {e["name"] for e in result}
    assert "pkg" in names
    assert "pkg/mod.py" in names


# --------------------------------------------------------------------
# Tool output formatting (COTHIS_TOOL_OUTPUT_FORMAT)
# --------------------------------------------------------------------


def test_format_default_is_json(monkeypatch: Any) -> None:
    """ " "Without ``COTHIS_TOOL_OUTPUT_FORMAT``, structured output is JSON."""
    from cothis.tools.format import format_tool_output

    monkeypatch.delenv("COTHIS_TOOL_OUTPUT_FORMAT", raising=False)
    out = format_tool_output([{"a": 1}])
    assert out == '[{"a": 1}]'


def test_format_csv_table(monkeypatch: Any) -> None:
    """ "``list[dict]`` renders as a CSV table with header + rows."""
    from cothis.tools.format import format_tool_output

    monkeypatch.setenv("COTHIS_TOOL_OUTPUT_FORMAT", "csv")
    out = format_tool_output(
        [{"name": "src", "type": "dir"}, {"name": "x", "type": "file"}]
    )
    lines = out.splitlines()
    assert lines[0] == "name,type"
    assert lines[1] == "src,dir"
    assert lines[2] == "x,file"


def test_format_tsv_uses_tab_delimiter(monkeypatch: Any) -> None:
    """ "``tsv`` is the same as csv but with tab separators."""
    from cothis.tools.format import format_tool_output

    monkeypatch.setenv("COTHIS_TOOL_OUTPUT_FORMAT", "tsv")
    out = format_tool_output([{"a": "1", "b": "2"}])
    assert "\t" in out
    assert out.splitlines()[0] == "a\tb"


def test_format_csv_flattens_nested_dict(monkeypatch: Any) -> None:
    """ "``csv`` flattens nested dicts with dotted key paths."""
    from cothis.tools.format import format_tool_output

    monkeypatch.setenv("COTHIS_TOOL_OUTPUT_FORMAT", "csv")
    out = format_tool_output({"name": "src", "meta": {"type": "dir", "size": 1024}})
    header = out.splitlines()[0]
    assert "meta.type" in header
    assert "meta.size" in header
    assert "name" in header


def test_format_csv_bare_list_falls_back_to_json(monkeypatch: Any) -> None:
    """ "A bare list of scalars isn't tabular → CSV falls back to JSON."""
    from cothis.tools.format import format_tool_output

    monkeypatch.setenv("COTHIS_TOOL_OUTPUT_FORMAT", "csv")
    out = format_tool_output(["a", "b", "c"])
    import json as _json

    assert _json.loads(out) == ["a", "b", "c"]


def test_format_yaml_handles_nested(monkeypatch: Any) -> None:
    """ "YAML renders nested structures natively (no flattening)."""
    from cothis.tools.format import format_tool_output

    monkeypatch.setenv("COTHIS_TOOL_OUTPUT_FORMAT", "yaml")
    out = format_tool_output({"name": "src", "meta": {"type": "dir"}})
    assert "name: src" in out
    assert "meta:" in out
    assert "  type: dir" in out  # nested key is indented, not flattened


def test_format_unknown_value_defaults_to_json(monkeypatch: Any) -> None:
    """ "An unrecognised ``COTHIS_TOOL_OUTPUT_FORMAT`` value falls back to JSON."""
    from cothis.tools.format import format_tool_output

    monkeypatch.setenv("COTHIS_TOOL_OUTPUT_FORMAT", "xml")
    out = format_tool_output({"a": 1})
    import json as _json

    assert _json.loads(out) == {"a": 1}


# --------------------------------------------------------------------
# ToolDef class (issue #4)
# --------------------------------------------------------------------


def test_tool_returns_tooldef_instance() -> None:
    """ "``@tool`` returns a ``ToolDef``, not a bare function."""

    @tool
    def f(x: str) -> str:
        """F."""
        return x

    assert isinstance(f, ToolDef)
    assert isinstance(f, Tool)  # satisfies the dispatch Protocol


def test_tooldef_satisfies_tool_protocol() -> None:
    """ "A ``ToolDef`` has ``__name__``, ``__call__``, and ``__cothis_schema__``."""

    @tool("ns.name")
    def my_tool(x: str) -> str:
        """My tool.

        Args:
            x: an arg.
        """
        return x

    assert my_tool.__name__ == "ns.name"
    assert callable(my_tool)
    assert my_tool.__cothis_schema__["function"]["name"] == "ns.name"
    # __call__ delegates to the wrapped function.
    assert my_tool(x="hello") == "hello"


def test_hook_registration_appends_to_ordered_list() -> None:
    """Each ``.pre_execute()`` / ``.after_execute()`` / etc. call appends to a list.

    Multiple callbacks on the same stage are stored in registration order.
    This is the storage contract the chain semantics (pipeline / AND) depend on.
    """

    @tool("x")
    def x(arg: str) -> str:
        """X."""
        return arg

    @x.pre_execute()
    def first(args: Any) -> Any:
        return args

    @x.pre_execute()
    def second(args: Any) -> Any:
        return args

    @x.after_execute()
    def after(result: Any, args: Any) -> Any:
        return result

    # pre_execute has two callbacks, in registration order.
    assert x._hooks["pre_execute"] == [first, second]
    # after_execute has one.
    assert x._hooks["after_execute"] == [after]
    # Stages with no registrations are empty.
    assert x._hooks["pre_load"] == []
    assert x._hooks["after_load"] == []
    assert x._hooks["on_error"] == []


def test_hook_decorator_returns_callback_unchanged() -> None:
    """ "The hook decorator returns the callback so it stays referenceable."""

    @tool("x")
    def x(arg: str) -> str:
        """X."""
        return arg

    @x.pre_load()
    def my_check() -> Any:
        return True

    # The decorator returned ``my_check``, so it's still in scope under that name.
    assert my_check in x._hooks["pre_load"]


def test_all_five_hook_stages_exist() -> None:
    """ "``ToolDef`` exposes all five lifecycle stages as decorator methods."""

    @tool("x")
    def x(arg: str) -> str:
        """X."""
        return arg

    # Each stage is a callable that returns a decorator.
    for stage_name in (
        "pre_load",
        "after_load",
        "pre_execute",
        "after_execute",
        "on_error",
    ):
        decorator_factory = getattr(x, stage_name)
        decorator = decorator_factory()
        # Registering a callback through it works.

        @decorator
        def cb(*args: Any) -> Any:
            pass

        assert cb in x._hooks[stage_name]


def test_builtin_tools_are_tooldef_instances() -> None:
    """``fs.read``, ``fs.dir``, ``fs.write`` are all ``ToolDef`` instances.

    Regression check: the migration from bare decorated functions to
    ``ToolDef`` must not break the built-in tools.
    """
    from cothis.tools.builtins import _list_dir, read, write

    for t in (read, _list_dir, write):
        assert isinstance(t, ToolDef)
        assert isinstance(t, Tool)
        assert t.__cothis_schema__ is not None


# --------------------------------------------------------------------
# Python tool discovery (issue #5)
# --------------------------------------------------------------------


def test_load_python_tools_discovers_single_file(tmp_path: Any) -> None:
    """ "A ``.py`` file with a ``@tool`` function is discovered and loaded."""
    from cothis.tools.core import load_tools_from_layer

    (tmp_path / "greet.py").write_text(
        'from cothis import tool\n\n@tool("test.greet")\n'
        'def greet(name: str) -> str:\n    """Greet."""\n    return f"hi {name}"\n',
        encoding="utf-8",
    )
    tools = load_tools_from_layer(tmp_path)
    assert len(tools) == 1
    assert tools[0].__name__ == "test.greet"
    assert tools[0](name="x") == "hi x"


def test_load_python_tools_discovers_package(tmp_path: Any) -> None:
    """ "A package directory with ``__init__.py`` is discovered."""
    from cothis.tools.core import load_tools_from_layer

    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        'from cothis import tool\n\n@tool("pkg.tool")\n'
        'def t(x: str) -> str:\n    """T."""\n    return x\n',
        encoding="utf-8",
    )
    tools = load_tools_from_layer(tmp_path)
    assert len(tools) == 1
    assert tools[0].__name__ == "pkg.tool"


def test_load_python_tools_import_failure_doesnt_crash(
    tmp_path: Any, caplog: Any
) -> None:
    """ "A broken ``.py`` file logs an error but doesn't crash the loader."""
    import logging

    from cothis.tools.core import load_tools_from_layer

    (tmp_path / "broken.py").write_text("import nonexistent_module\n", encoding="utf-8")
    # A valid file alongside the broken one should still load.
    (tmp_path / "good.py").write_text(
        'from cothis import tool\n\n@tool("good")\n'
        'def g() -> str:\n    """G."""\n    return "ok"\n',
        encoding="utf-8",
    )
    with caplog.at_level(logging.ERROR, logger="cothis.tools"):
        tools = load_tools_from_layer(tmp_path)
    assert len(tools) == 1  # the broken one was skipped, the good one loaded
    assert tools[0].__name__ == "good"
    # The broken file's error was logged — actionable (file path + exception).
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "broken.py" in error_records[0].message
    assert "nonexistent_module" in error_records[0].message


def test_load_python_tools_empty_dir_returns_empty(tmp_path: Any) -> None:
    """ "An empty directory yields an empty tool list."""
    from cothis.tools.core import load_tools_from_layer

    assert load_tools_from_layer(tmp_path) == []


def test_load_python_tools_missing_dir_returns_empty() -> None:
    """ "A non-existent directory yields an empty tool list."""
    from cothis.tools.core import load_tools_from_layer

    assert load_tools_from_layer(Path("/nonexistent/path")) == []


def test_load_python_tools_ignores_non_tooldef_attributes(
    tmp_path: Any,
) -> None:
    """ "Module-level constants and plain functions are NOT collected."""
    from cothis.tools.core import load_tools_from_layer

    (tmp_path / "mixed.py").write_text(
        "from cothis import tool\n\n"
        "CONSTANT = 42\n"
        "def helper():\n    pass\n\n"
        '@tool("real.tool")\n'
        'def real(x: str) -> str:\n    """Real."""\n    return x\n',
        encoding="utf-8",
    )
    tools = load_tools_from_layer(tmp_path)
    assert len(tools) == 1  # only the @tool function, not CONSTANT/helper
    assert tools[0].__name__ == "real.tool"


# --------------------------------------------------------------------
# Load-time hooks: pre_load / after_load / on_error (issue #6)
# --------------------------------------------------------------------


def test_loader_discovers_tool_with_pre_load_hooks(tmp_path: Any) -> None:
    """The loader discovers a tool regardless of its pre_load hooks.

    Hooks are NOT run by the loader — they run later in ``discover_tools``
    after cross-layer merge (see ADR-0003). The loader returns all
    discovered candidates.
    """
    from cothis.tools.core import load_tools_from_layer

    (tmp_path / "t.py").write_text(
        "from cothis import tool\n\n"
        "calls = []\n\n"
        '@tool("t")\n'
        'def t() -> str:\n    """T."""\n    return "ok"\n\n'
        "@t.pre_load()\n"
        "def first():\n    calls.append(1); return True\n\n"
        "@t.pre_load()\n"
        "def second():\n    calls.append(2); return True\n",
        encoding="utf-8",
    )
    tools = load_tools_from_layer(tmp_path)
    assert len(tools) == 1  # discovered (hooks not run yet)


def test_pre_load_any_false_skips_tool() -> None:
    """If any ``pre_load`` callback returns False, ``_run_load_hooks`` returns False.

    The loader discovers the tool; the skip happens when ``discover_tools`` runs
    load hooks on it (see ADR-0003). The third callback never runs.
    """

    @tool("t")
    def t() -> str:
        """T."""
        return "ok"

    @t.pre_load()
    def first() -> Any:
        return True

    @t.pre_load()
    def blocker() -> Any:
        return False

    @t.pre_load()
    def third() -> Any:
        raise AssertionError("should not run")

    assert t._run_load_hooks() is False  # blocker returned False


def test_pre_load_exception_skips_and_triggers_on_error() -> None:
    """ "``pre_load`` raising skips the tool and fires ``on_error`` with phase."""
    errors: list[tuple[str, str]] = []

    @tool("t")
    def t() -> str:
        """T."""
        return "ok"

    @t.pre_load()
    def boom() -> None:
        raise RuntimeError("env check failed")

    @t.on_error()
    def observe(exc: Exception, phase: str, args: Any, result: Any) -> None:
        errors.append((str(exc), phase))

    assert t._run_load_hooks() is False
    assert errors == [("env check failed", "pre_load")]


def test_after_load_multiple_callbacks_all_run() -> None:
    """ "Multiple ``after_load`` callbacks all run in order (no short-circuit)."""
    calls: list[int] = []

    @tool("t")
    def t() -> str:
        """T."""
        return "ok"

    @t.after_load()
    def first() -> None:
        calls.append(1)

    @t.after_load()
    def second() -> None:
        calls.append(2)

    assert t._run_load_hooks() is True
    assert calls == [1, 2]


def test_after_load_exception_skips_and_triggers_on_error() -> None:
    """ "``after_load`` raising skips the tool and fires ``on_error``."""
    errors: list[tuple[str, str]] = []

    @tool("t")
    def t() -> str:
        """T."""
        return "ok"

    @t.after_load()
    def boom() -> None:
        raise RuntimeError("init failed")

    @t.on_error()
    def observe(exc: Exception, phase: str, args: Any, result: Any) -> None:
        errors.append((str(exc), phase))

    assert t._run_load_hooks() is False
    assert errors == [("init failed", "after_load")]


def test_on_error_self_exception_swallowed() -> None:
    """If ``on_error`` itself raises, the exception is swallowed (warning logged)."""

    @tool("t")
    def t() -> str:
        """T."""
        return "ok"

    @t.pre_load()
    def boom() -> None:
        raise RuntimeError("orig")

    @t.on_error()
    def broken_observer(exc: Exception, phase: str, args: Any, result: Any) -> None:
        raise ConnectionError("telemetry down")

    # Should not crash — on_error's own exception is swallowed.
    assert t._run_load_hooks() is False  # tool still skipped (pre_load raised)


def test_no_hooks_registered_loads_normally(tmp_path: Any) -> None:
    """ "A tool with no hooks is discovered and registered as before."""
    from cothis.tools.core import load_tools_from_layer

    (tmp_path / "t.py").write_text(
        "from cothis import tool\n\n"
        '@tool("plain")\n'
        'def t(x: str) -> str:\n    """Plain."""\n    return x\n',
        encoding="utf-8",
    )
    tools = load_tools_from_layer(tmp_path)
    assert len(tools) == 1
    assert tools[0].__name__ == "plain"


# --------------------------------------------------------------------
# Execute-time hooks: pre_execute / after_execute / on_error (issue #7)
# --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_execute_pipeline_multiple_callbacks(monkeypatch: Any) -> None:
    """ "Multiple ``pre_execute`` callbacks form a pipeline (A's output feeds B)."""
    from unittest.mock import MagicMock

    import any_llm

    from cothis.agent import Agent

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])

    @tool("pipe")
    def echo(x: str) -> str:
        """Echo."""
        return x

    @echo.pre_execute()
    def add_a_suffix(args):
        args["x"] = args["x"] + "-A"
        return args

    @echo.pre_execute()
    def add_b_suffix(args):
        # Sees A's modification (pipeline).
        args["x"] = args["x"] + "-B"
        return args

    agent._tool_map["pipe"] = echo
    tc = MagicMock()
    tc.function.name = "pipe"
    tc.function.arguments = '{"x": "start"}'
    result = await agent._execute(tc)
    assert result == "start-A-B"  # both pre_execute callbacks ran in order


@pytest.mark.asyncio
async def test_pre_execute_exception_short_circuits_and_returns_error(
    monkeypatch: Any,
) -> None:
    """ "``pre_execute`` raising short-circuits; error string returned to LLM."""
    from unittest.mock import MagicMock

    import any_llm

    from cothis.agent import Agent

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])

    @tool("guarded")
    def guarded(x: str) -> str:
        """Guarded."""
        return x

    @guarded.pre_execute()
    def block(args):
        raise ValueError("blocked by pre_execute")

    agent._tool_map["guarded"] = guarded
    tc = MagicMock()
    tc.function.name = "guarded"
    tc.function.arguments = '{"x": "hi"}'
    result = await agent._execute(tc)
    assert result.startswith("Error calling guarded: blocked by pre_execute")


@pytest.mark.asyncio
async def test_after_execute_pipeline_multiple_callbacks(monkeypatch: Any) -> None:
    """ "Multiple ``after_execute`` callbacks form a pipeline on result."""
    from unittest.mock import MagicMock

    import any_llm

    from cothis.agent import Agent

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])

    @tool("transform")
    def base() -> str:
        """Base."""
        return "hello"

    @base.after_execute()
    def upper(result, args):
        return result.upper()

    @base.after_execute()
    def add_suffix(result, args):
        # Sees upper's output (pipeline).
        return result + "!"

    agent._tool_map["transform"] = base
    tc = MagicMock()
    tc.function.name = "transform"
    tc.function.arguments = "{}"
    result = await agent._execute(tc)
    assert result == "HELLO!"  # both after_execute callbacks ran in order


@pytest.mark.asyncio
async def test_after_execute_exception_uses_original_result(monkeypatch: Any) -> None:
    """ "``after_execute`` raising uses the original result (don't hide output)."""
    from unittest.mock import MagicMock

    import any_llm

    from cothis.agent import Agent

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])

    @tool("safe")
    def base() -> str:
        """Base."""
        return "real-output"

    @base.after_execute()
    def broken(result, args):
        raise RuntimeError("after_execute crashed")

    agent._tool_map["safe"] = base
    tc = MagicMock()
    tc.function.name = "safe"
    tc.function.arguments = "{}"
    result = await agent._execute(tc)
    # Original result preserved — broken after_execute doesn't hide it.
    assert result == "real-output"


@pytest.mark.asyncio
async def test_on_error_fires_on_tool_body_exception(monkeypatch: Any) -> None:
    """ "Tool body exception fires ``on_error`` with phase='tool'."""
    from unittest.mock import MagicMock

    import any_llm

    from cothis.agent import Agent

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])

    errors = []

    @tool("crash")
    def base() -> str:
        """Base."""
        raise RuntimeError("tool body failed")

    @base.on_error()
    def observe(exc, phase, args, result):
        errors.append((str(exc), phase))

    agent._tool_map["crash"] = base
    tc = MagicMock()
    tc.function.name = "crash"
    tc.function.arguments = "{}"
    result = await agent._execute(tc)
    assert result.startswith("Error calling crash")
    assert errors == [("tool body failed", "tool")]


@pytest.mark.asyncio
async def test_on_error_at_execute_phase_correct(monkeypatch: Any) -> None:
    """ "``on_error`` phase is 'pre_execute' when pre_execute raises."""
    from unittest.mock import MagicMock

    import any_llm

    from cothis.agent import Agent

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])

    errors = []

    @tool("t")
    def base() -> str:
        """Base."""
        return "never"

    @base.pre_execute()
    def boom(args):
        raise RuntimeError("pre_execute failed")

    @base.on_error()
    def observe(exc, phase, args, result):
        errors.append((str(exc), phase))

    agent._tool_map["t"] = base
    tc = MagicMock()
    tc.function.name = "t"
    tc.function.arguments = "{}"
    await agent._execute(tc)
    assert errors == [("pre_execute failed", "pre_execute")]


@pytest.mark.asyncio
async def test_no_hooks_execute_baseline_unchanged(monkeypatch: Any) -> None:
    """ "A tool with no execute hooks dispatches exactly as before."""
    from unittest.mock import MagicMock

    import any_llm

    from cothis.agent import Agent

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )
    agent = Agent(model="x", provider="openrouter", tools=[])

    @tool("plain")
    def base(x: str) -> str:
        """Plain."""
        return f"got {x}"

    agent._tool_map["plain"] = base
    tc = MagicMock()
    tc.function.name = "plain"
    tc.function.arguments = '{"x": "hello"}'
    assert await agent._execute(tc) == "got hello"


# --------------------------------------------------------------------
# Hook observability (issue #8): logger.debug covers hook lifecycle
# --------------------------------------------------------------------


def test_pre_load_skip_logged(caplog: Any) -> None:
    """``pre_load`` returning False emits a debug log naming the tool + reason."""
    import logging

    from cothis.tools import tool

    @tool("skipme")
    def t() -> str:
        """T."""
        return "never"

    @t.pre_load()
    def block() -> bool:
        return False

    with caplog.at_level(logging.DEBUG, logger="cothis.tools"):
        result = t._run_load_hooks()
    assert result is False
    skip_records = [
        r for r in caplog.records if "pre_load" in r.message and "False" in r.message
    ]
    assert len(skip_records) == 1
    assert "skipme" in skip_records[0].message


def test_on_error_fire_logged(caplog: Any) -> None:
    """``on_error`` fires when pre_load raises, and its debug log names the phase."""
    import logging

    from cothis.tools import tool

    @tool("boom")
    def t() -> str:
        """T."""
        return "never"

    @t.pre_load()
    def explode() -> None:
        raise RuntimeError("env gone")

    @t.on_error()
    def observe(exc: Exception, phase: str, args: Any, result: Any) -> None:
        pass

    with caplog.at_level(logging.DEBUG, logger="cothis.tools"):
        result = t._run_load_hooks()
    assert result is False
    # The skip message should name the tool + the exception.
    skip_msgs = [r.message for r in caplog.records if "pre_load" in r.message]
    assert any("boom" in m and "env gone" in m for m in skip_msgs)


# --------------------------------------------------------------------
# Duplicate tool name detection (issue #3, story 44; issue #12)
# Three behaviors coexist (ADR-0003):
#   - same-layer duplicate (any format combo) → raise
#   - cross-layer (project vs user, custom vs builtin) → shadow
#   - builtin override → shadow
# These tests pin the first (raise); cross-layer shadow + builtin override
# live in tests/test_cli.py.
# --------------------------------------------------------------------


def test_yaml_duplicate_names_detected(tmp_path: Any) -> None:
    """Two YAML files with the same ``name:`` raise ValueError naming both paths."""
    from cothis.tools.core import load_tools_from_layer

    (tmp_path / "a.yaml").write_text(
        'name: dup\ncommand: ["echo", "a"]\n', encoding="utf-8"
    )
    (tmp_path / "b.yaml").write_text(
        'name: dup\ncommand: ["echo", "b"]\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate tool name.*dup") as exc_info:
        load_tools_from_layer(tmp_path)
    msg = str(exc_info.value)
    assert "a.yaml" in msg
    assert "b.yaml" in msg


def test_python_duplicate_names_detected(tmp_path: Any) -> None:
    """Two Python tools with the same name raise ValueError naming both paths."""
    from cothis.tools.core import load_tools_from_layer

    (tmp_path / "a.py").write_text(
        'from cothis import tool\n@tool("dup")\n'
        'def a() -> str:\n    """A."""\n    return "a"\n',
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        'from cothis import tool\n@tool("dup")\n'
        'def b() -> str:\n    """B."""\n    return "b"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate tool name.*dup") as exc_info:
        load_tools_from_layer(tmp_path)
    msg = str(exc_info.value)
    assert "a.py" in msg
    assert "b.py" in msg


def test_cross_format_same_layer_duplicate_raises(tmp_path: Any) -> None:
    """YAML + Python in the SAME directory claiming one name → raise.

    Format is never a layer (ADR-0003 Q1): a YAML file and a Python file
    in the same directory are same-layer, so they raise — not shadow.
    This is the case the pre-#12 per-format ``seen`` dicts couldn't catch.
    """
    from cothis.tools.core import load_tools_from_layer

    (tmp_path / "y.yaml").write_text(
        'name: dup\ncommand: ["echo", "yaml"]\n', encoding="utf-8"
    )
    (tmp_path / "p.py").write_text(
        'from cothis import tool\n@tool("dup")\n'
        'def p() -> str:\n    """P."""\n    return "py"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate tool name.*dup") as exc_info:
        load_tools_from_layer(tmp_path)
    msg = str(exc_info.value)
    assert "y.yaml" in msg
    assert "p.py" in msg


def test_no_duplicate_names_loads_normally(tmp_path: Any) -> None:
    """Distinct names load without error."""
    from cothis.tools.core import load_tools_from_layer

    (tmp_path / "a.yaml").write_text(
        'name: first\ncommand: ["echo", "a"]\n', encoding="utf-8"
    )
    (tmp_path / "b.yaml").write_text(
        'name: second\ncommand: ["echo", "b"]\n', encoding="utf-8"
    )
    tools = load_tools_from_layer(tmp_path)
    assert len(tools) == 2
