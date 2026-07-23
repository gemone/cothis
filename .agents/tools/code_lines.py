"""Demo Python tool for the auto-discovery loader.

This tool is a realistic example of what an agent project author would
write: it uses ``@tool`` for schema fidelity, returns a structured result
(dict) that benefits from the JSON/CSV/YAML output formatter, and lives
alongside the YAML tools under ``.agents/tools/``.

Run::

    uv run cothis ask "count lines in src/cothis/agent.py"
"""

from pathlib import Path

from cothis import tool

_CODE_LINES_DESCRIPTION = """Count lines of code in a file.

Returns a ``{total, blank, comment, code}`` dict — ``total`` is the
file's line count, ``blank`` is empty lines, ``comment`` is lines
starting with ``#`` or ``\"\"\"``, ``code`` is the rest.

Example::

    code.lines(path='src/app.py')
    → {"total": 100, "blank": 10, "comment": 20, "code": 70}
"""


@tool("code.lines", description=_CODE_LINES_DESCRIPTION)
def count_lines(path: str) -> dict[str, int]:
    """Count lines of code in a file.

    Args:
        path: Path to the file to count. Relative paths are resolved
            against the current working directory.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    blank = sum(1 for line in lines if not line.strip())
    comment = sum(
        1
        for line in lines
        if line.strip().startswith("#") or line.strip().startswith('"""')
    )
    return {
        "total": len(lines),
        "blank": blank,
        "comment": comment,
        "code": len(lines) - blank - comment,
    }
