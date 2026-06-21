"""Built-in tools exposed to the cothis agent."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

Tool = Callable[..., Any]


def _named(name: str) -> Callable[[Tool], Tool]:
    """Override a callable's tool name.

    ``any-llm`` derives the tool name from ``func.__name__``, which by
    default cannot contain a dot (Python identifiers). This decorator
    rewrites ``__name__`` so we can register namespaced tools such as
    ``fs.read`` and ``fs.write``.
    """

    def decorator(func: Tool) -> Tool:
        func.__name__ = name
        return func

    return decorator


@_named("fs.read")
def read(path: str) -> str:
    """Read the contents of a UTF-8 text file from the filesystem.

    Use this to inspect an existing file before reading or modifying it.

    Args:
        path: Path to the file to read. Relative paths are resolved
            against the current working directory.

    Returns:
        The file contents decoded as UTF-8.
    """
    return Path(path).read_text(encoding="utf-8")


@_named("fs.write")
def write(path: str, content: str) -> str:
    """Write text to a file on the filesystem, creating it if needed.

    Parent directories are created automatically. Existing files are
    overwritten.

    Args:
        path: Path to the file to write.
        content: The text to write to the file.

    Returns:
        A short confirmation with the number of characters written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


TOOLS: list[Tool] = [read, write]
