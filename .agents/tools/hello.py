"""Demo Python tool for the auto-discovery loader (issue #5).

Run::

    uv run cothis ask "say hello using the hello.world tool"
"""

from cothis import tool


@tool("hello.world")
def hello_world(name: str = "world") -> str:
    """Greet someone.

    Args:
        name: Who to greet. Omit for "world".
    """
    return f"Hello, {name}!"
