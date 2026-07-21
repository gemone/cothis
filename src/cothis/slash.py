"""``cothis.slash`` — chat REPL slash command dispatch.

Async registry mapping ``/cmd`` → handler. The REPL checks the leading
``/`` before calling the agent; non-slash input passes through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cothis.session import Session


@dataclass
class SlashContext:
    """Per-call context handed to every slash handler.

    Carries the live :class:`Session` so commands like ``/reload-skills``
    can mutate the catalog in place. Grow this only when a real command
    needs a new field — premature surface bloat is harder to walk back.
    """

    session: Session | None = None
    args: str = ""


class SlashHandler(Protocol):
    """Async callable ``(ctx, args: str) -> str | None``."""

    async def __call__(self, ctx: SlashContext, args: str) -> str | None: ...


@dataclass
class _Entry:
    handler: SlashHandler
    summary: str


# Module-level registry — no class wrapping needed (AGENTS.md #56 precedent:
# dict with no real invariant → module-level functions, not a class shell).
_entries: dict[str, _Entry] = {}


def register(name: str, handler: SlashHandler, *, summary: str = "") -> None:
    """Map ``name`` to ``handler``. Re-registering replaces silently.

    cothis: silent overwrite — simplest contract; upgrade to a
    warning if a real collision bites.
    """
    _entries[name] = _Entry(handler=handler, summary=summary)


def names() -> list[str]:
    """Return registered command names, sorted."""
    return sorted(_entries)


async def dispatch(line: str, *, ctx: SlashContext | None = None) -> str | None:
    """Route ``line`` to its handler. Returns the handler's message
    (``""`` if the handler returned ``None``), or ``None`` if ``line``
    is not a slash command (REPL should forward to the agent).

    Unknown ``/cmd`` produces a local listing of registered commands.
    """
    if not line.startswith("/"):
        return None
    ctx = ctx if ctx is not None else SlashContext()
    name, _, args = line[1:].partition(" ")
    if not name:
        return None
    if name == "help":
        return _render_help()
    entry = _entries.get(name)
    if entry is None:
        return _render_unknown(name)
    ctx.args = args
    result = await entry.handler(ctx, args)
    return "" if result is None else result


def _render_help() -> str:
    if not _entries:
        return "No slash commands registered."
    lines = ["/help — show this listing"]
    for name in names():
        summary = _entries[name].summary
        tail = f" — {summary}" if summary else ""
        lines.append(f"/{name}{tail}")
    return "\n".join(lines)


def _render_unknown(name: str) -> str:
    if not _entries:
        return f"unknown slash command: /{name} (no commands registered)"
    lines = [f"unknown slash command: /{name}", "available:"]
    for n in names():
        summary = _entries[n].summary
        tail = f" — {summary}" if summary else ""
        lines.append(f"  /{n}{tail}")
    return "\n".join(lines)
