"""``cothis.slash`` — chat REPL slash command dispatch.

A small async registry mapping ``/cmd`` → handler. The REPL checks the
leading ``/`` before calling the agent; non-slash input passes through
untouched. Unknown commands print a local error listing available
commands — no LLM round-trip.

Handlers receive a :class:`SlashContext` carrying the session (and any
context the second consumer needs; the surface stays minimal until then).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

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
    """Async callable taking a :class:`SlashContext` (plus optional args
    parsed off the input line) and returning an optional message string."""

    async def __call__(self, ctx: SlashContext, args: str = "") -> str | None: ...


@dataclass
class _Entry:
    handler: Any  # SlashHandler (Protocol; can't bind the async type cleanly)
    summary: str
    takes_args: bool


def _handler_takes_args(handler: Any) -> bool:
    """``True`` iff ``handler`` declares ``(ctx, args)`` (not just ``(ctx)``).

    Detected once at register time so dispatch doesn't pay a signature
    introspection per call. ``functools.partial`` and lambdas are
    handled by ``inspect.signature``'s normal unwrap behaviour.
    """
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    params = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(params) >= 2


class SlashRegistry:
    """``/cmd`` → async handler map + dispatch.

    Register with :meth:`register`, dispatch with :meth:`dispatch`. The
    REPL passes every typed line through ``dispatch``; non-slash lines
    return ``None`` so the REPL forwards them to the agent loop instead.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(
        self, name: str, handler: Any, *, summary: str = "",
    ) -> None:
        """Map ``name`` to ``handler``. Re-registering replaces silently."""
        self._entries[name] = _Entry(
            handler=handler,
            summary=summary,
            takes_args=_handler_takes_args(handler),
        )

    def names(self) -> list[str]:
        return sorted(self._entries)

    async def dispatch(
        self, line: str, *, ctx: SlashContext | None = None,
    ) -> str | None:
        """Route ``line`` to its handler. Returns the handler's message
        (``""`` if the handler returned ``None``), or ``None`` if ``line``
        is not a slash command (REPL should forward to the agent).

        Unknown ``/cmd`` produces a local listing of registered commands.
        """
        if not line.startswith("/"):
            return None
        ctx = ctx if ctx is not None else SlashContext()
        name, _, args = line[1:].partition(" ")
        name = name.strip()
        if not name:
            return None
        if name == "help":
            return self._render_help()
        entry = self._entries.get(name)
        if entry is None:
            return self._render_unknown(name)
        ctx.args = args
        result = (
            await entry.handler(ctx, args) if entry.takes_args
            else await entry.handler(ctx)
        )
        return "" if result is None else result

    def _render_help(self) -> str:
        if not self._entries:
            return "No slash commands registered."
        lines = ["/help — show this listing"]
        for name in self.names():
            summary = self._entries[name].summary
            tail = f" — {summary}" if summary else ""
            lines.append(f"/{name}{tail}")
        return "\n".join(lines)

    def _render_unknown(self, name: str) -> str:
        if not self._entries:
            return f"unknown slash command: /{name} (no commands registered)"
        lines = [f"unknown slash command: /{name}", "available:"]
        for n in self.names():
            summary = self._entries[n].summary
            tail = f" — {summary}" if summary else ""
            lines.append(f"  /{n}{tail}")
        return "\n".join(lines)
