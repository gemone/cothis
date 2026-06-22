"""Command-line interface for cothis."""

from __future__ import annotations

import asyncio
import os
import sys

# Enable any-llm's unified exception hierarchy so provider-specific errors
# are converted into any_llm.exceptions.* regardless of which provider
# the user picks.
# Must run before cothis.agent imports any_llm.
os.environ.setdefault("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

import click
import typer
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from cothis.agent import Agent, ToolCallEvent
from cothis.tools import TOOLS

app = typer.Typer()
console = Console()

# cothis: the system prompt is hardcoded (not user-configurable) and identical
# across ``ask`` and ``chat``. Both commands share the same persona so the
# behavior a user learns in one mode transfers to the other. Ceiling: no
# env var / flag override today. Upgrade path: add ``--system-prompt`` /
# ``COTHIS_SYSTEM_PROMPT`` and fall back to this constant.
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, helpful assistant with filesystem access. "
    "Use the `fs.read` and `fs.write` tools to inspect and modify files."
)

# Set by the root callback's --debug option. Consumed by main() to decide
# whether to surface the full traceback.
_debug = False


@app.callback()
def _root(
    debug: bool = typer.Option(
        False,
        "--debug",
        envvar="DEBUG",
        help="Show full tracebacks on error (default: suppressed).",
    ),
) -> None:
    """cothis — a basic any-llm agent loop."""
    global _debug
    _debug = debug


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="The user prompt to send to the agent."),
    provider: str = typer.Option(
        "openrouter",
        "--provider",
        "-p",
        envvar="COTHIS_PROVIDER",
        help="any-llm provider key (e.g. openrouter, mistral, openai, anthropic).",
    ),
    model: str = typer.Option(
        "openai/gpt-oss-120b",
        "--model",
        "-m",
        envvar="COTHIS_MODEL",
        help="Model identifier for the chosen provider.",
    ),
    max_iterations: int = typer.Option(
        30, "--max-iterations", help="LLM round-trip cap."
    ),
) -> None:
    """Run the agent once and print its final answer."""
    # cothis: two-phase status — loading covers lazy any_llm import + Agent
    # construction; thinking covers the full run() loop (LLM calls + tool
    # execution). rich's Status wraps a Live(transient=True) internally, so
    # each spinner's text disappears on exit and the plain-text answer is
    # the only thing left on screen.
    #
    # The answer goes through ``typer.echo`` (not ``console.print``) so ask
    # stays pipe-friendly: ``cothis ask "..." | jq`` / ``> file`` see clean
    # stdout with no ANSI escape codes. Markdown rendering is reserved for
    # the interactive ``chat`` command.
    with console.status("loading...", spinner="dots"):
        agent = Agent(
            model=model,
            provider=provider,
            tools=TOOLS,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_iterations=max_iterations,
        )
    with console.status("thinking...", spinner="dots"):
        answer = asyncio.run(agent.run(prompt))
    typer.echo(answer)


@app.command()
def chat(
    provider: str = typer.Option(
        "openrouter",
        "--provider",
        "-p",
        envvar="COTHIS_PROVIDER",
        help="any-llm provider key (e.g. openrouter, mistral, openai, anthropic).",
    ),
    model: str = typer.Option(
        "openai/gpt-oss-120b",
        "--model",
        "-m",
        envvar="COTHIS_MODEL",
        help="Model identifier for the chosen provider.",
    ),
    max_iterations: int = typer.Option(
        30, "--max-iterations", help="LLM round-trip cap."
    ),
) -> None:
    """Run an interactive multi-turn chat session.

    One Agent instance is reused across turns, so conversation history
    accumulates. The final answer of each turn is streamed token-by-token
    and rendered live as Markdown; intermediate tool-calling turns are
    covered by a ``thinking...`` spinner (no per-tool status today).
    """
    asyncio.run(
        _chat_session(
            model=model,
            provider=provider,
            max_iterations=max_iterations,
        )
    )


async def _chat_session(
    *,
    model: str,
    provider: str,
    max_iterations: int,
) -> None:
    # One event loop owns the whole session so any cross-turn async state
    # inside AnyLLM (HTTP keep-alive, client caches) stays bound to the
    # same loop. ``ask`` doesn't need this because it discards the Agent
    # after one ``run``, but ``chat`` reuses it.
    with console.status("loading...", spinner="dots"):
        agent = Agent(
            model=model,
            provider=provider,
            tools=TOOLS,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_iterations=max_iterations,
        )

    # PromptSession is the native async entry into prompt_toolkit. We need
    # ``prompt_async`` rather than the sync ``prompt()`` because we are
    # already inside an event loop; the sync call internally drives its own
    # loop and errors out with "asyncio.run() cannot be called from a
    # running event loop".
    #
    # We use prompt_toolkit instead of stdlib ``input()`` because CPython
    # auto-loads GNU readline for ``input``, and readline mis-counts the
    # terminal column width of CJK / wide chars — backspacing over a
    # Chinese character leaves visual residue on screen (the received
    # string is correct, but the display looks broken). prompt_toolkit does
    # its own width accounting via ``wcwidth`` and renders the input line
    # itself, so delete is clean for any script. Also gives us history /
    # Emacs keys for free.
    #
    # We deliberately do *not* run the sync ``prompt`` via
    # ``asyncio.to_thread``. The chat loop is strictly serial
    # (input → stream → input), so nothing concurrent benefits from the
    # offload, and wrapping it in a worker thread plus pressing Ctrl-C
    # repeatedly races interpreter shutdown — the worker stays blocked on
    # stdin while the main thread unwinds, producing a noisy
    # ``KeyboardInterrupt`` traceback from the atexit join. Calling
    # ``prompt_async`` lets SIGINT route through the loop's own signal
    # handling, which prompt_toolkit handles cleanly.
    session = PromptSession()
    while True:
        try:
            prompt_text = await session.prompt_async(">>> ")
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D / Ctrl-C at the prompt: end the session quietly.
            # Execution-mid Ctrl-C still bubbles up through main().
            console.print()
            break
        if not prompt_text.strip():
            continue

        await _stream_answer(agent, prompt_text)


async def _stream_answer(agent: Agent, prompt: str) -> None:
    """Run one turn of the agent and stream the final answer as Markdown.

    Event protocol from ``Agent.run_stream``:
      * ``ToolCallEvent``  — printed inline (``calling fs.read(...)``) so the
        user can see why a multi-step turn is taking time. Printed *above*
        the spinner's animation row, which rich's Status handles cleanly.
      * ``str``            — a content delta of the final answer. The first
        content delta stops the spinner and switches to a Live-rendered
        Markdown view that re-formats as tokens arrive.
    """
    stream = agent.run_stream(prompt)
    status = console.status("thinking...", spinner="dots")
    status.start()
    try:
        first_content: str | None = None
        async for event in stream:
            if isinstance(event, ToolCallEvent):
                # Tool calls are printed inline while the spinner runs. The
                # spinner is transient (rich's Status uses Live(transient=True)
                # internally) so once it stops, only these printed lines and
                # the final answer remain on screen.
                console.print(_format_tool_call(event), style="dim")
                continue
            # First content delta: stop the spinner, switch to Live rendering.
            first_content = event
            break
    finally:
        status.stop()

    if first_content is None:
        # Stream ended with only tool-call events (or nothing at all).
        # Nothing to Live-render; the printed tool calls already left a trail.
        return

    # Live re-renders the whole Markdown on each update so code blocks,
    # lists etc. format progressively. Live's own __exit__ leaves the final
    # frame on screen, which is exactly what we want.
    accumulated = first_content
    with Live(Markdown(accumulated), console=console, refresh_per_second=10) as live:
        async for chunk in stream:
            # ``run_stream`` only yields ToolCallEvents on intermediate turns
            # and the loop above already drained those before the first
            # content delta, so we never expect a ToolCallEvent here. The
            # isinstance guard is still required: ty can't prove narrowing
            # across the async-for boundary, and without it ``accumulated +=
            # chunk`` doesn't typecheck (str + ToolCallEvent is invalid).
            if isinstance(chunk, ToolCallEvent):
                continue
            accumulated += chunk
            live.update(Markdown(accumulated))
    console.print()


def _format_tool_call(event: ToolCallEvent) -> str:
    """One-line human-readable summary of a tool call.

    Uses ``repr`` for values so strings stay quoted and distinguishable
    from numbers in the printed output (``fs.read(path="/x")`` vs
    ``fs.read(path=/x)``).
    """
    args = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
    return f"calling {event.name}({args})"


def main() -> None:
    """Console-script entry point.

    Runs the typer app with ``standalone_mode=False`` so we can decide
    ourselves whether to surface tracebacks. Click's own usage/abort
    errors are still formatted nicely; everything else is printed as
    ``Error: <message>`` (no traceback) unless ``--debug`` is set.
    """
    try:
        app(standalone_mode=False)
    except click.ClickException as exc:
        # Usage / bad-parameter errors — Click formats these itself.
        exc.show()
        sys.exit(exc.exit_code)
    except click.Abort:
        typer.echo("Aborted!", err=True)
        sys.exit(1)
    except SystemExit:
        raise
    except BaseException as exc:
        if _debug:
            raise
        # Still tell the user what went wrong — just without the traceback.
        typer.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
