"""Command-line interface for cothis."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# Must run before cothis.agent imports any_llm.
os.environ.setdefault("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

import click
import typer
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from cothis.agent import Agent, ToolCallEvent
from cothis.tools import discover_tools

app = typer.Typer()
console = Console()

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. Use the tools you are given "
    "to inspect and modify files and run commands as needed."
)

_debug = False

_PROJECT_TOOLS_DIR = Path(".agents/tools")
# Empty/unset ``COTHIS_HOME`` → default ``~/.cothis``.
_COTHIS_HOME = Path(
    os.environ.get("COTHIS_HOME") or Path.home() / ".cothis"
).expanduser()
_USER_TOOLS_DIR = _COTHIS_HOME / "tools"


@app.callback()
def _root(
    debug: bool = typer.Option(
        False,
        "--debug",
        envvar="DEBUG",
        help="Show full tracebacks + all debug logs (cothis, openai, httpx).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        envvar="VERBOSE",
        help="Show cothis tool-call I/O (without openai/httpx noise). Implied by --debug.",
    ),
) -> None:
    """cothis — an any-llm agent loop."""
    global _debug
    _debug = debug
    if debug or verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    if verbose and not debug:
        for noisy in ("openai", "httpx", "httpcore", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.INFO)


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
    with console.status("loading...", spinner="dots"):
        agent = Agent(
            model=model,
            provider=provider,
            tools=discover_tools(_PROJECT_TOOLS_DIR, _USER_TOOLS_DIR),
            system=DEFAULT_SYSTEM_PROMPT,
            max_iterations=max_iterations,
        )
    with console.status("thinking...", spinner="dots"):
        answer = asyncio.run(_run_and_close(agent, prompt))
    typer.echo(answer)


async def _run_and_close(agent: Agent, prompt: str) -> str:
    """Run one ``ask`` turn and close MCP sessions afterwards.

    ``ask`` discards the Agent after a single run, so any MCP subprocesses it
    started must be shut down here (no long-lived session to reuse them).
    """
    try:
        return await agent.run(prompt)
    finally:
        await agent.aclose()


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
    with console.status("loading...", spinner="dots"):
        agent = Agent(
            model=model,
            provider=provider,
            tools=discover_tools(_PROJECT_TOOLS_DIR, _USER_TOOLS_DIR),
            system=DEFAULT_SYSTEM_PROMPT,
            max_iterations=max_iterations,
        )

    # prompt_toolkit over stdlib ``input()``: CPython auto-loads GNU readline
    # for ``input``, which mis-counts CJK / wide-char column width and leaves
    # visual residue on backspace. prompt_toolkit does its own ``wcwidth``
    # accounting and renders the line itself.
    #
    # ``prompt_async`` (not sync ``prompt`` via ``asyncio.to_thread``): the
    # latter races interpreter shutdown on Ctrl-C — the worker stays blocked
    # on stdin while the main thread unwinds, producing a noisy traceback.
    session = PromptSession()
    try:
        while True:
            try:
                prompt_text = await session.prompt_async(">>> ")
            except EOFError, KeyboardInterrupt:
                console.print()
                break
            if not prompt_text.strip():
                continue

            await _stream_answer(agent, prompt_text)
    finally:
        await agent.aclose()


async def _stream_answer(agent: Agent, prompt: str) -> None:
    """Run one turn of the agent and stream the final answer as Markdown.

    Event protocol from ``Agent.run_stream``:
      * ``ToolCallEvent``  — printed inline (``calling fs.read(...)``) so the
        user can see why a multi-step turn is taking time. Printed *above*
        the spinner's animation row, which rich's Status handles cleanly.
      * ``str``            — a content delta of the final answer.

    The ReAct loop is multi-turn: tool-call turns and content turns alternate.
    This consumer drives a two-state display:
      * ``thinking``  — spinner running; ToolCallEvents printed inline.
      * ``streaming`` — spinner stopped; a Live Markdown view re-renders as
        content deltas arrive.
    Transitions happen per-event, not per-turn, because a single provider
    turn can interleave tool calls and content. The consumer must drain
    the *whole* generator — closing it early (the old ``break`` + ``return``
    shape) truncated the ReAct loop the moment a tool-call-only turn had no
    content delta to show, so the agent stopped after one tool call.
    """
    stream = agent.run_stream(prompt)
    status = console.status("thinking...", spinner="dots")
    live: Live | None = None
    accumulated = ""
    status.start()
    try:
        async for event in stream:
            if isinstance(event, ToolCallEvent):
                if live is not None:
                    live.stop()
                    live = None
                    accumulated = ""
                    status.start()
                console.print(_format_tool_call(event), style="dim")
                continue
            # Content delta — first one spins up Live, subsequent ones update it.
            if live is None:
                status.stop()
                accumulated = event
                live = Live(
                    Markdown(accumulated), console=console, refresh_per_second=10
                )
                live.start()
            else:
                accumulated += event
                live.update(Markdown(accumulated))
    finally:
        if live is not None:
            live.stop()
            console.print()
        else:
            status.stop()


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
        typer.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
