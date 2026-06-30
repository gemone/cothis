"""Command-line interface for cothis."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

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
from cothis.tools import TOOLS, Tool, load_tools_from_layer

app = typer.Typer()
console = Console()
logger = logging.getLogger("cothis.cli")

# cothis: the system prompt is hardcoded (not user-configurable) and identical
# across ``ask`` and ``chat``. Both commands share the same persona so the
# behavior a user learns in one mode transfers to the other. Ceiling: no
# env var / flag override today. Upgrade path: add ``--system-prompt`` /
# ``COTHIS_SYSTEM_PROMPT`` and fall back to this constant.
#
# cothis: the prompt deliberately does NOT name tools. Which tools are
# available is surfaced to the model purely via the ``tools=`` schemas
# passed to the completion API — naming them here is redundant and drifts
# the moment a YAML/Python tool is added or a built-in is removed. The
# model learns its capabilities from the tool schemas, not the prompt.
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. Use the tools you are given "
    "to inspect and modify files and run commands as needed."
)

# Set by the root callback's --debug option. Consumed by main() to decide
# whether to surface the full traceback.
_debug = False

# Discovery paths (see CONTEXT.md "Discovery path"). Project-local is
# cwd-relative; user-global is the XDG-style config directory. Both are
# optional — ``_all_tools`` handles missing dirs without error.
_PROJECT_TOOLS_DIR = Path(".agents/tools")
_USER_TOOLS_DIR = Path.home() / ".config" / "cothis" / "tools"


def _all_tools(project_dir: Path, user_dir: Path) -> list[Tool]:
    """Built-in tools plus any declared in the two discovery layers.

    Loads YAML and Python tool declarations from ``user_dir`` (user-global)
    and ``project_dir`` (project-local). Both are optional; absence is not
    an error. Each directory is one **layer** (see CONTEXT.md "Layer").

    cothis: ceiling — cross-layer name conflicts (user-global vs project-local,
    or custom vs builtin) currently raise ``ValueError``. #10 and #11 replace
    this with shadow semantics (project-local shadows user-global shadows
    builtin, with a ``WARNING`` instead of a raise). The per-layer loader
    (``load_tools_from_layer``) already catches same-layer conflicts (any
    format combination in the same directory).

    Lifecycle hooks (``pre_load`` / ``after_load``) run AFTER loading, on each
    tool that survives the duplicate check — a ``pre_load=False`` or hook
    exception drops the tool (``on_error`` fires for audit, ``WARNING``
    logged). See ADR-0003.
    """
    from cothis.tools import _check_duplicate_name

    user_tools = load_tools_from_layer(user_dir)
    project_tools = load_tools_from_layer(project_dir)
    all_tools: list[Tool] = [*TOOLS, *user_tools, *project_tools]

    # Cross-source duplicate check (ceiling — see docstring).
    seen: dict[str, str] = {}
    for tool in all_tools:
        source = getattr(tool, "_source", None) or "builtins"
        _check_duplicate_name(tool, source, seen)

    # Run load hooks on each surviving tool (post-resolution, pre-registration).
    # Bare callables (no ``_run_load_hooks`` attr) skip hooks entirely.
    registered: list[Tool] = []
    for tool in all_tools:
        run_hooks = getattr(tool, "_run_load_hooks", None)
        if run_hooks is None or run_hooks():
            registered.append(tool)

    logger.warning("discovery: %d tools active", len(registered))
    return registered


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
    """cothis — a basic any-llm agent loop."""
    global _debug
    _debug = debug
    # ``--debug`` = everything (cothis + openai + httpx + traceback).
    # ``-v`` / ``--verbose`` = cothis only (tool-call I/O, gating skips) —
    # the signal you actually want when checking what reached the model,
    # without the HTTP/TLS noise swamping it.
    if debug or verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    if verbose and not debug:
        # Quiet the chatty downstream loggers; keep ``cothis.*`` at DEBUG.
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
            tools=_all_tools(_PROJECT_TOOLS_DIR, _USER_TOOLS_DIR),
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
            tools=_all_tools(_PROJECT_TOOLS_DIR, _USER_TOOLS_DIR),
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
        except EOFError, KeyboardInterrupt:
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
                # Back to thinking state: tear down Live if we were streaming,
                # restart the spinner, print the tool call above it.
                if live is not None:
                    live.stop()
                    live = None
                    accumulated = ""
                    status.start()
                console.print(_format_tool_call(event), style="dim")
                continue
            # Content delta. First one of a fresh streaming phase: stop the
            # spinner, spin up Live. Subsequent ones just update it.
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
