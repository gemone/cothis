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
from rich.console import Console
from rich.markdown import Markdown

from cothis.agent import Agent
from cothis.tools import TOOLS

app = typer.Typer()
console = Console()

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
    # each spinner's text disappears on exit and the Markdown answer is the
    # only thing left on screen.
    with console.status("loading...", spinner="dots"):
        agent = Agent(
            model=model,
            provider=provider,
            tools=TOOLS,
            system_prompt=(
                "You are a concise, helpful assistant with filesystem access. "
                "Use the `fs.read` and `fs.write` tools to inspect and modify files."
            ),
            max_iterations=max_iterations,
        )
    with console.status("thinking...", spinner="dots"):
        answer = asyncio.run(agent.run(prompt))
    console.print(Markdown(answer))


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
