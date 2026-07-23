"""Command-line interface for cothis."""

from __future__ import annotations

import asyncio
import gzip
import logging
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

# Must run before cothis.agent imports any_llm.
os.environ.setdefault("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

# If COTHIS_PROFILE_STARTUP is set, re-exec under -X importtime and exit
# before any third-party import runs. Imports only stdlib so the
# measurement cost is negligible when the flag is unset.
from cothis._profile_startup import maybe_profile

maybe_profile()

import click  # cost: ~5ms
import typer  # cost: ~30ms (loads click + shell completion)
from prompt_toolkit.shortcuts import PromptSession  # cost: ~40ms
from rich.console import Console  # cost: ~15ms
from rich.live import Live  # cost: ~5ms
from rich.markdown import Markdown  # cost: ~5ms

from cothis.agent import Agent, MaxIterationsError, ToolCallEvent
from cothis.session import (
    Session,
    SessionHasChildrenError,
    SessionLockedError,
)
from cothis.session.archive import (
    ArchiveIndex,
    archive_session,
    promote_session,
    run_archival_pass,
)
from cothis.session.storage import Storage, display_cwd, is_visible
from cothis.tools import discover_tools

app = typer.Typer()
console = Console()

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. Use the tools you are given "
    "to inspect and modify files and run commands as needed."
)

_debug = False

_PROJECT_TOOLS_DIR = Path(".agents/tools")


# cothis: ``_cothis_home`` / ``_user_tools_dir`` read ``$COTHIS_HOME``
# lazily per call (#66). Mirrors the lazy pattern used by
# ``_resolve_db_path``.
def _cothis_home() -> Path:
    """``$COTHIS_HOME`` or ``~/.cothis``. Read lazily per call."""
    return Path(
        os.environ.get("COTHIS_HOME") or Path.home() / ".cothis"
    ).expanduser()


def _user_tools_dir() -> Path:
    """``$COTHIS_HOME/tools``. Read lazily per call."""
    return _cothis_home() / "tools"


# cothis: defense-in-depth hex-32 validation at the CLI boundary. The
# storage layer (``Session._validate_session_id``) already enforces
# this, so the check here never changes behaviour for well-formed
# input; it gives a friendlier error than a deep FK constraint miss.
def _validate_session_id_arg(sid: str) -> None:
    if len(sid) != 32 or not all(c in "0123456789abcdef" for c in sid):
        raise typer.BadParameter(
            f"expected a 32-char lowercase hex session id; got {sid!r}"
        )


def _resolve_db_path() -> Path:
    """Resolve the SQLite db path for session persistence.

    Three modes (highest precedence first):

    1. ``COTHIS_SESSIONS_TYPE=project`` → ``<cwd>/.agents/sessions/session.db``
       (split layout — db lives in the project, sessions scoped per-project).
    2. ``COTHIS_SESSIONS_DIR=<path>`` → ``<path>/session.db``
       (split layout at a caller-chosen location).
    3. neither set → ``$COTHIS_HOME/agents.db``
       (default single-file layout — all sessions in one global db).

    Lock files live elsewhere (``$XDG_CACHE_HOME/cothis/<id>.lock``) and are
    resolved inside ``Session``; this function only owns the db path. Split
    modes share the ``session.db`` filename to distinguish them from the
    default ``agents.db`` (which is the unified entry the user sees by
    default and may eventually hold config/audit tables too).
    """
    if os.environ.get("COTHIS_SESSIONS_TYPE") == "project":
        return Path.cwd() / ".agents" / "sessions" / "session.db"
    if dir_env := os.environ.get("COTHIS_SESSIONS_DIR"):
        return Path(dir_env).expanduser() / "session.db"
    return _cothis_home() / "agents.db"


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
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        envvar="COTHIS_MAX_TOKENS",
        help="Output-token cap. Default: resolved from bundled litellm metadata for the model.",
    ),
) -> None:
    """Run the agent once and print its final answer."""
    with console.status("loading...", spinner="dots"):
        agent = Agent(
            model=model,
            provider=provider,
            tools=discover_tools(_PROJECT_TOOLS_DIR, _user_tools_dir()),
            system=DEFAULT_SYSTEM_PROMPT,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            cwd=Path.cwd(),
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
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        envvar="COTHIS_MAX_TOKENS",
        help="Output-token cap. Default: resolved from bundled litellm metadata for the model.",
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume a session by id (shortcut to the end of main; no picker).",
    ),
    skill: list[str] = typer.Option(
        [],
        "--skill",
        "-s",
        help=(
            "Pre-activate a skill at session start (repeatable). "
            "Synthesises a load_skill pair after the first user message."
        ),
    ),
) -> None:
    """Run an interactive multi-turn chat session.

    One Agent instance is reused across turns, so conversation history
    accumulates. The final answer of each turn is streamed token-by-token
    and rendered live as Markdown; intermediate tool-calling turns are
    covered by a ``thinking...`` spinner (no per-tool status today).

    ``--resume <id>`` shortcuts to the end of ``main``: no interactive
    picker. Errors with "not found, run ``cothis history``" if the id is
    missing or out of this directory's scope.
    """
    asyncio.run(
        _chat_session(
            model=model,
            provider=provider,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            resume=resume,
            preactivate_skills=list(skill),
        )
    )


async def _chat_session(
    *,
    model: str,
    provider: str,
    max_iterations: int,
    max_tokens: int | None,
    resume: str | None = None,
    preactivate_skills: list[str] | None = None,
) -> None:
    # ``chat`` is the only command that persists. ``Session.new`` takes the
    # cross-process lock eagerly; the sessions row + title are written
    # lazily on the first user message's drain. ``ask`` constructs no
    # Session (ephemeral). ``SessionLockedError`` from new() propagates
    # through asyncio.run → main()'s BaseException handler → "Error: …" +
    # exit 1.
    db_path = _resolve_db_path()
    if resume is not None:
        _validate_session_id_arg(resume)
        # Resume path: load by id (errors out cleanly if missing). The
        # cwd filter is enforced inside Session.load via the storage
        # row's cwd; the picker in ``history <id>`` already did the
        # scoping, so we don't re-check here.
        try:
            session = Session.load(db_path, resume, cwd=Path.cwd())
        except KeyError:
            raise typer.BadParameter(
                f"session {resume!r} not found; run `cothis history` to list"
            )
    else:
        session = Session.new(db_path, cwd=Path.cwd(), model=model)
    try:
        with console.status("loading...", spinner="dots"):
            agent = Agent(
                model=model,
                provider=provider,
                tools=discover_tools(_PROJECT_TOOLS_DIR, _user_tools_dir()),
                system=DEFAULT_SYSTEM_PROMPT,
                max_iterations=max_iterations,
                max_tokens=max_tokens,
                cwd=Path.cwd(),
                preactivate_skills=preactivate_skills or [],
            )
            agent.attach_session(session)

        # prompt_toolkit over stdlib ``input()``: CPython auto-loads GNU readline
        # for ``input``, which mis-counts CJK / wide-char column width and leaves
        # visual residue on backspace. prompt_toolkit does its own ``wcwidth``
        # accounting and renders the line itself.
        #
        # ``prompt_async`` (not sync ``prompt`` via ``asyncio.to_thread``): the
        # latter races interpreter shutdown on Ctrl-C — the worker stays blocked
        # on stdin while the main thread unwinds, producing a noisy traceback.
        prompts = PromptSession()
        try:
            while True:
                try:
                    prompt_text = await prompts.prompt_async(">>> ")
                except EOFError, KeyboardInterrupt:
                    console.print()
                    break
                if not prompt_text.strip():
                    continue

                await _stream_answer(agent, prompt_text)
        finally:
            await agent.aclose()
    finally:
        # Idempotent: if attach succeeded, ``agent.aclose()`` above already
        # closed the session (drained + joined + storage closed). If Agent
        # construction failed before attach, this is the cleanup path.
        session.close()


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
    has_max_iterations_error = False
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
    except MaxIterationsError as exc:
        has_max_iterations_error = True
        if live is not None:
            live.stop()
        else:
            status.stop()
        console.print(f"[red]Error:[/red] {exc}")
        return
    finally:
        if has_max_iterations_error:
            pass  # already handled in except
        elif live is not None:
            live.stop()
            console.print()
        elif accumulated:
            console.print(Markdown(accumulated))
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


# ---------------------------------------------------------------------
# history / delete commands
# ---------------------------------------------------------------------


def _preview_message(msg: dict) -> str:
    """One-line preview for ``cothis history <id>``'s listing.

    Shows the first text block's first line, prefixed by role. Falls
    back to a block-type summary when no text is present (e.g. a
    pure ``tool_result`` user message).
    """
    role = msg.get("role", "?")
    for block in msg.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            text = block["text"].splitlines()[0]
            return f"[{role}] {text[:80]}"
    types = sorted({b.get("type", "?") for b in msg.get("content", [])})
    return f"[{role}] <{','.join(types)}>"


def _print_history_listing(rows: list) -> None:
    """Print ``id  timestamp  cwd  title`` rows, one per line."""
    if not rows:
        console.print("[dim]no sessions in this directory's scope[/dim]")
        return
    cwd = Path.cwd()
    for row in rows:
        title = row.title or "(no title)"
        cwd_display = display_cwd(Path(row.cwd), cwd)
        typer.echo(f"{row.id}  {row.updated_at}  {cwd_display}  {title}")


@app.command()
def history(
    session_id: str | None = typer.Argument(
        None, help="Show this session's messages and pick a resume/fork point."
    ),
) -> None:
    """List sessions visible from the current directory, or inspect one.

    Without an argument: list every session whose ``cwd`` is the current
    directory or an ancestor of it (project-root sessions are visible
    from subdirectories). Each row shows ``id, updated_at, cwd, title``.

    With an argument: print the session's full message list numbered,
    then prompt for an index. ``r`` (or Enter on the last) resumes at
    the end of ``main``; a number forks a new session from that message
    (git-branch semantics — the original is untouched).
    """
    db_path = _resolve_db_path()
    if not db_path.exists():
        console.print("[dim]no sessions database yet[/dim]")
        return
    if session_id is None:
        rows = Session.list_visible(db_path, Path.cwd())
        _print_history_listing(rows)
        return
    # Inspect one: peek_messages enforces the cwd visibility filter when
    # cwd= is passed, so the picker refuses out-of-scope sessions the same
    # way Session.load(cwd=...) does.
    _validate_session_id_arg(session_id)
    try:
        messages = Session.peek_messages(db_path, session_id, cwd=Path.cwd())
    except KeyError:
        raise typer.BadParameter(
            f"session {session_id!r} not found (or not in this directory's scope); "
            f"run `cothis history` to list"
        )
    if not messages:
        console.print("[dim]session is empty[/dim]")
        return
    for i, msg in enumerate(messages):
        console.print(f"[magenta]{i:3d}[/magenta]  {_preview_message(msg)}")
    console.print()
    choice = console.input(
        "[bold]r[/bold]esume at end, [bold]<n>[/bold] to fork at message n, [bold]q[/bold]uit > "
    ).strip().lower()
    if choice in ("", "r", "q"):
        if choice == "q":
            return
        console.print(
            f"run [cyan]cothis chat --resume {session_id}[/cyan] to continue"
        )
        return
    try:
        idx = int(choice)
    except ValueError:
        raise typer.BadParameter(f"expected a number or 'r', got {choice!r}")
    if not 0 <= idx < len(messages):
        raise typer.BadParameter(f"index {idx} out of range (0..{len(messages) - 1})")
    # Map the in-memory message index back to the storage seq cap. Each message
    # occupies one distinct msg_idx; the seq cap is the max seq across
    # that message's blocks.
    storage = Storage(db_path)
    try:
        idx_to_max_seq = storage.msg_idx_to_max_seq(session_id)
        sr = storage.load_session(session_id)
        model = (sr.model if sr is not None else "") or ""
    finally:
        storage.close()
    msg_idxs = sorted(idx_to_max_seq)
    target_msg_idx = msg_idxs[idx]
    cap = idx_to_max_seq[target_msg_idx]
    forked = Session.fork(
        db_path,
        session_id,
        cap,
        cwd=Path.cwd(),
        model=model,
    )
    try:
        forked_id = forked.session_id
    finally:
        forked.close()
    console.print(
        f"forked as [cyan]{forked_id}[/cyan]; "
        f"run [cyan]cothis chat --resume {forked_id}[/cyan] to continue"
    )


@app.command(name="delete")
def delete_cmd(
    session_id: str = typer.Argument(..., help="Session id to delete (must be a leaf)."),
) -> None:
    """Delete a session from the local or cold database.

    Refuses if the session has any forked children — deleting a non-leaf
    node would orphan them. Delete the children first (use
    ``cothis history`` to find them). Leaf-only check spans both hot
    and cold DBs (#87).
    """
    _validate_session_id_arg(session_id)
    try:
        Session.delete(_resolve_db_path(), session_id)
    except SessionHasChildrenError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"deleted session [cyan]{session_id}[/cyan]")


@app.command(name="archive")
def archive_cmd(
    action: str = typer.Argument(
        "all", help="'all' (default), '<session_id>', 'restore <id>', 'compress <file>'"
    ),
    target: str = typer.Argument(
        None, help="Session id (for restore) or file path (for compress)."
    ),
) -> None:
    """Archive, restore, or compress sessions.

    \b
    Examples:
        cothis archive              # archive all idle sessions
        cothis archive <session_id> # archive one session
        cothis archive restore <id> # promote archived session back
        cothis archive compress <file>  # gzip a cold DB file
    """
    # cothis: hand-rolled dispatch instead of nested typer.Typer() because
    # the first positional arg is either a subcommand (restore/compress)
    # or a session id — Typer's subcommand model can't express that
    # ambiguity. Nested Typer would force `cothis archive session <id>`,
    # adding a word to the common path.
    db_path = _resolve_db_path()
    archive_dir = db_path.parent / "archive"

    if action == "all":
        now_iso = datetime.now(UTC).isoformat()
        archived = run_archival_pass(
            hot_db_path=db_path,
            archive_dir=archive_dir,
            threshold_days=90,
            now_iso=now_iso,
        )
        if archived == 0:
            console.print("no sessions to archive")
        else:
            console.print(f"archived {archived} session(s)")
    elif action == "restore":
        if not target:
            raise typer.BadParameter("restore requires a session id")
        _validate_session_id_arg(target)
        index = ArchiveIndex(archive_dir / "index.json")
        ok = promote_session(
            hot_db_path=db_path,
            archive_dir=archive_dir,
            session_id=target,
            index=index,
        )
        if ok:
            console.print(f"restored session [cyan]{target}[/cyan]")
        else:
            raise typer.BadParameter(
                f"session {target!r} not found in archive index"
            )
    elif action == "compress":
        if not target:
            raise typer.BadParameter("compress requires a file path")
        if not target.lower().endswith(".db"):
            raise typer.BadParameter(f"file must end in .db: {target}")
        file_path = (archive_dir / target).resolve()
        # cothis: prevent path escape — compress must stay inside archive_dir.
        # TOCTOU: resolve() → exists() → open() has a symlink-swap window;
        # acceptable for single-user CLI (no adversary on the same fs).
        try:
            file_path.relative_to(archive_dir.resolve())
        except ValueError:
            raise typer.BadParameter(f"file must be inside {archive_dir}")
        if not file_path.exists():
            raise typer.BadParameter(f"no such file: {target}")
        out_path = file_path.with_suffix(file_path.suffix + ".gz")
        with file_path.open("rb") as src, gzip.open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        console.print(f"compressed to [cyan]{out_path.name}[/cyan]")
    else:
        _validate_session_id_arg(action)
        now_iso = datetime.now(UTC).isoformat()
        index = ArchiveIndex(archive_dir / "index.json")
        # cothis: surface missing-id as BadParameter (#121). Previously
        # the CLI unconditionally printed success even when the session
        # wasn't in the hot DB (archive_session no-ops silently).
        result = archive_session(
            hot_db_path=db_path,
            archive_dir=archive_dir,
            session_id=action,
            archive_db_name=f"{now_iso[:7]}.db",
            archived_at=now_iso,
            index=index,
        )
        if result is None:
            raise typer.BadParameter(
                f"session {action!r} not found in hot db; "
                f"did you mean 'cothis archive restore {action}'? "
                f"run 'cothis history' to list hot sessions"
            )
        console.print(f"archived session [cyan]{action}[/cyan]")


def main() -> None:
    """Console-script entry point.

    Runs the typer app with ``standalone_mode=False`` so we can decide
    ourselves whether to surface tracebacks. Click's own usage/abort
    errors are still formatted nicely; everything else is printed as
    ``Error: <message>`` (no traceback) unless ``--debug`` is set.

    KeyboardInterrupt (Ctrl-C) is handled explicitly: silent exit with
    the POSIX-conventional code 130 (128 + SIGINT), or re-raised under
    ``--debug`` so the traceback surfaces.
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
    except KeyboardInterrupt:
        # POSIX convention: SIGINT → exit status 130 (128 + 2). Silent
        # unless --debug, mirroring git/ssh/python -c.
        # _chat_session's inner prompt handler exits silently on Ctrl-C;
        # this branch mirrors that contract for the streaming path.
        if _debug:
            raise
        sys.exit(130)
    except Exception as exc:
        if _debug:
            raise
        typer.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
