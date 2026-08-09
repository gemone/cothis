"""Command-line interface for cothis."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

# If COTHIS_PROFILE_STARTUP is set, re-exec under -X importtime and exit
# before any third-party import runs. Imports only stdlib so the
# measurement cost is negligible when the flag is unset.
from cothis.profile_startup import maybe_profile

maybe_profile()

import click  # cost: ~5ms
import typer  # cost: ~30ms (loads click + shell completion)
from rich.console import Console  # cost: ~15ms

from cothis.agent import Agent, MaxIterationsError, ToolCallEvent, ToolResultEvent
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
from cothis.session.storage import SearchHit, Storage, display_cwd, is_visible
from cothis.tools import discover_tools

if TYPE_CHECKING:
    from cothis.protocol.acp import ACPServer

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


def _check_resume_exists(db_path: Path, resume: str) -> None:
    """Fail fast if a resume id doesn't exist before launching the TUI (#394).

    The TUI path validated only format, not existence — a well-formed-but-
    nonexistent id launched the TUI and spawned a doomed worker. Mirrors
    the legacy REPL's ``Session.load`` gate so both paths deliver the same
    "not found" message. The loaded session is closed immediately (the
    TUI's ``on_mount`` re-loads it).
    """
    try:
        Session.load(db_path, resume, cwd=Path.cwd()).close()
    except KeyError:
        raise typer.BadParameter(
            f"session {resume!r} not found; run `cothis history` to list"
        )


def _resolve_db_path(cwd: Path | None = None) -> Path:
    """Resolve the SQLite db path for session persistence.

    Three modes (highest precedence first):

    1. ``COTHIS_SESSIONS_TYPE=project`` → ``<cwd>/.agents/sessions/session.db``
       (split layout — db lives in the project, sessions scoped per-project).
    2. ``COTHIS_SESSIONS_DIR=<path>`` → ``<path>/session.db``
       (split layout at a caller-chosen location).
    3. neither set → ``$COTHIS_HOME/agents.db``
       (default single-file layout — all sessions in one global db).

    ``cwd`` defaults to the process cwd. Only project mode (1) is
    cwd-relative; passing an explicit ``cwd`` lets a caller resolve the db
    for a directory other than its own — the TUI uses this so a
    worktree-pick creates the session in the *worktree's* db, matching the
    worker subprocess that runs there (#402). The other two modes ignore
    ``cwd`` (path-based, not cwd-based).

    Lock files live elsewhere (``$XDG_CACHE_HOME/cothis/<id>.lock``) and are
    resolved inside ``Session``; this function only owns the db path. Split
    modes share the ``session.db`` filename to distinguish them from the
    default ``agents.db`` (which is the unified entry the user sees by
    default and may eventually hold config/audit tables too).
    """
    base = cwd if cwd is not None else Path.cwd()
    if os.environ.get("COTHIS_SESSIONS_TYPE") == "project":
        return base / ".agents" / "sessions" / "session.db"
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
    """cothis — a direct-SDK agent loop."""
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
        help="provider key (e.g. openrouter, mistral, openai, anthropic).",
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
    summary_model: str | None = typer.Option(
        None,
        "--summary-model",
        help=(
            "Summariser model (provider/model or bare model) used when "
            "compacting the conversation. Overrides COTHIS_SUMMARY_MODEL; "
            "when unset the env var is read inside the agent's "
            "resolve_summary_model, then the session pair."
        ),
    ),
    min_retained_turns: int = typer.Option(
        4,
        "--min-retained-turns",
        envvar="COTHIS_MIN_RETAINED_TURNS",
        help="Min turn-groups retained after compaction (default 4).",
    ),
    max_concurrent_tools: int = typer.Option(
        8,
        "--max-concurrent-tools",
        envvar="COTHIS_MAX_CONCURRENT_TOOLS",
        help=(
            "Max tool executions live at once within a single fan-out turn "
            "(default 8). Caps concurrent MCP/shell/network pressure on "
            "pathological fan-outs; normal turns (1-4 tools) are unaffected."
        ),
    ),
    max_tool_result_chars: int = typer.Option(
        20_000,
        "--max-tool-result-chars",
        envvar="COTHIS_MAX_TOOL_RESULT_CHARS",
        help=(
            "Max characters retained from a single tool result before it is "
            "truncated to a marker (default 20000). Caps prompt bloat from "
            "verbose tools; the full output is not recoverable once capped."
        ),
    ),
    tool_timeout: float | None = typer.Option(
        None,
        "--tool-timeout",
        envvar="COTHIS_TOOL_TIMEOUT",
        help=(
            "Per-tool wall-clock bound in seconds. Omit or leave unset for "
            "no timeout (the default, fully backward compatible). Must be "
            "> 0; interrupts async hangs at the next await point (a stalled "
            "MCP round-trip), not pure-sync blocking tool bodies."
        ),
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
            summary_model=summary_model,
            min_retained_turns=min_retained_turns,
            max_concurrent_tools=max_concurrent_tools,
            max_tool_result_chars=max_tool_result_chars,
            tool_timeout=tool_timeout,
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


@app.command(name="acp")
def acp(
    token: str = typer.Option(
        ...,
        "--token",
        envvar="COTHIS_ACP_TOKEN",
        help="Bearer token ACP clients must present in their hello handshake.",
    ),
    provider: str = typer.Option(
        "openrouter",
        "--provider",
        "-p",
        envvar="COTHIS_PROVIDER",
        help="provider key (e.g. openrouter, mistral, openai, anthropic).",
    ),
    model: str = typer.Option(
        "openai/gpt-oss-120b",
        "--model",
        "-m",
        envvar="COTHIS_MODEL",
        help="Model identifier for the chosen provider.",
    ),
) -> None:
    """Serve cothis over ACP on stdio (length-prefixed JSON frames).

    An editor (or any ACP client) spawns ``cothis acp --token ...`` and speaks
    the Agent Client Protocol over stdin/stdout: a ``hello`` handshake, then
    ``create`` / ``prompt`` commands. Assistant text + tool calls stream back
    as ``session_progress`` events. All diagnostics go to stderr; stdout
    carries only protocol frames.
    """
    from cothis.acp_bridge import AgentSessionBackend
    from cothis.protocol.acp import ACPServer

    backend = AgentSessionBackend(
        provider=provider,
        model=model,
        tools=discover_tools(_PROJECT_TOOLS_DIR, _user_tools_dir()),
        system=DEFAULT_SYSTEM_PROMPT,
    )
    server = ACPServer(backend, token=token)
    try:
        asyncio.run(_acp_stdio(server))
    except KeyboardInterrupt:
        pass


class _StdioByteConnection:
    """A :class:`ByteConnection` over process stdin/stdout.

    Reads inbound byte chunks off stdin in a thread (so the asyncio loop is
    never blocked on a read) and writes framed replies to stdout. ``close``
    is a no-op beyond marking the connection closed — the process owns these
    streams; closing stdin/stdout mid-serve would truncate pending writes.
    """

    def __init__(self) -> None:
        self.closed = False

    async def send(self, chunk: bytes) -> None:
        await asyncio.to_thread(self._write, chunk)

    async def close(self, final_chunk: bytes | None = None) -> None:
        if final_chunk is not None:
            await asyncio.to_thread(self._write, final_chunk)
        self.closed = True

    def _write(self, chunk: bytes) -> None:
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()

    def __aiter__(self) -> _StdioByteConnection:
        return self

    async def __anext__(self) -> bytes:
        loop = asyncio.get_event_loop()
        # ``os.read`` on the stdin fd returns whatever bytes are available
        # (up to n), blocking only when nothing is pending, and b"" on EOF —
        # read1-like semantics with a clean static type (no cast/ignore).
        chunk = await loop.run_in_executor(None, os.read, sys.stdin.fileno(), 65536)
        if not chunk:
            raise StopAsyncIteration
        return bytes(chunk)


async def _acp_stdio(server: ACPServer) -> None:
    """Drive one :class:`ACPServer` over a stdio connection until EOF."""
    await server.serve_connection(_StdioByteConnection())


@app.command()
def chat(
    provider: str = typer.Option(
        "openrouter",
        "--provider",
        "-p",
        envvar="COTHIS_PROVIDER",
        help="provider key (e.g. openrouter, mistral, openai, anthropic).",
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
    summary_model: str | None = typer.Option(
        None,
        "--summary-model",
        help=(
            "Summariser model (provider/model or bare model) used when "
            "compacting the conversation. Overrides COTHIS_SUMMARY_MODEL; "
            "when unset the env var is read inside the agent's "
            "resolve_summary_model, then the session pair."
        ),
    ),
    min_retained_turns: int = typer.Option(
        4,
        "--min-retained-turns",
        envvar="COTHIS_MIN_RETAINED_TURNS",
        help="Min turn-groups retained after compaction (default 4).",
    ),
    max_concurrent_tools: int = typer.Option(
        8,
        "--max-concurrent-tools",
        envvar="COTHIS_MAX_CONCURRENT_TOOLS",
        help=(
            "Max tool executions live at once within a single fan-out turn "
            "(default 8). Caps concurrent MCP/shell/network pressure on "
            "pathological fan-outs; normal turns (1-4 tools) are unaffected."
        ),
    ),
    max_tool_result_chars: int = typer.Option(
        20_000,
        "--max-tool-result-chars",
        envvar="COTHIS_MAX_TOOL_RESULT_CHARS",
        help=(
            "Max characters retained from a single tool result before it is "
            "truncated to a marker (default 20000). Caps prompt bloat from "
            "verbose tools; the full output is not recoverable once capped."
        ),
    ),
    tool_timeout: float | None = typer.Option(
        None,
        "--tool-timeout",
        envvar="COTHIS_TOOL_TIMEOUT",
        help=(
            "Per-tool wall-clock bound in seconds. Omit or leave unset for "
            "no timeout (the default, fully backward compatible). Must be "
            "> 0; interrupts async hangs at the next await point (a stalled "
            "MCP round-trip), not pure-sync blocking tool bodies."
        ),
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
    tui: bool = typer.Option(
        False,
        "--tui",
        help=(
            "Launch the worker-based Textual shell instead of the rich REPL "
            "(the default chat experience)."
        ),
    ),
    legacy: bool = typer.Option(
        False,
        "--legacy",
        help=(
            "Accepted for compatibility; the rich REPL is now the default "
            "chat experience, so this flag is a no-op."
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

    **Default: the rich streaming chat** — prompt_toolkit input (``❯ ``)
    with a virtualized transcript (only visible lines render), follow-end
    scroll, and a ``/`` command menu. Pass ``--tui`` for the opt-in worker-based
    Textual shell (multi-session picker, transient ``/sessions``
    switching, persistent-focus composer). ``--skill`` works in both.

    On the ``--tui`` path all five worker-subprocess tuning flags —
    ``--max-concurrent-tools``, ``--max-tool-result-chars``,
    ``--tool-timeout``, ``--summary-model``, and ``--min-retained-turns`` —
    are forwarded to the spawned worker as ``COTHIS_*`` env vars and take
    effect there (mirroring the rich REPL / ``ask`` behavior).
    """
    if tui:
        # Opt-in worker-based Textual shell (previously the default).
        if resume is not None:
            _validate_session_id_arg(resume)
            _check_resume_exists(_resolve_db_path(), resume)
        _launch_tui_app(
            model=model,
            provider=provider,
            resume=resume,
            max_concurrent_tools=max_concurrent_tools,
            max_tool_result_chars=max_tool_result_chars,
            tool_timeout=tool_timeout,
            summary_model=summary_model,
            min_retained_turns=min_retained_turns,
        )
        return
    # Rich REPL is the default chat experience (prompt_toolkit + rich
    # streaming). ``--skill`` works here directly (preactivate_skills).
    asyncio.run(
        _chat_session(
            model=model,
            provider=provider,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            resume=resume,
            preactivate_skills=list(skill),
            summary_model=summary_model,
            min_retained_turns=min_retained_turns,
            max_concurrent_tools=max_concurrent_tools,
            max_tool_result_chars=max_tool_result_chars,
            tool_timeout=tool_timeout,
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
    summary_model: str | None = None,
    min_retained_turns: int = 4,
    max_concurrent_tools: int = 8,
    max_tool_result_chars: int = 20_000,
    tool_timeout: float | None = None,
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
                summary_model=summary_model,
                min_retained_turns=min_retained_turns,
                max_concurrent_tools=max_concurrent_tools,
                max_tool_result_chars=max_tool_result_chars,
                tool_timeout=tool_timeout,
            )
            agent.attach_session(session)

        # Full-screen streaming chat: virtualized transcript (rich +
        # prompt_toolkit) with follow-end scroll and a ``/`` command menu.
        # ``run_streaming_chat`` owns ``agent.aclose()``.
        from cothis.streaming_tui import run_streaming_chat

        await run_streaming_chat(agent)
    finally:
        # Idempotent: if the streaming chat already closed the agent
        # (drained + joined + storage closed), this is a no-op; if Agent
        # construction failed before attach, this is the cleanup path.
        session.close()


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


@app.command(name="search")
def search_cmd(
    query: str = typer.Argument(
        ..., help="FTS5 query (terms, \"phrases\", prefix*)."
    ),
    limit: int = typer.Option(
        50, "--limit", "-n", help="Max hits to print."
    ),
) -> None:
    """Search stored message text across all sessions in this DB.

    Runs a full-text (FTS5) MATCH over the searchable text of every stored
    block in the resolved DB and prints ranked hits, newest-relevant first.
    DB-wide by design (no cwd scoping yet) — recall over every session in
    this DB, including forked and promoted ones. ``query`` is a raw FTS5
    expression, so quote phrases (``\"deploy script\"``), use ``prefix*``,
    and combine with ``AND``/``OR``/``NOT``.

    Each line: ``<session_id>  <seq>  [<role>/<type>]  <snippet>``.
    """
    db_path = _resolve_db_path()
    if not db_path.exists():
        console.print("[dim]no sessions database yet[/dim]")
        return
    storage = Storage(db_path)
    try:
        try:
            hits: list[SearchHit] = storage.search(query, limit=limit)
        except sqlite3.OperationalError as exc:
            # A malformed FTS5 MATCH expression (unterminated quote, bad
            # syntax) surfaces as OperationalError. The CLI is the
            # user-facing surface, so print a clean line instead of a
            # traceback + exit 1 — mirrors the ``no matches`` handling.
            console.print(f"[dim]invalid query: {exc}[/dim]")
            return
        if not hits:
            console.print("[dim]no matches[/dim]")
            return
        for hit in hits:
            typer.echo(
                f"{hit.session_id}  {hit.seq}  "
                f"[{hit.role}/{hit.type}]  {hit.snippet}"
            )
    finally:
        storage.close()


@app.command(name="archive")
def archive_cmd(
    action: str = typer.Argument(
        "all", help="'all' (default), 'list', '<session_id>', 'restore <id>', 'compress <file>'"
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
        cothis archive list           # list archived (cold) sessions
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
    elif action == "list":
        archived = Session.list_archived(db_path)
        if not archived:
            console.print("no archived sessions")
            return
        for sid, sr, archived_at in archived:
            title = sr.title or f"session {sid[:8]}"
            cwd_hint = str(sr.cwd)
            console.print(
                f"[cyan]{sid[:8]}…[/cyan]  {title}  "
                f"[dim]({cwd_hint}, archived {archived_at[:10]})[/dim]"
            )
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


# ---------------------------------------------------------------------
# TUI entrypoint (#234) — Supervisor-backed spawn on worktree pick.
# Feature parity reached: the TUI drives full multi-turn sessions
# (WS attach + run_turn + tool-call rendering + ask_user modal +
# worktree picker + cwd fallback). ``chat`` stays as the legacy REPL
# until #237 makes the TUI the default; ``--legacy`` will keep the
# old REPL as an escape hatch during the staged migration.
# ---------------------------------------------------------------------


class _DrivenCothisApp:
    """Production CothisApp wiring: spawn-on-pick via real Supervisor (#234).

    Defined as a factory function (not a class statement at module level) because
    it needs ``CothisApp`` which is imported lazily — Textual is heavy (~200ms)
    and the CLI's other commands mustn't pay that cost on import. The factory
    returns a ``CothisApp`` subclass whose ``on_worktree_pick`` runs the spawn
    recipe (Session.new + Supervisor.spawn_worker + schedule attach_session_ws).

    Tests construct the subclass directly via this factory + mock Supervisor /
    Session to verify the spawn args without spawning real subprocesses.
    """

    if TYPE_CHECKING:
        from cothis.supervisor import Supervisor
        from cothis.tui import CothisApp

    @staticmethod
    def build(
        *,
        supervisor: Supervisor,
        model: str,
        provider: str,
        provider_env: dict[str, str],
        resume_session_id: str | None = None,
    ) -> CothisApp:
        """Return a ``CothisApp`` subclass instance with spawn wired into on_worktree_pick.

        If ``resume_session_id`` is set, the app auto-spawns a worker for that
        session on ``on_mount`` (bypasses the worktree picker).

        ``provider_env`` is passed verbatim as ``extra_env`` to
        ``Supervisor.spawn_worker`` in both spawn closures. It carries both the
        operator's ``*_API_KEY`` vars and any resolved ``COTHIS_*`` tuning vars
        the worker's Agent / ``resolve_summary_model`` should pick up (the
        merge happens in :func:`_launch_tui_app`).

        ``on_worktree_pick`` resolves the session db itself (relative to the
        picked worktree) so the spawned worker reads the same db in every
        ``_resolve_db_path`` mode — including cwd-relative project mode (#402).
        """
        from cothis.session import Session
        from cothis.tui import CothisApp

        class _App(CothisApp):
            def on_worktree_pick(self, path: str) -> None:  # type: ignore[override]
                cwd = Path(path)
                # Resolve the db relative to the *picked worktree* — not the
                # TUI's launch dir — so the spawned worker (which runs in the
                # worktree and resolves ``_resolve_db_path()`` there) reads
                # the same db the TUI writes (#400 / #402). In default +
                # ``COTHIS_SESSIONS_DIR`` modes the path is cwd-independent,
                # so this matches the launch-cwd resolution; in project mode
                # it scopes the session to its worktree instead of leaving it
                # in the launch dir where the worker can't find it.
                wt_db = _resolve_db_path(cwd=cwd)
                wt_db.parent.mkdir(parents=True, exist_ok=True)
                session = Session.new(
                    wt_db, cwd=cwd, model=model, flush_sync=True,
                )
                session.append_message(
                    "user",
                    [{"type": "text", "text": f"(session created in {cwd.name})"}],
                )
                sid = session.session_id
                session.close()

                try:
                    handle = supervisor.spawn_worker(
                        sid,
                        model=model,
                        provider=provider,
                        cwd=cwd,
                        extra_env=provider_env,
                    )
                except Exception as exc:
                    # cothis: spawn failed — roll back the just-persisted
                    # session row so it doesn't surface as a ghost in
                    # ``cothis history`` (#390). Do NOT unlink the db file
                    # — it's the shared db (other sessions live in it).
                    Session.delete(wt_db, sid)
                    logging.getLogger(__name__).error(
                        "tui: spawn failed for session %s in %s; "
                        "rolled back: %s",
                        sid[:8], cwd, exc,
                    )
                    return

                logging.getLogger(__name__).info(
                    "tui: spawned session %s in %s", sid[:8], cwd,
                )

                # ``attach_session_ws`` is async; ``create_task`` keeps the
                # sync callback (Textual modal dismiss) unblocked.
                # Seed the footer's model + session cells from the
                # spawn handle's sid + the CLI's known model so the status
                # bar shows correct values BEFORE the first turn_finished
                # frame lands (the worker emits no per-spawn frame).
                # Pass ``wt_db`` so attach replays this session's
                # stored history into ConversationView before the pump
                # starts. The freshly-created session only has its seed
                # user message, so this renders that opening turn.
                asyncio.create_task(
                    self.attach_session_ws(
                        sid, handle.ws_url, handle.token, db_path=wt_db,
                    ),
                )
                self.footer_session = sid
                self.footer_model = model

            async def on_mount(self) -> None:
                # Run the base on_mount (focus SessionList) so the bare `n`
                # "new session" shortcut still works in production — the
                # InputBar wrapper removal (#375) otherwise lets the TextArea
                # grab launch focus and swallow `n` as text.
                await super().on_mount()
                # cothis: start the crash-monitor loop (#398). Detects
                # crashed workers + restarts them with backoff; the
                # ``on_restart`` callback re-attaches the TUI's WS to the
                # new worker's fresh port/token. The task is stashed so
                # ``on_unmount`` can cancel it cleanly — otherwise the
                # event loop closes on a pending ``asyncio.sleep`` and
                # logs "Task was destroyed but it is pending!".
                self._monitor_task = asyncio.create_task(
                    supervisor.monitor_worker_health(),
                )
                # Auto-spawn for --resume: bypass the worktree picker
                # and attach the resumed session directly.
                if resume_session_id is None:
                    return
                handle = supervisor.spawn_worker(
                    resume_session_id,
                    model=model,
                    provider=provider,
                    cwd=Path.cwd(),
                    extra_env=provider_env,
                )
                logging.getLogger(__name__).info(
                    "tui: resumed session %s", resume_session_id[:8],
                )
                # Replay the resumed session's stored history on
                # attach. ``_resolve_db_path()`` (no cwd) resolves the same
                # db the worker reads — the worker runs at Path.cwd()
                # (the TUI's launch cwd), matching launch-cwd resolution
                # in all three ``_resolve_db_path`` modes.
                await self.attach_session_ws(
                    resume_session_id, handle.ws_url, handle.token,
                    db_path=_resolve_db_path(),
                )
                # Seed footer model/session so the status bar is
                # populated before the first turn_finished frame.
                self.footer_session = resume_session_id
                self.footer_model = model

            async def on_unmount(self) -> None:
                # cothis: cancel the crash-monitor loop on app exit (#398
                # review). Without this the event loop closes while the
                # monitor task is mid-``asyncio.sleep`` and asyncio logs
                # "Task was destroyed but it is pending!".
                task = getattr(self, "_monitor_task", None)
                if task is None or task.done():
                    return
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            async def _reattach_on_restart(
                self, sid: str, ws_url: str, token: str,
            ) -> None:
                # cothis: crash-restart re-attach (#398 review). Wraps
                # ``attach_session_ws`` so a WS-connect failure on the
                # fresh port (re-crash, slow bind, port collision — the
                # exact failures the recovery path must survive) is logged
                # instead of swallowed as an un-retrieved task exception.
                # No ``db_path`` is passed (replay is skipped) — the
                # view is already populated from the original attach, so
                # replaying would duplicate the history. ``db_path``
                # defaults to ``None`` → no replay.
                logging.getLogger(__name__).info(
                    "tui: re-attaching %s to restarted worker (ws=%s)",
                    sid[:8], ws_url,
                )
                try:
                    await self.attach_session_ws(sid, ws_url, token)
                except Exception:  # noqa: BLE001 — best-effort re-attach
                    logging.getLogger(__name__).warning(
                        "tui: re-attach failed for %s (ws=%s)",
                        sid[:8], ws_url, exc_info=True,
                    )

        app = _App()
        # cothis: wire crash-restart re-attach (#398). When the monitor
        # restarts a crashed worker, the new worker has a fresh WS port +
        # token; the TUI must re-attach so the next prompt reaches the new
        # worker, not the dead connection. Routed through
        # ``_reattach_on_restart`` so a connect failure is logged, not
        # lost as an un-retrieved task exception.
        supervisor.on_restart = lambda sid, handle: asyncio.create_task(
            app._reattach_on_restart(sid, handle.ws_url, handle.token),
        )
        return app


def _worker_tuning_env(
    *,
    max_concurrent_tools: int | None = None,
    max_tool_result_chars: int | None = None,
    tool_timeout: float | None = None,
    summary_model: str | None = None,
    min_retained_turns: int | None = None,
) -> dict[str, str]:
    """Map resolved chat tuning values to the ``COTHIS_*`` env vars the spawned
    worker's Agent / ``resolve_summary_model`` read.

    Only non-None values are forwarded: ``None`` means "do not override; let
    the worker inherit the shell env / Agent defaults", which is what keeps
    ``cothis tui`` (passes nothing) regression-free — the worker still gets
    its env verbatim from ``spawn_worker``'s ``dict(os.environ)``.

    ``COTHIS_MIN_RETAINED_TURNS`` is honored by the worker's Agent via the
    ``model_post_init`` override-or-None reader, so a forwarded value takes
    effect there just like the other tuning vars.
    """
    env: dict[str, str] = {}
    if max_concurrent_tools is not None:
        env["COTHIS_MAX_CONCURRENT_TOOLS"] = str(max_concurrent_tools)
    if max_tool_result_chars is not None:
        env["COTHIS_MAX_TOOL_RESULT_CHARS"] = str(max_tool_result_chars)
    if tool_timeout is not None:
        env["COTHIS_TOOL_TIMEOUT"] = str(tool_timeout)
    if summary_model is not None:
        env["COTHIS_SUMMARY_MODEL"] = summary_model
    if min_retained_turns is not None:
        env["COTHIS_MIN_RETAINED_TURNS"] = str(min_retained_turns)
    return env


def _launch_tui_app(
    model: str,
    provider: str,
    resume: str | None = None,
    *,
    max_concurrent_tools: int | None = None,
    max_tool_result_chars: int | None = None,
    tool_timeout: float | None = None,
    summary_model: str | None = None,
    min_retained_turns: int | None = None,
) -> None:
    """Construct Supervisor + _DrivenCothisApp and launch the Textual TUI.

    Shared between ``cothis tui`` and ``cothis chat`` (default TUI path,
    #237 staged migration). Both commands construct the same Supervisor +
    spawn-wired app; the only difference is the entrypoint name.

    ``resume`` (optional): if set, auto-spawn a worker for the existing
    session on startup (bypasses the worktree picker).

    The five tuning kwargs (all default ``None``) are forwarded to the spawned
    worker as ``COTHIS_*`` env vars via :func:`_worker_tuning_env`. ``chat``
    passes its resolved typer values (so they take effect in the worker);
    ``tui`` passes nothing, preserving today's inherit-shell-env behavior.
    """
    from cothis.supervisor import Supervisor
    from cothis.tui import run as run_tui

    sup = Supervisor()

    provider_env = {
        k: v
        for k, v in os.environ.items()
        if k.endswith("_API_KEY") and v
    }
    tuning_env = _worker_tuning_env(
        max_concurrent_tools=max_concurrent_tools,
        max_tool_result_chars=max_tool_result_chars,
        tool_timeout=tool_timeout,
        summary_model=summary_model,
        min_retained_turns=min_retained_turns,
    )
    worker_env = {**provider_env, **tuning_env}

    app = _DrivenCothisApp.build(
        supervisor=sup,
        model=model,
        provider=provider,
        provider_env=worker_env,
        resume_session_id=resume,
    )
    # cothis: graceful shutdown on TUI exit (#403). Without this, spawned
    # workers are orphaned (reparented to init), still holding session locks
    # — a later ``cothis chat --resume`` raises SessionLockedError until the
    # orphan is manually killed. ``sup.close()`` sends each worker the
    # graceful ``shutdown`` control message (flush + release), so the session
    # is cleanly persisted and the lock released on every exit path.
    try:
        run_tui(app=app)
    finally:
        sup.close()


@app.command()
def tui(
    model: str = typer.Option(
        "openai/gpt-oss-120b",
        "--model",
        "-m",
        envvar="COTHIS_MODEL",
        help="model id (mirrors ``chat``). Used when spawning new sessions.",
    ),
    provider: str = typer.Option(
        "openrouter",
        "--provider",
        "-p",
        envvar="COTHIS_PROVIDER",
        help="provider key (mirrors ``chat``). Used when spawning new sessions.",
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume a session by id (auto-spawns on startup, bypasses the picker).",
    ),
) -> None:
    """Launch the Textual TUI with Supervisor-backed session spawn (#234).

    Wires ``on_worktree_pick`` (the pick hook) to a real Supervisor: when
    the user picks a worktree via the ``n`` keypress, a new session bound to
    that cwd is created + a worker is spawned + the TUI auto-attaches its WS.

    ``--resume <id>`` auto-spawns a worker for an existing session on
    startup (bypasses the worktree picker). ``--legacy`` is not needed
    — ``tui`` always launches the TUI.

    ``on_session_selected`` still defaults to log + return (focus routing
    across sessions is a follow-up). ``on_menu_open`` + ``on_ask_user_request``
    mount their modals (ConfigMenuModal / AskUserModal) as shipped.
    """
    if resume is not None:
        _validate_session_id_arg(resume)
        _check_resume_exists(_resolve_db_path(), resume)
    _launch_tui_app(model=model, provider=provider, resume=resume)


# ---------------------------------------------------------------------
# Worker subprocess entrypoint (#250, deferred from #225)
#
# Spawns one ``SessionWorker`` that owns a single session + binds a WS
# server on a random loopback port. Prints one JSON line to stdout
# (``{"uri": ..., "token": ...}``) when the bind completes, then serves
# until a ``shutdown`` control message arrives. The Supervisor (#227)
# reads the JSON line to learn the URI + bearer token; the future
# integration test (#250 path (a)) drives this via ``subprocess.Popen``.
# ---------------------------------------------------------------------


@app.command()
def worker(
    session: str = typer.Option(
        ...,
        "--session",
        "-s",
        help="Session id to load (created by the Supervisor before spawning).",
    ),
    provider: str = typer.Option(
        "openrouter",
        "--provider",
        "-p",
        envvar="COTHIS_PROVIDER",
        help="provider key (mirrors ``chat``).",
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
        help="Output-token cap. Default: resolved from bundled litellm metadata.",
    ),
) -> None:
    """Run one SessionWorker for ``--session``; emit bind JSON, serve forever.

    Bind handshake on stdout (single JSON line, flushed):

        {"uri": "ws://127.0.0.1:<port>/agent", "token": "<bearer>"}

    Then runs the accept loop until ``shutdown`` arrives on the WS or
    the process is killed externally. Exit code 0 on clean shutdown.
    """
    asyncio.run(
        _worker_session(
            session=session,
            model=model,
            provider=provider,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
        )
    )


async def _worker_session(
    *,
    session: str,
    model: str,
    provider: str,
    max_iterations: int,
    max_tokens: int | None,
) -> None:
    """Build Agent + Session + SessionWorker; emit bind JSON; serve."""
    from cothis.skills import discover_skills, load_skill_selection
    from cothis.worker import SessionWorker

    _validate_session_id_arg(session)
    db_path = _resolve_db_path()
    try:
        loaded = Session.load(db_path, session, cwd=Path.cwd())
    except KeyError:
        raise typer.BadParameter(
            f"session {session!r} not found; run `cothis history` to list"
        )
    # cothis: preactivate the TUI's persisted skill selection (#415).
    # Filter against currently-available skills so an unavailable persisted
    # name (a project-scoped skill chosen in another project, or one
    # uninstalled between sessions) can't crash the worker in
    # ``_run_preactivation`` with "Unknown skill" — mirrors the TUI's
    # menu-open filter (``saved & set(skills)``). ``discover_skills`` is
    # mtime-cached (#414), so the extra call is ~free.
    available = {s.name for s in discover_skills(Path.cwd())}
    try:
        agent = Agent(
            model=model,
            provider=provider,
            tools=discover_tools(_PROJECT_TOOLS_DIR, _user_tools_dir()),
            system=DEFAULT_SYSTEM_PROMPT,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            cwd=Path.cwd(),
            preactivate_skills=sorted(set(load_skill_selection()) & available),
        )
        agent.attach_session(loaded)

        worker = SessionWorker(agent)
        uri = await worker.start()
        # Bind handshake: emit one JSON line + flush so the supervisor
        # reading line-by-line sees it immediately. ``token`` is the
        # bearer the client must present on the WS handshake.
        print(json.dumps({"uri": uri, "token": worker.token}), flush=True)
        try:
            await worker.serve_forever()
        except (asyncio.CancelledError, KeyboardInterrupt):
            # ``shutdown`` from a client closes the WS server, which the
            # websockets library surfaces as CancelledError on the
            # ``serve_forever`` await. Treat as the normal exit signal —
            # cleanup runs in finally, process exits 0.
            pass
    finally:
        # ``worker.stop`` is idempotent; ``agent.aclose`` tears down MCP
        # handles + drains the session queue. Both run on every exit path
        # so a Ctrl-C / kill between bind + serve still cleans up.
        if "worker" in locals():
            await worker.stop()
        if "agent" in locals():
            await agent.aclose()


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



@app.command(name="install")
def install_cmd(
    specs: list[str] = typer.Argument(..., help="One or more PyPI package specs to install as extensions (e.g. 'rich', 'httpx>=0.27')."),
) -> None:
    """Install one or more extensions into the shared extensions venv."""
    from cothis.extensions import ExtensionError, ExtensionManager

    try:
        typer.echo(f"installing {len(specs)} extension(s)...")
        exts = ExtensionManager(_cothis_home()).install(specs)
        for ext in exts:
            typer.echo(f"installed extension {ext.name} {ext.version or ''}")
    except (ExtensionError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)


@app.command(name="extensions")
def extensions_cmd(
    action: str = typer.Argument("list", help="'list' (default) — list installed extensions."),
) -> None:
    """List installed extensions."""
    from cothis.extensions import ExtensionManager

    if action != "list":
        typer.echo(f"unknown action {action!r}; use 'list'", err=True)
        raise typer.Exit(1)
    exts = ExtensionManager(_cothis_home()).discover()
    if not exts:
        typer.echo("no extensions installed")
        return
    for ext in exts:
        typer.echo(f"{ext.name}\t{ext.version or '?'}\t{ext.spec}")


@app.command(name="skills")
def skills_cmd(
    action: str = typer.Argument("list", help="'list' (default) — list discovered Agent Skills."),
) -> None:
    """List discovered Agent Skills (project > user-cothis > user-agents layers)."""
    from cothis.skills import discover_skills

    if action != "list":
        typer.echo(f"unknown action {action!r}; use 'list'", err=True)
        raise typer.Exit(1)
    skills = discover_skills(Path.cwd())
    if not skills:
        typer.echo("no skills found")
        return
    # Layer roots match ``discover_skills``'s own resolution (skills.py):
    # project cwd/.agents/skills, user-cothis $COTHIS_HOME/skills (via the
    # shared ``_cothis_home`` helper), user-agents ~/.agents/skills. Labels
    # are derived by path-prefix matching the equally-unresolved ``source``
    # (discover_skills stores ``skill_dir / SKILL.md`` without ``.resolve()``),
    # not by re-running env logic.
    project = Path.cwd() / ".agents" / "skills"
    user_cothis = _cothis_home() / "skills"
    user_agents = Path.home() / ".agents" / "skills"

    def _layer(src: Path) -> str:
        if src.is_relative_to(project):
            return "project"
        if src.is_relative_to(user_cothis):
            return "user-cothis"
        if src.is_relative_to(user_agents):
            return "user-agents"
        return "?"

    for s in skills:
        typer.echo(f"{s.name}\t{_layer(s.source)}\t{s.description[:80]}")


if __name__ == "__main__":
    main()
