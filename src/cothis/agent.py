"""An agent loop built on top of any-llm.

The loop is the standard ReAct-style cycle:

1. Send the conversation + tool schemas to the model.
2. If the model asks for tool calls, execute them and append the results.
3. Repeat until the model produces a message without tool calls, or
   ``max_iterations`` is reached.

Example
-------
>>> from cothis.agent import Agent
>>> print(agent.run("What is 47 * 83?"))
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

# cothis: the Anthropic stream-event types (``RawMessageStartEvent`` etc.)
# and ``TextDelta`` are imported lazily inside ``Agent.run_stream`` instead
# of at module top. The ``anthropic`` SDK pulls in ``anthropic.lib.vertex``
# and ``anthropic.lib.bedrock`` on first import (~1s); ``any_llm.types.messages``
# hard-imports ``anthropic.types`` at top level (~1.5s more). Deferring both
# to the first real LLM call keeps ``cothis --help`` / non-LLM paths from
# paying ~2.5s of SDK load. The ``isinstance`` dispatch in ``run_stream``
# is how ty narrows the ``MessageStreamEvent`` union (it can't narrow by
# string ``event.type`` comparison). The union is itself just these
# anthropic SDK classes (any-llm re-exports them).
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# cothis: ``Tool`` must be runtime-imported (not TYPE_CHECKING-only) because
# pydantic resolves the ``list[Tool]`` field annotation at model-build time
# via ``typing.get_type_hints``, which needs ``Tool`` in the module globals.
# ``from __future__ import annotations`` makes the annotation a string, so
# ruff's TC001 rule can't see the runtime use and wants it moved under
# TYPE_CHECKING — which would crash pydantic. This noqa is the honest
# representation of that constraint.
from cothis.model_metadata import resolve_max_tokens
from cothis.skills import discover_skills, format_catalog
from cothis.tools import (
    AfterExecuteError,
    HandleManager,
    MCPServer,
    Tool,
    ensure_handle_ready,
    format_tool_output,
    handle_call_done,
    mark_inflight,
    run_hooks_safe,
    schema_for,
)
from cothis.tools.fs._hygiene import workdir_context
from cothis.tools.mcp import MCPSessionHandle

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from any_llm import AnyLLM
    from any_llm.types.messages import MessageResponse, MessageStreamEvent

    from cothis.session import Session


Message = dict[str, Any]

logger = logging.getLogger("cothis.agent")


def _system_param(system: str | list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Build the ``amessages`` ``system`` parameter as a block list.

    A ``str`` persona is assembled via ``_assemble_system`` into
    ``[persona_block, agents_md_block?, catalog_slot?]``. A pre-built block
    list is passed through unchanged. ``None`` → ``None``.
    """
    if system is None:
        return None
    if isinstance(system, str):
        return _assemble_system(system)
    return system


def _assemble_system(persona: str) -> list[dict[str, Any]]:
    """Build the system block list: ``[persona, agents_md?, catalog?]``.

    Each block carries ``cache_control: {type: ephemeral}``. The AGENTS.md
    block is included only when at least one file is found; the catalog
    block slot is reserved for #30 (no-op until skills land).
    """
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": persona, "cache_control": {"type": "ephemeral"}}
    ]
    agents_md = _load_agents_md()
    if agents_md is not None:
        blocks.append(
            {
                "type": "text",
                "text": agents_md,
                "cache_control": {"type": "ephemeral"},
            }
        )
    # cothis: skills catalog (#68). Discover from 3 layers and append
    # the ``<available_skills>`` block. Omitted entirely when no skills
    # or when discovery fails (best-effort; the agent still runs).
    try:
        catalog = format_catalog(discover_skills(Path.cwd()))
        if catalog is not None:
            blocks.append(
                {
                    "type": "text",
                    "text": catalog,
                    "cache_control": {"type": "ephemeral"},
                }
            )
    except Exception as exc:  # noqa: BLE001 — catalog is best-effort
        logger.debug("skills discovery failed: %s", exc)
    return blocks


def _load_agents_md() -> str | None:
    """Read AGENTS.md from configured layers, concat, XML-tag.

    Layers are read in ``COTHIS_AGENTS_ORDER`` (default:
    ``user-agents,user-cothis,project``). Each layer matches the first file
    matching any filename in ``COTHIS_AGENTS_PATTERN`` (default:
    ``AGENTS.md``). User-global layers are skipped when
    ``COTHIS_AGENTS_USER_GLOBAL`` is ``0`` / ``false`` / ``no`` / ``off``.
    Returns ``None`` when no file is found (or all found files are empty).
    """
    import os

    pattern = os.environ.get("COTHIS_AGENTS_PATTERN", "AGENTS.md")
    patterns = [p.strip() for p in pattern.split(",")]

    order = os.environ.get("COTHIS_AGENTS_ORDER", "user-agents,user-cothis,project")
    layer_order = [o.strip() for o in order.split(",")]

    user_global = os.environ.get("COTHIS_AGENTS_USER_GLOBAL", "1")
    user_global = user_global.lower() not in ("0", "false", "no", "off")

    home = Path.home()
    # cothis: mirror cli.py's COTHIS_HOME resolution so empty/unset/~/ all
    # behave identically across modules. Extract a shared resolver if a
    # third caller appears.
    cothis_home = Path(
        os.environ.get("COTHIS_HOME") or (home / ".cothis")
    ).expanduser()

    # Map layer names to their directories (only layers present in the order).
    # Unknown layer names are logged at debug (visible under --debug) but
    # never fatal — the user might have a future layer in their env. A typo
    # in COTHIS_AGENTS_ORDER surfaces there rather than as a silent empty block.
    _KNOWN_LAYERS = ("user-agents", "user-cothis", "project")
    layer_dirs: dict[str, Path] = {}
    for name in layer_order:
        if name == "user-agents" and user_global:
            layer_dirs[name] = home / ".agents"
        elif name == "user-cothis" and user_global:
            layer_dirs[name] = cothis_home
        elif name == "project":
            layer_dirs[name] = Path.cwd()
        elif name not in _KNOWN_LAYERS:
            logger.debug(
                "COTHIS_AGENTS_ORDER: unknown layer %r skipped (known: %s)",
                name, ", ".join(_KNOWN_LAYERS),
            )

    parts: list[str] = []
    for name in layer_order:
        layer_dir = layer_dirs.get(name)
        if layer_dir is None:
            continue
        content = _read_first_matching(layer_dir, patterns)
        if content is None:
            continue
        parts.append(f'<agents_md type="{name}">\n{content}\n</agents_md>')

    if not parts:
        return None
    return "\n\n".join(parts)


def _read_first_matching(directory: Path, patterns: list[str]) -> str | None:
    """Read the first file in *directory* matching any *patterns* pattern.

    Returns ``None`` when no file is found or the matched file is empty
    (after stripping). Patterns are literal filenames, not globs.
    """
    import locale

    fallback = locale.getpreferredencoding(False)
    for pat in patterns:
        filepath = Path(directory) / pat
        try:
            text = _read_text(filepath, fallback)
        except OSError:
            continue
        if text and text.strip():
            return text
    return None


def _read_text(filepath: Path, fallback_encoding: str) -> str | None:
    """Read *filepath* as UTF-8, then *fallback_encoding*; ``None`` if neither decodes.

    ponytail: two-tier decode covers UTF-8 (~99% of AGENTS.md) plus
    Windows/legacy encodings (GBK/CP1252/etc. via the locale fallback).
    Skip — never ``errors="replace"`` — on decode failure: garbled bytes
    injected into the system prompt are worse than the block being absent.
    Auto-detection (chardet) would be a new dep for a rare case; revisit
    only if the locale tier measurably falls short.
    """
    try:
        return filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return filepath.read_text(encoding=fallback_encoding)
    except (UnicodeDecodeError, LookupError):
        return None


def _assistant_msg_from_response(response: MessageResponse) -> dict[str, Any]:
    """Convert a non-stream ``MessageResponse`` into the dict stored in ``_messages``.

    The dict carries response metadata (``id``/``model``/``stop_reason``/
    ``usage``) so callers can inspect it; ``_request_messages`` strips those
    before the next ``amessages`` call (Anthropic's native API rejects extra
    fields on message dicts). ``content`` blocks are ``model_dump``'d with
    ``exclude_none`` so no ``None`` fields leak into the replayed request.
    """
    return {
        "role": "assistant",
        "content": [b.model_dump(exclude_none=True) for b in response.content],
        "id": response.id,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "usage": response.usage.model_dump(exclude_none=True)
        if response.usage
        else None,
    }


def _request_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project stored messages to ``{role, content}`` for ``amessages``.

    Anthropic's native API validates message dicts strictly; the metadata we
    keep on assistant messages for inspection would be rejected, so strip to
    the two request-side fields at send time.
    """
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def _tool_result_block(
    tool_use_id: str, content: str, is_error: bool
) -> dict[str, Any]:
    """Build a ``tool_result`` content block for a user message.

    ``content`` is the (already-serialised) tool output string (matches
    ``_execute``'s string contract). On error the Anthropic-native
    ``is_error: true`` flag is set so the model can tell a failed call from a
    successful one.
    """
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


def _append_merged(
    messages: list[dict[str, Any]], role: str, block: dict[str, Any]
) -> None:
    """Append ``block`` to ``messages``, merging into the last same-role message.

    Anthropic requires strict user/assistant alternation, so per-execution
    ``tool_result`` blocks (all ``role="user"`` from one assistant turn) must
    land in ONE user message, not N. If the last message is already
    ``role``, extend its ``content``; otherwise open a new message.
    """
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].append(block)
    else:
        messages.append({"role": role, "content": [block]})


def _sanitize_tool_name(name: str) -> str:
    """Map a tool name onto the OpenAI tool-name pattern ``^[a-zA-Z0-9_-]+$``.

    cothis namespaces tools with ``.``/``:`` (``fs.read``, ``mcp:context7``),
    which Anthropic accepts but strict OpenAI-compatible providers (DeepSeek)
    reject with HTTP 400. Applied symmetrically — schema names sent to the
    model and ``_tool_map`` keys both run through this, so dispatch matches.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _concat_text(content: list[dict[str, Any]]) -> str:
    """Concatenate every ``text`` block in an assistant content list."""
    return "".join(
        b.get("text", "") for b in content if b.get("type") == "text"
    )


def _tool_uses_in(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every ``tool_use`` block in an assistant content list.

    The turn decision keys on this (not ``stop_reason``): a generation cut
    off mid-tool-call by ``max_tokens`` carries ``stop_reason == "max_tokens"``
    but still leaves a ``tool_use`` block in the content. Keying on ``stop_reason``
    would return that turn as final, storing an assistant message whose
    ``tool_use`` is unpaired — Anthropic requires every ``tool_use`` to be
    followed by a matching ``tool_result``, so the next turn 400s and the
    session is poisoned. Detecting by block presence keeps the pairing
    invariant intact regardless of why the turn ended.
    """
    return [b for b in content if b.get("type") == "tool_use"]


def _init_stream_block(content_block: Any) -> dict[str, Any]:
    """Seed a block dict from ``ContentBlockStartEvent.content_block``.

    The start event's ``content_block`` already carries the block's type and
    initial fields (``ToolUseBlock``: ``id``/``name``/``input={}``;
    ``TextBlock``: ``text=""``; ``ThinkingBlock``: ``thinking=""``). Dump it
    and add a private ``_input_json`` accumulator for ``tool_use`` partials.
    """
    d = content_block.model_dump(exclude_none=True)
    if d.get("type") == "tool_use":
        d["_input_json"] = ""
    return d


def _apply_stream_delta(block: dict[str, Any], delta: Any) -> None:
    """Accumulate one ``ContentBlockDeltaEvent.delta`` into ``block`` in place.

    ``TextDelta``/``ThinkingDelta`` append; ``SignatureDelta`` overwrites (it
    carries the block's final signature, finalised at ``content_block_stop`` —
    appending would corrupt it); ``InputJSONDelta`` appends to the private
    ``_input_json`` string, parsed by ``_finalize_stream_block``.
    """
    dtype = delta.type
    if dtype == "text_delta":
        block["text"] = block.get("text", "") + delta.text
    elif dtype == "thinking_delta":
        block["thinking"] = block.get("thinking", "") + delta.thinking
    elif dtype == "signature_delta":
        block["signature"] = delta.signature
    elif dtype == "input_json_delta":
        block["_input_json"] = block.get("_input_json", "") + delta.partial_json


def _finalize_stream_block(block: dict[str, Any]) -> None:
    """Parse a ``tool_use`` block's accumulated ``_input_json`` into ``input``."""
    if block.get("type") != "tool_use":
        return
    raw = block.pop("_input_json", "")
    # cothis: partial_json should be complete JSON by content_block_stop.
    # Malformed → empty dict so dispatch surfaces a clean error rather than
    # crashing mid-stream. Upgrade path: surface the bad JSON to the model.
    try:
        block["input"] = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        block["input"] = {}


@dataclass(frozen=True)
class ToolCallEvent:
    """Streamed event: the agent is about to invoke a tool.

    Yielded by ``Agent.run_stream`` before each tool dispatch so the CLI
    can surface "calling fs.read(...)" inline. ``arguments`` is the parsed
    dict (matches what will be passed to the tool); the raw JSON string is
    dropped because the parsed form is what the user wants to read.
    """

    name: str
    arguments: dict[str, Any]


class MaxIterationsError(RuntimeError):
    """Raised when the agent exhausts its iteration budget before finishing."""


def _coalesce_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop empty text blocks and merge adjacent same-type blocks.

    Reasoning-capable providers (e.g. ``gpt-oss-120b`` on openrouter) emit
    ``reasoning`` and ``content`` as **mutually exclusive** per chunk: while
    reasoning, ``delta.content == ""`` (explicit "no text right now"); while
    answering, ``delta.reasoning`` is absent. any-llm's OpenAI→Messages
    stream converter, however, opens a text-block lifecycle on any chunk
    where ``delta.content is not None`` — so each reasoning chunk opens and
    closes an empty ``text`` block, and the resulting assistant message
    holds dozens of tiny ``thinking`` fragments interleaved with empty
    ``text`` blocks. When that malformed message is replayed on the next
    turn, some providers silently return empty content — observed as
    ``cothis chat`` finishing a tool turn and returning to ``>>>`` with no
    answer shown.

    This helper filters at storage time: empty ``text`` blocks (the explicit
    "no content" signal) are dropped, and adjacent same-type blocks are
    merged into one. The result matches the canonical Messages shape (one
    ``thinking`` + one ``text`` + ``tool_use``). ``tool_use`` blocks are
    preserved verbatim and never merged.

    cothis: ``thinking`` signatures are not preserved across merges — this
    slice does not pass the ``thinking`` param, so Anthropic doesn't validate
    them on the way back, and other providers' reasoning blocks never carry
    a real signature anyway. The proper fix is upstream (any-llm should use
    ``if delta.content:`` rather than ``if delta.content is not None:``);
    this is cothis's defensive layer so the agent works regardless.
    """
    out: list[dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        # Drop empty text blocks. An empty text block is the explicit "no
        # content" signal from the provider (delta.content == "" during
        # reasoning chunks); keeping it poisons the next-turn request.
        if btype == "text" and not (block.get("text") or "").strip():
            continue
        # Merge adjacent text/thinking blocks into the previous one.
        if (
            out
            and out[-1].get("type") == btype
            and btype in ("text", "thinking")
        ):
            if btype == "text":
                out[-1]["text"] = (out[-1].get("text") or "") + (
                    block.get("text") or ""
                )
            else:  # thinking
                out[-1]["thinking"] = (out[-1].get("thinking") or "") + (
                    block.get("thinking") or ""
                )
            continue
        # Copy so we don't mutate the caller's dict on a later merge.
        out.append(dict(block))
    return out


class Agent(BaseModel):
    """A minimal ReAct-style agent loop over any-llm.

    Parameters
    ----------
    model:
        Model identifier, e.g. ``"mistral-small-latest"``.
    provider:
        any-llm provider key, e.g. ``"mistral"``, ``"openai"``, ``"anthropic"``.
    tools:
        Python callables the agent can invoke. ``@tool``-decorated functions,
        YAML tools, and MCP tools carry a pre-built Anthropic-shape schema;
        bare callables get one built from their docstring + signature.
    system:
        Optional system prompt. A ``str`` becomes a single persona text block
        (with ephemeral ``cache_control``); a pre-built Anthropic block list
        is passed through as-is. Sent as the ``amessages`` ``system``
        parameter, never as a ``{role: system}`` message.
    max_iterations:
        Safety cap on the number of LLM round-trips per ``run``.
    max_tokens:
        Output-token cap forwarded to ``amessages``. ``None`` (default) →
        resolved once from the bundled litellm model metadata at the first
        ``amessages`` call (see ``cothis.model_metadata``); an explicit int
        wins. CLI users set this via ``--max-tokens`` / ``COTHIS_MAX_TOKENS``.
    api_key / api_base:
        Forwarded to ``AnyLLM.create``. Default to the provider's env vars.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    provider: str
    tools: list[Tool] = Field(default_factory=list)
    system: str | list[dict[str, Any]] | None = None
    max_iterations: int = 10
    max_tokens: int | None = None
    api_key: str | None = None
    api_base: str | None = None
    cwd: Path | None = None

    # Runtime-only state: not validated, not serialised.
    _llm: AnyLLM = PrivateAttr()
    _tool_map: dict[str, Tool] = PrivateAttr(default_factory=dict)
    # Anthropic-shaped message dicts (user/assistant only). Assistant dicts
    # carry response metadata (id/model/stop_reason/usage); ``_request_messages``
    # strips them before the next ``amessages`` call.
    _messages: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    # Cached ``max_tokens`` resolved from litellm metadata on first use. ``-1``
    # sentinel = not yet resolved; ``resolve_max_tokens`` never returns < 1.
    _resolved_max_tokens: int = PrivateAttr(default=-1)
    _handle_manager: HandleManager = PrivateAttr(default_factory=HandleManager)
    _mcp_servers: list[MCPServer] = PrivateAttr(default_factory=list)
    _mcp_group: Any = PrivateAttr(default=None)
    _mcp_tool_names: set[str] = PrivateAttr(default_factory=set)
    _mcp_started: bool = PrivateAttr(default=False)
    _handles_started: bool = PrivateAttr(default=False)
    # Optional persistence sink. ``ask`` leaves this ``None`` (ephemeral,
    # no Session constructed); ``chat`` calls ``attach_session`` after
    # construction. The three enqueue points (``_ensure_messages`` for the
    # user turn, post-MessageStop for the assistant content, per-execution
    # inside the tool loop for ``tool_result``) all guard on this.
    _session: Session | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        from any_llm import AnyLLM

        self._llm = AnyLLM.create(
            self.provider,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        self._mcp_servers = [t for t in self.tools if isinstance(t, MCPServer)]
        # ponytail: ``_tool_map`` keys are wire-sanitised (``fs.read`` →
        # ``fs_read``) so OpenAI-compatible providers (DeepSeek) accept them.
        # Tool objects keep their original ``__name__`` for routing/logging;
        # the sanitisation is a registration-layer concern only.
        self._tool_map = {}
        for tool in self.tools:
            if isinstance(tool, MCPServer):
                continue
            key = _sanitize_tool_name(tool.__name__)
            if key in self._tool_map:
                # ponytail: collision ceiling — two names sanitising to the
                # same wire key (``fs.read`` vs ``fs_read``) shadow silently.
                # ``discover_tools`` dedupes on the unsanitised ``__name__``,
                # so it can't see these; surface here. Last-write-wins matches
                # the dict-comprehension semantics this replaces.
                logger.warning(
                    "Tool %r shadowed by %r (both map to wire name %r); "
                    "keeping the latter.",
                    self._tool_map[key].__name__,
                    tool.__name__,
                    key,
                )
            self._tool_map[key] = tool
        # Bind the handle manager to every tool that declared a ResourceHandle
        # Tools without ``_handle_cls`` are skipped by ``bind``.
        for tool in self._tool_map.values():
            self._handle_manager.bind(tool)

    def attach_session(self, session: Session) -> None:
        """Bind a :class:`~cothis.session.Session` for persistence.

        ``ask`` never calls this (ephemeral); ``chat`` calls it after
        construction with a fresh or resumed ``Session``. After this,
        ``run_stream``'s three enqueue points fire on every user input,
        every per-execution ``tool_result``, and every assistant
        MessageStop. ``aclose`` drains + closes the session.

        For a **resumed** session (``Session.load``), the loaded
        ``session.messages`` seed ``self._messages`` so the model sees
        prior history on its next ``amessages`` call — without this,
        resume would be amnesiac (the Session has the history but the
        Agent would send an empty conversation). A fresh session
        (``Session.new``) has empty ``messages`` and the seed is a no-op.
        """
        self._session = session
        if session.messages:
            # Replace, not extend: a fresh Agent has no history of its own,
            # and the Session's messages are the ground truth. Rebuild is
            # a shallow copy of each block dict so accidental in-place
            # mutation in one doesn't poison the other.
            self._messages = [
                {"role": m["role"], "content": list(m["content"])}
                for m in session.messages
            ]

    def _effective_max_tokens(self) -> int:
        """The ``max_tokens`` to pass to ``amessages`` for this Agent.

        Resolved lazily on first call via :func:`resolve_max_tokens`, which
        applies the full precedence (override > model > {provider}/model >
        fallback) and the non-positive-override safety (a stray ``0`` from a
        misconfigured env var is treated as "not set"). Cached on the
        instance (not per-call) so the metadata lookup + JSON parse happen
        at most once per ``Agent``.
        """
        if self._resolved_max_tokens < 0:
            self._resolved_max_tokens = resolve_max_tokens(
                self.model, self.provider, self.max_tokens
            )
        return self._resolved_max_tokens

    async def run(self, user_input: str) -> str:
        """Run the agent loop to completion and return the final answer.

        Wraps the body in ``workdir_context(self.cwd)`` so every tool call
        inside the turn sees the same cwd.
        """
        with workdir_context(self.cwd):
            return await self._run_inner(user_input)

    async def _run_inner(self, user_input: str) -> str:
        """Run the agent loop to completion and return the final answer.

        Non-streaming: the full ReAct loop runs to completion before this
        returns. Use ``run_stream`` when the caller wants the final answer
        token-by-token (e.g. ``cothis chat``).

        Turn decision keys on the **presence of ``tool_use`` blocks** in the
        response content (not ``stop_reason``): if the model emitted any
        ``tool_use``, execute them and continue; otherwise return the
        concatenated ``text`` blocks (empty → ``""``). This keeps the
        Anthropic pairing invariant intact even when a generation is cut off
        mid-tool-call by ``max_tokens`` (``stop_reason == "max_tokens"`` but
        a partial ``tool_use`` is still present — keying on ``stop_reason``
        would leave it unpaired and 400 the next turn).

        Side effect: appends the user message, each assistant response, and
        every tool result to ``self._messages``. This is what lets ``chat``
        reuse one Agent across turns — but it also means calling ``run``
        twice on the same instance leaks the first conversation into the
        second. ``ask`` is unaffected because it discards the Agent after a
        single call.
        """
        system_param = self._react_setup(user_input)
        await self._ensure_mcp()
        await self._ensure_handles()

        for _turn in range(self.max_iterations):
            response = cast(
                "MessageResponse",
                await self._llm.amessages(
                    model=self.model,
                    messages=_request_messages(self._messages),
                    max_tokens=self._effective_max_tokens(),
                    system=system_param,
                    tools=self._tool_schemas(),
                ),
            )
            msg = _assistant_msg_from_response(response)
            self._messages.append(msg)
            if self._session is not None:
                self._session.append_message("assistant", msg["content"])

            tool_uses = _tool_uses_in(msg["content"])
            if tool_uses:
                for block in tool_uses:
                    is_error, output = await self._execute_tool(block)
                    self._merge_tool_result(block["id"], output, is_error)
                continue

            # Non-tool turn: final answer (text concat; empty → "").
            answer = _concat_text(msg["content"])
            if not answer:
                # cothis: surface the diagnostic fields the response already
                # carries. ``stop_reason`` tells the user whether it was a
                # ``max_tokens`` cutoff (raise --max-tokens), an ``end_turn``
                # (model chose to say nothing — switch models), or something
                # else. Avoids hard-coding one direction of advice.
                logger.warning(
                    "Model returned empty text (stop_reason=%s, blocks=%s, "
                    "usage=%s).",
                    msg.get("stop_reason"),
                    [b.get("type", "?") for b in msg["content"]],
                    msg.get("usage"),
                )
            return answer

        raise MaxIterationsError(
            f"Agent did not finish within {self.max_iterations} iterations."
        )

    async def run_stream(self, user_input: str) -> AsyncIterator[str | ToolCallEvent]:
        """Run the ReAct loop on Anthropic ``MessageStreamEvent``, yielding deltas.

        Wraps the body in ``workdir_context(self.cwd)`` so every tool call
        inside the turn sees the same cwd.
        """
        with workdir_context(self.cwd):
            async for event in self._run_stream_inner(user_input):
                yield event

    async def _run_stream_inner(self, user_input: str) -> AsyncIterator[str | ToolCallEvent]:
        """Run the ReAct loop on Anthropic ``MessageStreamEvent``, yielding deltas.

        Yields:
            ``str``: a ``TextDelta`` fragment of the model's final answer, as
                soon as it arrives. The CLI accumulates these into a
                Live-rendered Markdown view.
            ``ToolCallEvent``: emitted immediately before each individual
                tool dispatch (not batched), so multi-tool turns surface
                "calling X" → X runs → "calling Y" → Y runs in order.

        The accumulator consumes the Anthropic stream-event union directly
        (any-llm synthesises these events from OpenAI chunks for non-Anthropic
        providers via ``messages_compat``): block state is seeded from
        ``ContentBlockStartEvent.content_block``, mutated by per-block deltas
        (``TextDelta``/``ThinkingDelta`` append, ``SignatureDelta`` overwrites,
        ``InputJSONDelta`` accumulates then parses at block stop). The turn
        decision keys on **presence of ``tool_use`` blocks** (not
        ``stop_reason``) so a ``max_tokens`` cutoff mid-tool-call still
        executes the partial call and keeps the pairing invariant intact;
        ``MessageStopEvent`` is only a stream-termination latch (the first one
        seen ends iteration; openrouter emits duplicates).

        cothis: thinking blocks are accumulated passively (this slice does not
        pass the ``thinking`` param, so claude won't emit them and other
        providers never do); they are replayed verbatim if they ever arrive,
        since stripping them makes the model re-invoke tools.

        Side effect: same as ``run`` — mutates ``self._messages``.
        """
        system_param = self._react_setup(user_input)
        await self._ensure_mcp()
        await self._ensure_handles()

        # Lazy import of the Anthropic stream-event types and ``TextDelta``.
        from anthropic.types import (  # noqa: I001
            RawContentBlockDeltaEvent,
            RawContentBlockStartEvent,
            RawContentBlockStopEvent,
            RawMessageDeltaEvent,
            RawMessageStartEvent,
            RawMessageStopEvent,
        )
        from any_llm.types.messages import TextDelta

        tool_schemas = self._tool_schemas()
        model = self.model
        llm = self._llm
        max_iterations = self.max_iterations

        for _turn in range(max_iterations):
            stream = cast(
                "AsyncIterator[MessageStreamEvent]",
                await llm.amessages(
                    model=model,
                    messages=_request_messages(self._messages),
                    max_tokens=self._effective_max_tokens(),
                    system=system_param,
                    tools=tool_schemas,
                    stream=True,
                ),
            )

            blocks: dict[int, dict[str, Any]] = {}
            stop_reason: str | None = None
            response_id: str | None = None
            response_model: str | None = None
            response_usage: Any = None
            _yielded_text_indexes: set[int] = set()
            async for event in stream:
                if isinstance(event, RawMessageStartEvent):
                    response_id = event.message.id
                    response_model = event.message.model
                elif isinstance(event, RawContentBlockStartEvent):
                    blocks[event.index] = _init_stream_block(event.content_block)
                elif isinstance(event, RawContentBlockDeltaEvent):
                    _apply_stream_delta(blocks[event.index], event.delta)
                    if isinstance(event.delta, TextDelta):
                        yield event.delta.text
                        _yielded_text_indexes.add(event.index)
                elif isinstance(event, RawContentBlockStopEvent):
                    block = blocks.get(event.index)
                    if block is not None:
                        _finalize_stream_block(block)
                elif isinstance(event, RawMessageDeltaEvent):
                    if event.delta.stop_reason is not None:
                        stop_reason = event.delta.stop_reason
                    response_usage = event.usage
                elif isinstance(event, RawMessageStopEvent):
                    # Termination latch: first message_stop ends this turn's
                    # stream. openrouter emits duplicates; break on the first.
                    break

            content = [blocks[i] for i in sorted(blocks)]

            # Safety net: yield any text accumulated in blocks that wasn't
            # yielded during streaming (e.g. isinstance check missed it).
            # Done BEFORE coalesce so list indexes still match block indexes.
            for i, block in enumerate(content):
                if (
                    block.get("type") == "text"
                    and i not in _yielded_text_indexes
                ):
                    text = block.get("text", "")
                    if text:
                        yield text

            # Coalesce adjacent same-type blocks before storing. The
            # OpenAI→Messages stream converter fragments blocks on every
            # reasoning/text transition; without this, a gpt-oss turn can
            # leave an assistant message with dozens of tiny thinking
            # fragments + empty text blocks, which the next provider call
            # silently rejects (chat returns to ``>>>`` with no answer).
            content = _coalesce_content(content)

            self._messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "id": response_id,
                    "model": response_model,
                    "stop_reason": stop_reason,
                    "usage": response_usage.model_dump(exclude_none=True)
                    if response_usage is not None
                    else None,
                }
            )
            # Assistant atomic enqueue (Q2-A): the whole content list shares
            # one txn on drain, so a crash can never leave an orphan
            # ``tool_use`` without its sibling blocks. The stored dict
            # strips the per-inspection metadata (id/model/stop_reason/usage)
            # — Session stores block fields, not message-level metadata
            # (Q11-none: those are None on reload anyway, and
            # ``_request_messages`` strips them before send).
            if self._session is not None:
                self._session.append_message("assistant", content)

            if _tool_uses_in(content):
                for block in content:
                    if block.get("type") != "tool_use":
                        continue
                    # cothis: restore the original (pre-sanitisation) name for
                    # human-facing output; the model emits the wire name.
                    display_name = getattr(
                        self._tool_map.get(block["name"]),
                        "__name__",
                        block["name"],
                    )
                    yield ToolCallEvent(name=display_name, arguments=block["input"])
                    is_error, output = await self._execute_tool(block)
                    self._merge_tool_result(block["id"], output, is_error)
                continue

            # Non-tool turn: final answer streamed already; just end the loop.
            return

        raise MaxIterationsError(
            f"Agent did not finish within {self.max_iterations} iterations."
        )

    # --- internals -----------------------------------------------------

    def _react_setup(self, user_input: str) -> list[dict[str, Any]] | None:
        """Shared setup for ``_run_inner`` and ``_run_stream_inner``.

        Ensures messages, then returns the snapshot system prompt.
        Callers await ``_ensure_mcp`` / ``_ensure_handles`` after.
        """
        self._ensure_messages(user_input)
        return _system_param(self.system)

    def _merge_tool_result(
        self, block_id: str, output: str, is_error: bool
    ) -> None:
        """Merge a ``tool_result`` into ``_messages`` + Session.

        Shared by ``_run_inner`` and ``_run_stream_inner`` — the
        per-execution result enqueue was duplicated verbatim. Single
        source now; adding per-result logic is a one-site edit.
        """
        result_block = _tool_result_block(block_id, output, is_error)
        _append_merged(self._messages, "user", result_block)
        if self._session is not None:
            self._session.append_block("user", result_block)

    async def _ensure_handles(self) -> None:
        """Acquire eager/pinned handles once, on first run (ADR-0005).

        Runs after ``model_post_init`` bound every declared handle and after
        ``_ensure_mcp`` has bound any MCP-session handles, so ``eager`` /
        ``pin`` handles from both sources start together. Idempotent.
        """
        if self._handles_started:
            return
        self._handles_started = True
        await self._handle_manager.start_eager()

    async def _ensure_mcp(self) -> None:
        """Resolve MCP servers into callable tools once, on first run.

        Deferred out of ``model_post_init`` because connecting is async and
        needs a running event loop (ADR-0005). Creates one
        ``ClientSessionGroup`` for the whole Agent (the SDK manages
        connections, tool aggregation, name prefixing, and teardown), then
        connects each declared server into it. Each remote tool the group
        exposes becomes an ``MCPClientTool`` in ``_tool_map``. A server that
        fails to connect contributes no tools (it logs a WARNING itself via
        ``connect_into``) and does not block the others.

        Runs at most once (``_mcp_started`` guard): the group's sessions are
        persistent for the Agent's lifetime, reused across every ``run`` /
        ``run_stream`` call. ``aclose`` tears them down in one ``__aexit__``.
        """
        if self._mcp_started:
            return
        self._mcp_started = True
        if not self._mcp_servers:
            return

        from mcp import ClientSessionGroup

        # Prefix each tool with the server's self-reported
        # ``Implementation.name``, falling back to the YAML ``name:`` label
        # when the server reports an empty name (ADR-0005). The hook fires
        # inside ``connect_to_server`` with nothing identifying *which*
        # cothis server is connecting, so the fallback label travels through
        # a shared mutable cell: the writer sets it immediately before each
        # connect (the startup loop below; ``MCPSessionHandle.acquire`` on
        # every re-acquire), and the hook reads it during that connect.
        # cothis: ceiling — the handoff assumes no two servers connect
        # concurrently (true today: startup, ``start_eager`` and
        # ``ensure_acquired`` all await connects inline). Parallel connects
        # would need the label keyed by something the hook can see.
        fallback: dict[str, str] = {"label": ""}

        def _prefix(name: str, server_info: Any) -> str:
            return f"{server_info.name or fallback['label']}.{name}"

        group = ClientSessionGroup(component_name_hook=_prefix)
        await group.__aenter__()
        self._mcp_group = group
        for server in self._mcp_servers:
            fallback["label"] = server._label
            tools, session = await server.connect_into(group)
            if not tools:
                continue  # connect failed; ``connect_into`` already warned
            # Build a per-server ResourceHandle subclass so each server is one
            # pool entry. The startup connection is adopted as the handle's
            # first acquire (no wasted reconnect): the session is seeded onto
            # the instance, and ``adopt`` marks it live. keepalive/pin come
            # from the YAML declaration (ADR-0005).
            handle_cls = type(
                f"MCPSessionHandle_{server._label}",
                (MCPSessionHandle,),
                {
                    "_group": group,
                    "_params": server.params,
                    "_fallback": fallback,
                    "_fallback_label": server._label,
                    "keepalive": server.keepalive,
                    "pin": server.pin,
                    "eager": server.pin,
                },
            )
            instance = handle_cls()
            instance._session = session
            self._handle_manager.adopt(handle_cls, instance)
            for mcp_tool in tools:
                name = _sanitize_tool_name(mcp_tool.__name__)
                if name in self._tool_map:
                    logger.error(
                        "MCP tool %r skipped: name already registered "
                        "(server %r); keeping the existing tool",
                        mcp_tool.__name__,
                        server.__name__,
                    )
                    continue
                mcp_tool._handle_cls = handle_cls
                self._handle_manager.bind(mcp_tool)
                self._tool_map[name] = mcp_tool
                self._mcp_tool_names.add(name)

    async def aclose(self) -> None:
        """Close the attached Session, MCP session group, and reset for safe reuse.

        ``ask`` calls this after its single ``run``; ``chat`` calls it when
        the session ends. Order: drain + close the Session first (user
        writes are the data the user cares about), then MCP teardown. Tears
        down the ``ClientSessionGroup`` (idempotent — no-op if never
        started), drops the resolved ``MCPClientTool`` entries from
        ``_tool_map`` (tracked by name in ``_mcp_tool_names``, so no
        ``isinstance`` walk), and clears the ``_mcp_started`` guard, so a
        later ``run`` on the same Agent reconnects with fresh sessions
        instead of dispatching against closed ones. Safe to call more than
        once.
        """
        # Drain + close the Session first (R13). ``Session.close`` is sync
        # but cheap (drain + join + sqlite close); we call it directly
        # rather than via ``asyncio.to_thread`` because filelock's lock
        # counter is thread-local — acquiring on the main thread (at
        # ``Session.new``) and releasing on a worker thread would leave
        # the OS lock held forever. ``ask`` path: ``_session is None``,
        # no-op.
        try:
            # cothis: each cleanup step runs in isolation (#140). A raise
            # in one step is logged and the next step still runs; the
            # state reset at the end is wrapped in ``finally`` so it
            # always runs, even if every cleanup raised.
            if self._session is not None:
                try:
                    self._session.close()
                except Exception as exc:  # noqa: BLE001 — teardown must not raise
                    logger.warning("Session close failed: %s", exc)
                self._session = None
            # Release handles first: MCP-session handles disconnect their own
            # sessions via ``disconnect_from_server``. Then exit the group
            # context (closes whatever group-level resources remain). Reversed
            # from the prior order so MCP handles never call
            # ``disconnect_from_server`` on sessions the group already tore down.
            try:
                await self._handle_manager.release_all()
            except Exception as exc:  # noqa: BLE001
                logger.warning("handle release failed: %s", exc)
            if self._mcp_group is not None:
                try:
                    await self._mcp_group.__aexit__(None, None, None)
                except Exception as exc:  # noqa: BLE001 — teardown must not raise
                    logger.debug("MCP group close error: %s", exc)
                self._mcp_group = None
        finally:
            # Remove the MCP-expanded tools by tracked name + reset
            # state — always runs so a later ``run`` reconnects cleanly
            # even if any cleanup raised.
            for name in self._mcp_tool_names:
                self._tool_map.pop(name, None)
            self._mcp_tool_names.clear()
            self._mcp_started = False
            self._handles_started = False

    def _tool_schemas(self) -> list[Any] | None:
        """Tools in Anthropic shape (``{name, description, input_schema}``) for ``amessages``.

        Built from ``_tool_map`` (not ``self.tools``) so it reflects the
        resolved set: MCP server handles are excluded and the
        ``MCPClientTool`` instances they produced (added by ``_ensure_mcp``)
        are included. Delegates each entry to ``tools.schema_for`` so the
        schema-serialisation rule (pre-built Anthropic dict vs callable) lives
        next to the Tool definitions, not here.

        Returns ``None`` when there are no tools.
        """
        if not self._tool_map:
            return None
        # Map keys are already wire-sanitised at registration — use them
        # verbatim so the schema name matches what dispatch looks up.
        return [
            {**schema_for(tool), "name": name}
            for name, tool in self._tool_map.items()
        ]

    def _ensure_messages(self, user_input: str) -> None:
        """Append the user turn to ``self._messages`` as an Anthropic block list.

        The system prompt is sent as the ``amessages`` ``system`` parameter
        (see ``_system_param``), never as a ``{role: system}`` message, so
        this only ever appends a user message. ``chat`` reuses one Agent
        across turns, so each call extends the in-memory history.

        If a Session is attached, the user message is enqueued for durable
        write here (R2) — one site covers both ``run`` and ``run_stream``;
        ``ask``'s ``_session is None`` skips it cheaply.

        Uses ``_append_merged`` (not raw ``append``): a resumed session
        can legitimately end in ``role="user"`` (crash mid-LLM-call, or
        trailing ``tool_result`` with no final assistant), and Anthropic
        rejects consecutive same-role messages with HTTP 400. Merging
        mirrors ``Session.append_block``'s semantics so ``_messages`` and
        ``session.messages`` stay consistent.
        """
        block = {"type": "text", "text": user_input}
        _append_merged(self._messages, "user", block)
        if self._session is not None:
            self._session.append_block("user", block)

    async def _execute_tool(self, tool_use: dict[str, Any]) -> tuple[bool, str]:
        """Dispatch a single ``tool_use`` block; return ``(is_error, output)``.

        Lifecycle (all tools carry hooks via ``_HookableTool``; YAML tools'
        hook chains are empty no-ops):
        1. ``pre_execute`` pipeline — callbacks may modify ``args``.
        2. ``tool(**args)`` — the actual tool body.
        3. ``after_execute`` pipeline — callbacks may modify ``result``.
        4. ``format_tool_output`` — json/csv/tsv/yaml serialisation.

        Any exception in 1–3 fires ``on_error`` (phase names the stage), then
        returns ``(True, error_str)`` so the caller can set the
        Anthropic-native ``is_error`` flag on the ``tool_result`` block.
        ``after_execute`` failure uses the original result (don't hide the
        tool's output). ``tool_use["input"]`` is already a dict (the Messages
        API delivers it parsed), so the old JSON-string parsing is gone.

        Dispatch is async (ADR-0004) to support async tools (MCP). Sync tools
        (ToolDef, ``_ShellTool``, bare callables) return non-coroutine values;
        the ``isawaitable`` check skips the await for them, so their behavior
        is unchanged.
        """
        name = tool_use["name"]
        args: dict[str, Any] = tool_use.get("input") or {}

        tool = self._tool_map.get(name)
        if tool is None:
            logger.debug("tool %s not in tool_map (unknown tool)", name)
            return True, f"Error: unknown tool {name!r}."

        # cothis: the model echoes the wire-sanitised name (``fs.read`` →
        # ``fs_read``); restore the original ``__name__`` so human-facing
        # logs and error messages match the documented tool names.
        name = getattr(tool, "__name__", name)

        try:
            args = run_hooks_safe(tool, "_run_pre_execute", args)
        except Exception as exc:  # noqa: BLE001 — author hook code
            logger.debug("← %s pre_execute raised: %s", name, exc)
            logger.debug("tool %r on_error fired (phase=pre_execute)", name)
            return True, f"Error calling {name}: {exc}"

        # Ensure the tool's ResourceHandle (if any) is acquired before the
        # body runs — self-healing path. No-op for tools without a
        # handle (duck-typed like ``run_hooks_safe``).
        try:
            await ensure_handle_ready(tool)
        except Exception as exc:  # noqa: BLE001 — acquire may fail (network, …)
            logger.debug("← %s handle acquire raised: %s", name, exc)
            run_hooks_safe(tool, "_run_on_error", exc, "handle", args)
            return True, f"Error calling {name}: {exc}"

        # Bracket the body in an in-flight window so the background reaper
        # can't reclaim this handle mid-call. Pairs with handle_call_done
        # in the finally below. No-op for tools without a bound handle.
        # mark_inflight is the FIRST line inside the try: so a failure in
        # the arg repr below still reaches the finally (and balances the
        # refcount). Acquire-raised has already early-returned above, so
        # reaching here means ensure_acquired succeeded.
        try:
            mark_inflight(tool)

            arg_repr = ", ".join(f"{k}={v!r}" for k, v in args.items())
            logger.debug("→ %s(%s)", name, arg_repr)
            result = tool(**args)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
            logger.debug("← %s raised: %s", name, exc)
            logger.debug("tool %r on_error fired (phase=tool)", name)
            run_hooks_safe(tool, "_run_on_error", exc, "tool", args)
            return True, f"Error calling {name}: {exc}"
        finally:
            # End the in-flight window so the reaper can reclaim this
            # handle again; no-op for tools without a bound handle.
            handle_call_done(tool)

        try:
            result = run_hooks_safe(tool, "_run_after_execute", result, args)
        except AfterExecuteError as after_exc:
            logger.debug(
                "← %s after_execute raised: %s; using original result",
                name,
                after_exc.__cause__,
            )
            logger.debug("tool %r on_error fired (phase=after_execute)", name)
            result = after_exc.original_result
        except Exception as exc:  # noqa: BLE001 — bare callable or other edge
            logger.debug(
                "← %s after_execute raised: %s; using original result", name, exc
            )
            logger.debug("tool %r on_error fired (phase=after_execute)", name)

        if isinstance(result, (dict, list)):
            rendered = format_tool_output(result)
        else:
            rendered = str(result)
        logger.debug("← %s: %s", name, rendered)
        return False, rendered
