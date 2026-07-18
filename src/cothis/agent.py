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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

# Runtime imports of the Anthropic stream-event types: ``isinstance`` dispatch
# in ``run_stream`` is how ty narrows the ``MessageStreamEvent`` union (it
# can't narrow by string ``event.type`` comparison). The union is itself just
# these anthropic SDK classes (any-llm re-exports them).
from anthropic.types import (  # noqa: I001
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)
from any_llm.types.messages import TextDelta
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# cothis: ``Tool`` must be runtime-imported (not TYPE_CHECKING-only) because
# pydantic resolves the ``list[Tool]`` field annotation at model-build time
# via ``typing.get_type_hints``, which needs ``Tool`` in the module globals.
# ``from __future__ import annotations`` makes the annotation a string, so
# ruff's TC001 rule can't see the runtime use and wants it moved under
# TYPE_CHECKING — which would crash pydantic. This noqa is the honest
# representation of that constraint.
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
from cothis.tools.mcp import MCPSessionHandle

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from any_llm import AnyLLM
    from any_llm.types.messages import MessageResponse, MessageStreamEvent

    from cothis.tools import MCPClientTool


Message = dict[str, Any]

logger = logging.getLogger("cothis.agent")

# cothis: ``max_tokens`` is hardcoded this slice; slice #32 resolves it from
# the bundled litellm ``model_prices.json`` (fallback 8192), overridable via
# ``COTHIS_MAX_TOKENS`` / ``--max-tokens``. Upgrade path: replace this const
# with the resolver once #32 lands.
_DEFAULT_MAX_TOKENS = 8192

_EPHEMERAL_CACHE: dict[str, str] = {"type": "ephemeral"}


def _system_param(system: str | list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Build the ``amessages`` ``system`` parameter as a block list.

    A ``str`` persona becomes a single text block carrying
    ``cache_control: {type: ephemeral}``. A pre-built block list is passed
    through unchanged (the caller owns ``cache_control`` placement — e.g. the
    AGENTS.md assembler in #33). ``None`` → ``None`` (no system param sent).
    """
    if system is None:
        return None
    if isinstance(system, str):
        return [
            {"type": "text", "text": system, "cache_control": dict(_EPHEMERAL_CACHE)}
        ]
    return system


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


def _concat_text(content: list[dict[str, Any]]) -> str:
    """Concatenate every ``text`` block in an assistant content list."""
    return "".join(
        b.get("text", "") for b in content if b.get("type") == "text"
    )


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


class Agent(BaseModel):
    """A minimal ReAct-style agent loop over any-llm.

    Parameters
    ----------
    model:
        Model identifier, e.g. ``"mistral-small-latest"``.
    provider:
        any-llm provider key, e.g. ``"mistral"``, ``"openai"``, ``"anthropic"``.
    tools:
        Python callables the agent can invoke. Each must have a docstring
        and type annotations; any-llm converts them into the provider's
        tool schema automatically.
    system:
        Optional system prompt. A ``str`` becomes a single persona text block
        (with ephemeral ``cache_control``); a pre-built Anthropic block list
        is passed through as-is. Sent as the ``amessages`` ``system``
        parameter, never as a ``{role: system}`` message.
    max_iterations:
        Safety cap on the number of LLM round-trips per ``run``.
    api_key / api_base:
        Forwarded to ``AnyLLM.create``. Default to the provider's env vars.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    provider: str
    tools: list[Tool] = Field(default_factory=list)
    system: str | list[dict[str, Any]] | None = None
    max_iterations: int = 10
    api_key: str | None = None
    api_base: str | None = None

    # Runtime-only state: not validated, not serialised.
    _llm: AnyLLM = PrivateAttr()
    _tool_map: dict[str, Tool] = PrivateAttr(default_factory=dict)
    # Anthropic-shaped message dicts (user/assistant only). Assistant dicts
    # carry response metadata (id/model/stop_reason/usage); ``_request_messages``
    # strips them before the next ``amessages`` call.
    _messages: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _handle_manager: HandleManager = PrivateAttr(default_factory=HandleManager)
    _mcp_servers: list[MCPServer] = PrivateAttr(default_factory=list)
    _mcp_group: Any = PrivateAttr(default=None)
    _mcp_tool_names: set[str] = PrivateAttr(default_factory=set)
    _mcp_started: bool = PrivateAttr(default=False)
    _handles_started: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: Any) -> None:
        from any_llm import AnyLLM

        self._llm = AnyLLM.create(
            self.provider,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        self._mcp_servers = [t for t in self.tools if isinstance(t, MCPServer)]
        self._tool_map = {
            tool.__name__: tool
            for tool in self.tools
            if not isinstance(tool, MCPServer)
        }
        # Bind the handle manager to every tool that declared a ResourceHandle
        # Tools without ``_handle_cls`` are skipped by ``bind``.
        for tool in self._tool_map.values():
            self._handle_manager.bind(tool)

    async def run(self, user_input: str) -> str:
        """Run the agent loop to completion and return the final answer.

        Non-streaming: the full ReAct loop runs to completion before this
        returns. Use ``run_stream`` when the caller wants the final answer
        token-by-token (e.g. ``cothis chat``).

        Turn decision uses ``response.stop_reason`` (``== "tool_use"`` → tool
        turn); any other ``stop_reason`` ends the loop and returns the
        concatenated ``text`` blocks (empty → ``""``). This removes the old
        content-None retry heuristic: ``stop_reason`` is the authoritative
        end-of-turn signal, so there is no "did the provider drop content?"
        ambiguity to retry through.

        Side effect: appends the user message, each assistant response, and
        every tool result to ``self._messages``. This is what lets ``chat``
        reuse one Agent across turns — but it also means calling ``run``
        twice on the same instance leaks the first conversation into the
        second. ``ask`` is unaffected because it discards the Agent after a
        single call.
        """
        self._ensure_messages(user_input)
        await self._ensure_mcp()
        await self._ensure_handles()

        for _turn in range(self.max_iterations):
            response = cast(
                "MessageResponse",
                await self._llm.amessages(
                    model=self.model,
                    messages=_request_messages(self._messages),
                    max_tokens=_DEFAULT_MAX_TOKENS,
                    system=_system_param(self.system),
                    tools=self._tool_schemas(),
                ),
            )
            msg = _assistant_msg_from_response(response)
            self._messages.append(msg)

            if response.stop_reason == "tool_use":
                result_blocks: list[dict[str, Any]] = []
                for block in msg["content"]:
                    if block.get("type") != "tool_use":
                        continue
                    is_error, output = await self._execute_tool(block)
                    result_blocks.append(
                        _tool_result_block(block["id"], output, is_error)
                    )
                self._messages.append({"role": "user", "content": result_blocks})
                continue

            # Non-tool turn: final answer (text concat; empty → "").
            return _concat_text(msg["content"])

        raise MaxIterationsError(
            f"Agent did not finish within {self.max_iterations} iterations."
        )

    async def run_stream(self, user_input: str) -> AsyncIterator[str | ToolCallEvent]:
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
        decision uses ``stop_reason`` from ``MessageDeltaEvent``;
        ``MessageStopEvent`` is only a stream-termination latch (the first one
        seen ends iteration; openrouter emits duplicates).

        cothis: thinking blocks are accumulated passively (this slice does not
        pass the ``thinking`` param, so claude won't emit them and other
        providers never do); they are replayed verbatim if they ever arrive,
        since stripping them makes the model re-invoke tools.

        Side effect: same as ``run`` — mutates ``self._messages``.
        """
        self._ensure_messages(user_input)
        await self._ensure_mcp()
        await self._ensure_handles()

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
                    max_tokens=_DEFAULT_MAX_TOKENS,
                    system=_system_param(self.system),
                    tools=tool_schemas,
                    stream=True,
                ),
            )

            blocks: dict[int, dict[str, Any]] = {}
            stop_reason: str | None = None
            response_id: str | None = None
            response_model: str | None = None
            response_usage: Any = None
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

            if stop_reason == "tool_use":
                result_blocks = []
                for block in content:
                    if block.get("type") != "tool_use":
                        continue
                    yield ToolCallEvent(
                        name=block["name"], arguments=block["input"]
                    )
                    is_error, output = await self._execute_tool(block)
                    result_blocks.append(
                        _tool_result_block(block["id"], output, is_error)
                    )
                self._messages.append({"role": "user", "content": result_blocks})
                continue

            # Non-tool turn: final answer streamed already; just end the loop.
            return

        raise MaxIterationsError(
            f"Agent did not finish within {self.max_iterations} iterations."
        )

    # --- internals -----------------------------------------------------

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
                if mcp_tool.__name__ in self._tool_map:
                    logger.error(
                        "MCP tool %r skipped: name already registered "
                        "(server %r); keeping the existing tool",
                        mcp_tool.__name__,
                        server.__name__,
                    )
                    continue
                mcp_tool._handle_cls = handle_cls
                self._handle_manager.bind(mcp_tool)
                self._tool_map[mcp_tool.__name__] = mcp_tool
                self._mcp_tool_names.add(mcp_tool.__name__)

    async def aclose(self) -> None:
        """Close the MCP session group and reset for safe reuse.

        ``ask`` calls this after its single ``run``; ``chat`` calls it when
        the session ends. Tears down the ``ClientSessionGroup`` (idempotent —
        no-op if never started), drops the resolved ``MCPClientTool`` entries
        from ``_tool_map`` (tracked by name in ``_mcp_tool_names``, so no
        ``isinstance`` walk), and clears the ``_mcp_started`` guard, so a
        later ``run`` on the same Agent reconnects with fresh sessions
        instead of dispatching against closed ones. Safe to call more than
        once.
        """
        # Release handles first: MCP-session handles disconnect their own
        # sessions via ``disconnect_from_server``. Then exit the group
        # context (closes whatever group-level resources remain). Reversed
        # from the prior order so MCP handles never call
        # ``disconnect_from_server`` on sessions the group already tore down.
        await self._handle_manager.release_all()
        if self._mcp_group is not None:
            try:
                await self._mcp_group.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 — teardown must not raise
                logger.debug("MCP group close error: %s", exc)
            self._mcp_group = None
        # Remove the MCP-expanded tools by tracked name.
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
        return [schema_for(tool) for tool in self._tool_map.values()]

    def _ensure_messages(self, user_input: str) -> None:
        """Append the user turn to ``self._messages`` as an Anthropic block list.

        The system prompt is sent as the ``amessages`` ``system`` parameter
        (see ``_system_param``), never as a ``{role: system}`` message, so
        this only ever appends a user message. ``chat`` reuses one Agent
        across turns, so each call extends the in-memory history.
        """
        self._messages.append(
            {"role": "user", "content": [{"type": "text", "text": user_input}]}
        )

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
