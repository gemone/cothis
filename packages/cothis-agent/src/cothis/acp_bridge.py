"""``cothis.acp_bridge`` — drives :class:`cothis.agent.Agent` over ACP.

The :class:`AgentSessionBackend` adapts the agent loop to the
:class:`cothis.protocol.acp.SessionBackend` interface so an
:class:`~cothis.protocol.acp.ACPServer` can serve real turns. It keeps an
in-memory session map (persistence is a follow-up) and translates the events
yielded by :meth:`Agent.run_stream` into ``TranscriptProgress`` updates:

* ``ContentDelta``  → ``assistant_delta`` (preceded by one ``item_started``
  for the streaming assistant message).
* ``ToolCallEvent`` → ``item_started`` (a ``running`` tool item).
* ``ToolResultEvent`` → ``item_finished`` (the tool item, ``complete`` or
  ``error``).

A turn ends with one ``item_finished`` for the assistant message
(``stopReason="stop"``). Tool calls split the assistant message: the in-flight
text message is finished with ``stopReason="toolUse"`` before the tool item
starts, matching the streaming lifecycle clients expect.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from cothis.protocol.messages import (
    AssistantDelta,
    AssistantStopReason,
    AssistantTranscriptItem,
    BackendError,
    ItemFinished,
    ItemStarted,
    ModelDescriptor,
    ModelRef,
    ProtocolError,
    SessionSnapshot,
    SessionSummary,
    TextContent,
    ThinkingLevel,
    ToolCallContent,
    ToolTranscriptItem,
    TranscriptItem,
    TranscriptProgress,
    UserTranscriptItem,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from cothis.protocol.acp import ProgressEmitter

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


class _Session:
    """In-memory session state: the agent + the transcript grown so far."""

    __slots__ = ("id", "agent", "cwd", "model", "thinking", "name", "transcript", "created_at", "updated_at")

    def __init__(
        self,
        *,
        sid: str,
        agent: Any,
        cwd: str,
        model: ModelRef,
        thinking: ThinkingLevel,
        name: str | None,
    ) -> None:
        self.id = sid
        self.agent = agent
        self.cwd = cwd
        self.model = model
        self.thinking = thinking
        self.name = name
        self.transcript: list[TranscriptItem] = []
        self.created_at = _now_ms()
        self.updated_at = self.created_at

    def summary(self) -> SessionSummary:
        return SessionSummary(
            id=self.id,
            cwd=self.cwd,
            phase="idle",
            model=self.model,
            thinkingLevel=self.thinking,
            createdAt=self.created_at,
            updatedAt=self.updated_at,
            name=self.name,
        )

    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            id=self.id,
            cwd=self.cwd,
            phase="idle",
            model=self.model,
            thinkingLevel=self.thinking,
            createdAt=self.created_at,
            updatedAt=self.updated_at,
            name=self.name,
            revision=len(self.transcript),
            transcript=list(self.transcript),
        )


class AgentSessionBackend:
    """A :class:`SessionBackend` backed by :class:`Agent`.

    ``make_agent`` builds a fresh :class:`Agent` per session. The default
    factory builds a real agent from the backend's defaults + the
    ``create`` overrides; tests inject a factory returning a fake whose
    ``run_stream`` yields canned events.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        tools: list[Any] | None = None,
        system: Any = None,
        api_key: str | None = None,
        api_base: str | None = None,
        thinking_level: ThinkingLevel = "off",
        make_agent: Callable[..., Any] | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._tools = list(tools or [])
        self._system = system
        self._api_key = api_key
        self._api_base = api_base
        self._thinking = thinking_level
        self._make_agent = make_agent or self._default_make_agent
        self._sessions: dict[str, _Session] = {}

    # ------------------------------------------------------------------ factory

    def _default_make_agent(
        self,
        *,
        cwd: str | None,
        model: str,
        provider: str,
    ) -> Any:
        # Imported lazily so importing this module never pays the SDK load cost.
        from cothis.agent import Agent

        return Agent(
            model=model,
            provider=provider,
            tools=list(self._tools),
            system=self._system,
            api_key=self._api_key,
            api_base=self._api_base,
            cwd=Path(cwd) if cwd else None,
        )

    def _get(self, session_id: str) -> _Session:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise BackendError(
                ProtocolError(
                    code="not_found",
                    message=f"session {session_id!r} not found",
                )
            )
        return sess

    # ------------------------------------------------------------------ backend

    async def models(self) -> list[ModelDescriptor]:
        # The honest advertisement: the single model this server is
        # configured to serve, enriched with the limits bundled litellm
        # metadata can resolve for it (``None`` where unknown — never
        # invented). A multi-model registry is a follow-up.
        from cothis.ai.model_metadata import model_info

        return [
            ModelDescriptor(
                provider=self._provider,
                id=self._model,
                **model_info(self._model, self._provider),
            )
        ]

    async def list_sessions(self) -> list[SessionSummary]:
        return [s.summary() for s in self._sessions.values()]

    async def create_session(
        self,
        cwd: str | None,
        name: str | None,
        model: ModelRef | None,
        thinking_level: ThinkingLevel | None,
    ) -> SessionSnapshot:
        provider = model.provider if model else self._provider
        model_id = model.id if model else self._model
        thinking = thinking_level or self._thinking
        effective_cwd = cwd or os.getcwd()
        agent = self._make_agent(cwd=effective_cwd, model=model_id, provider=provider)
        ref = ModelRef(provider=provider, id=model_id)
        sid = uuid.uuid4().hex
        sess = _Session(
            sid=sid, agent=agent, cwd=effective_cwd, model=ref,
            thinking=thinking, name=name,
        )
        self._sessions[sid] = sess
        return sess.snapshot()

    async def prompt(
        self, session_id: str, text: str, emit: ProgressEmitter
    ) -> SessionSnapshot:
        sess = self._get(session_id)

        # Record the user turn.
        user_item = UserTranscriptItem(
            role="user",
            id=uuid.uuid4().hex,
            content=[TextContent(type="text", text=text)],
            timestamp=_now_ms(),
        )
        sess.transcript.append(user_item)

        translator = _StreamTranslator(model=sess.model, emit=emit)
        try:
            await translator.run(sess.agent.run_stream(text))
        except asyncio.CancelledError:
            # The server's abort cancels its prompt task; the CancelledError
            # lands in translator.run's ``async for``. Finalise the in-flight
            # assistant item as ``aborted`` (bounded emits) and return the
            # aborted snapshot — the prompt task sends it as ``ok=True``.
            await translator.finalize_aborted()
        sess.transcript.extend(translator.finished_items)
        sess.updated_at = _now_ms()
        return sess.snapshot()

    async def abort(self, session_id: str) -> SessionSnapshot:
        """Return the current snapshot for a session (no-op-safe).

        The server cancels any active prompt task before calling this; the
        backend just reports the authoritative post-abort state. Raises
        ``not_found`` (via ``_get``) for an unknown session — never for a
        missing turn.
        """
        return self._get(session_id).snapshot()

    async def set_thinking(
        self, session_id: str, level: ThinkingLevel
    ) -> SessionSnapshot:
        """Update the session's thinking level (metadata-only in this iteration).

        Takes effect on the next ``prompt``. Wiring ``thinking`` into the LLM
        call is a follow-up; this changes the snapshot but not yet model
        behaviour.
        """
        sess = self._get(session_id)
        sess.thinking = level
        return sess.snapshot()

    async def set_model(
        self, session_id: str, model: ModelRef
    ) -> SessionSnapshot:
        """Re-target the session's live agent at a new model, in place.

        Mutates the agent's ``provider`` / ``model`` / ``_llm`` and invalidates
        the cached ``_resolved_max_tokens`` so it re-resolves for the new
        model. Preserves ``_messages`` / ``_tool_map`` / handles / MCP state.
        Cross-provider message-format compatibility is an existing cothis
        concern, not introduced here.

        The mutation is synchronous and not guarded by the per-session busy
        check (only ``prompt`` is): a ``set_model`` arriving while a turn is in
        flight may be observed by that turn's later LLM calls (e.g. a tool
        loop). For this iteration the expectation is best-effort, next-turn
        semantics; robust mid-turn handling is deferred.
        """
        from cothis.ai import get_provider

        sess = self._get(session_id)
        agent = sess.agent
        agent.provider = model.provider
        agent.model = model.id
        agent._llm = get_provider(
            model.provider, api_key=agent.api_key, api_base=agent.api_base
        )
        agent._resolved_max_tokens = -1
        sess.model = model
        return sess.snapshot()


class _StreamTranslator:
    """Converts one ``Agent.run_stream`` turn into TranscriptProgress emits.

    Tracks at most one in-flight assistant message; a tool call finishes it
    (``stopReason="toolUse"``) before the tool item starts. The assistant
    message is finished with ``stopReason="stop"`` at turn end.
    """

    def __init__(
        self, *, model: ModelRef, emit: ProgressEmitter
    ) -> None:
        self._model = model
        self._emit = emit
        self.finished_items: list[TranscriptItem] = []
        self._msg_id: str | None = None
        self._content: list[Any] = []
        self._content_index_by_kind: dict[str, int] = {}
        self._assistant_started = False

    async def run(self, stream: AsyncIterator[Any]) -> None:
        async for event in stream:
            name = type(event).__name__
            if name == "ContentDelta":
                await self._on_content_delta(event)
            elif name == "ToolCallEvent":
                await self._on_tool_call(event)
            elif name == "ToolResultEvent":
                await self._on_tool_result(event)
            # AskUserRequestEvent is not surfaced over ACP in I9.
        await self._finish_assistant(stop_reason="stop")

    async def finalize_aborted(self) -> None:
        """Finalise an in-flight assistant message as ``aborted``.

        Called from the backend's ``CancelledError`` path when the server
        aborts a turn mid-stream. Emits ``item_finished`` with
        ``stopReason="aborted"`` and appends the item (status ``"aborted"``)
        to ``finished_items``. No-op if nothing streamed before the
        cancellation.
        """
        await self._finish_assistant(stop_reason="aborted", status="aborted")

    # -- assistant message lifecycle ----------------------------------------

    def _start_assistant(self, kind: str) -> None:
        if self._assistant_started:
            return
        self._msg_id = uuid.uuid4().hex
        self._content = []
        self._content_index_by_kind = {}
        self._assistant_started = True

    async def _on_content_delta(self, delta: Any) -> None:
        if not self._assistant_started:
            self._start_assistant(delta.kind)
            await self._emit(
                ItemStarted(
                    type="item_started",
                    item=self._streaming_item(),
                )
            )
        idx = self._content_index_by_kind.setdefault(delta.kind, len(self._content))
        if idx == len(self._content):
            self._content.append({"type": "text", "text": ""})
        # Accumulate so the finished item's content is authoritative.
        self._content[idx]["text"] += delta.text
        await self._emit(
            AssistantDelta(
                type="assistant_delta",
                messageId=self._msg_id or "",
                contentIndex=idx,
                kind=delta.kind,
                delta=delta.text,
            )
        )

    def _streaming_item(self) -> AssistantTranscriptItem:
        return AssistantTranscriptItem(
            role="assistant",
            id=self._msg_id or "",
            content=list(self._content),
            model=self._model,
            status="streaming",
            timestamp=_now_ms(),
        )

    async def _finish_assistant(
        self,
        *,
        stop_reason: AssistantStopReason,
        status: Literal["complete", "aborted"] = "complete",
    ) -> None:
        if not self._assistant_started or self._msg_id is None:
            return
        item = AssistantTranscriptItem(
            role="assistant",
            id=self._msg_id,
            content=list(self._content),
            model=self._model,
            status=status,
            stopReason=stop_reason,
            timestamp=_now_ms(),
        )
        await self._emit(ItemFinished(type="item_finished", item=item))
        self.finished_items.append(item)  # type: ignore[arg-type]
        self._msg_id = None
        self._content = []
        self._content_index_by_kind = {}
        self._assistant_started = False

    # -- tool lifecycle -----------------------------------------------------

    async def _on_tool_call(self, event: Any) -> None:
        # A tool call ends any in-flight assistant message first.
        await self._finish_assistant(stop_reason="toolUse")
        tool_item = ToolTranscriptItem(
            role="tool",
            id=event.call_id,
            toolCallId=event.call_id,
            toolName=event.name,
            input=event.arguments,
            content=[],
            status="running",
            isError=False,
            timestamp=_now_ms(),
        )
        await self._emit(ItemStarted(type="item_started", item=tool_item))

    async def _on_tool_result(self, event: Any) -> None:
        tool_item = ToolTranscriptItem(
            role="tool",
            id=event.call_id,
            toolCallId=event.call_id,
            toolName=event.tool,
            input={},
            content=[TextContent(type="text", text="")],
            status="error" if event.is_error else "complete",
            isError=bool(event.is_error),
            timestamp=_now_ms(),
        )
        await self._emit(ItemFinished(type="item_finished", item=tool_item))
        self.finished_items.append(tool_item)  # type: ignore[arg-type]


__all__ = ["AgentSessionBackend"]
