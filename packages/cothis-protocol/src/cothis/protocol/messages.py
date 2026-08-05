"""ACP (Agent Client Protocol) message, command, and event types.

The wire spine (hello handshake, request/response/event envelopes,
``ProtocolError``) was seeded earlier with two unions deliberately left
loose — ``RequestEnvelope.request`` and ``EventEnvelope.event`` were typed
``JsonValue``. This module completes them: the ``Command`` / ``CommandResult``
unions, the ``ServerEvent`` union, transcript items, ``TranscriptProgress``,
and session/server snapshots. Together they are what "open ACP" means at the
type level — a client can now drive cothis by sending real commands and
consuming real streaming events.

Design is clean-room: the layering and the discriminated-union shape follow
the reference project's protocol package, implemented idiomatically in
pydantic v2. No code is copied.

Wire format lives in :mod:`cothis.protocol.wire` (length-prefixed frames +
a JSON codec). CBOR is a follow-up; the codec is an isolated seam.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

# cothis protocol version (integer). Pinned in the handshake; a mismatched
# version is a hard error (``version`` ProtocolError).
PROTOCOL_VERSION: int = 1


def is_supported_protocol_version(version: int) -> bool:
    """True only for the protocol version this build speaks."""
    return version == PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

ProtocolErrorCode = Literal[
    "auth",
    "version",
    "busy",
    "session_locked",
    "not_found",
    "invalid_request",
]


class ProtocolError(BaseModel):
    """A structured protocol error returned to the client."""

    model_config = ConfigDict(extra="forbid")

    code: ProtocolErrorCode
    message: str
    details: JsonValue | None = None


class BackendError(Exception):
    """Raised by a session backend to signal a protocol-level failure.

    Wraps a :class:`ProtocolError` (a model, not an exception) so the server
    can ``except BackendError`` and translate ``.error`` into an error
    response. Backend code raises ``BackendError(ProtocolError(...))``.
    """

    def __init__(self, error: ProtocolError) -> None:
        self.error = error
        super().__init__(error.message)


# ---------------------------------------------------------------------------
# Shared scalars / enums
# ---------------------------------------------------------------------------

ThinkingLevel = Literal[
    "off", "minimal", "low", "medium", "high", "xhigh", "max"
]

SessionPhase = Literal[
    "idle", "turn", "compaction", "branch_summary", "retry"
]

AssistantStopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


class ModelRef(BaseModel):
    """A ``(provider, id)`` model reference."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    id: str = Field(min_length=1)


class ModelDescriptor(BaseModel):
    """A model advertised in :class:`ServerSnapshot.models`.

    The ``(provider, id)`` pair identifies the model; the two optional limits
    are populated when cothis's bundled metadata resolves them for that model
    (``None`` means "unknown" — never invented). Only the model the server is
    configured to serve is advertised today; a multi-model registry is a
    follow-up.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    id: str = Field(min_length=1)
    maxOutputTokens: int | None = None
    contextWindow: int | None = None


# ---------------------------------------------------------------------------
# Content + transcript items
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str


class ToolCallContent(BaseModel):
    """A tool call embedded in an assistant message."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["toolCall"]
    toolCallId: str = Field(min_length=1)
    toolName: str = Field(min_length=1)
    input: JsonValue


# Assistant content is text and/or tool calls (thinking/image deferred).
AssistantContent = Annotated[
    TextContent | ToolCallContent, Field(discriminator="type")
]
# User + tool-result content is plain text for now.
UserContent = TextContent
ToolContent = TextContent


class UserTranscriptItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user"]
    id: str = Field(min_length=1)
    content: list[UserContent]
    timestamp: int = Field(ge=0)


class AssistantTranscriptItem(BaseModel):
    """An assistant turn. ``status`` tracks the streaming lifecycle."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"]
    id: str = Field(min_length=1)
    content: list[AssistantContent]
    model: ModelRef
    status: Literal["streaming", "complete", "error", "aborted"]
    stopReason: AssistantStopReason | None = None
    errorMessage: str | None = None
    timestamp: int = Field(ge=0)


class ToolTranscriptItem(BaseModel):
    """A tool invocation + its result. ``status`` tracks execution."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["tool"]
    id: str = Field(min_length=1)
    toolCallId: str = Field(min_length=1)
    toolName: str = Field(min_length=1)
    input: JsonValue
    content: list[ToolContent]
    status: Literal["running", "complete", "error"]
    isError: bool = False
    timestamp: int = Field(ge=0)


# Transcript items are discriminated by ``role``.
TranscriptItem = Annotated[
    UserTranscriptItem | AssistantTranscriptItem | ToolTranscriptItem,
    Field(discriminator="role"),
]


# ---------------------------------------------------------------------------
# Transcript progress (incremental streaming within a turn)
# ---------------------------------------------------------------------------


class ItemStarted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["item_started"]
    item: TranscriptItem


class AssistantDelta(BaseModel):
    """An incremental delta for a streaming assistant message."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["assistant_delta"]
    messageId: str = Field(min_length=1)
    contentIndex: int = Field(ge=0)
    kind: Literal["text", "thinking", "toolCall"]
    delta: str


class ItemUpdated(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["item_updated"]
    item: AssistantTranscriptItem | ToolTranscriptItem


class ItemFinished(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["item_finished"]
    item: AssistantTranscriptItem | ToolTranscriptItem


TranscriptProgress = Annotated[
    ItemStarted | AssistantDelta | ItemUpdated | ItemFinished,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    cwd: str = Field(min_length=1)
    phase: SessionPhase
    model: ModelRef
    thinkingLevel: ThinkingLevel
    createdAt: int = Field(ge=0)
    updatedAt: int = Field(ge=0)
    name: str | None = None
    attached: bool = False
    locked: bool = False


class SessionSnapshot(BaseModel):
    """A session summary plus its authoritative transcript."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    cwd: str = Field(min_length=1)
    phase: SessionPhase
    model: ModelRef
    thinkingLevel: ThinkingLevel
    createdAt: int = Field(ge=0)
    updatedAt: int = Field(ge=0)
    name: str | None = None
    attached: bool = False
    locked: bool = False
    revision: int = Field(ge=0)
    transcript: list[TranscriptItem]
    queuedSteerCount: int = Field(ge=0, default=0)


class ServerSnapshot(BaseModel):
    """Whole-server state sent in the handshake + ``server_snapshot``."""

    model_config = ConfigDict(extra="forbid")

    serverId: str = Field(min_length=1)
    protocolVersion: int
    revision: int = Field(ge=0)
    sessions: list[SessionSummary]
    # Models advertised to clients (the configured model + its resolved limits).
    models: list[ModelDescriptor] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Commands + results
# ---------------------------------------------------------------------------


class ListCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["list"]


class CreateCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["create"]
    cwd: str | None = None
    name: str | None = None
    model: ModelRef | None = None
    thinkingLevel: ThinkingLevel | None = None


class AttachCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["attach"]
    sessionId: str = Field(min_length=1)


class DetachCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["detach"]
    sessionId: str = Field(min_length=1)


class PromptCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["prompt"]
    sessionId: str = Field(min_length=1)
    text: str


class SteerCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["steer"]
    sessionId: str = Field(min_length=1)
    text: str


class AbortCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["abort"]
    sessionId: str = Field(min_length=1)


class SetModelCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["set_model"]
    sessionId: str = Field(min_length=1)
    model: ModelRef


class SetThinkingCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["set_thinking"]
    sessionId: str = Field(min_length=1)
    thinkingLevel: ThinkingLevel


Command = Annotated[
    ListCommand | CreateCommand | AttachCommand | DetachCommand | PromptCommand | SteerCommand | AbortCommand | SetModelCommand | SetThinkingCommand,
    Field(discriminator="command"),
]


class ListResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["list"]
    sessions: list[SessionSummary]


class _SessionResult(BaseModel):
    """Shared shape for results that return a single session snapshot."""

    model_config = ConfigDict(extra="forbid")

    session: SessionSnapshot


class CreateResult(_SessionResult):
    command: Literal["create"]


class AttachResult(_SessionResult):
    command: Literal["attach"]


class PromptResult(_SessionResult):
    command: Literal["prompt"]


class SteerResult(_SessionResult):
    command: Literal["steer"]


class AbortResult(_SessionResult):
    command: Literal["abort"]


class SetModelResult(_SessionResult):
    command: Literal["set_model"]


class SetThinkingResult(_SessionResult):
    command: Literal["set_thinking"]


class DetachResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Literal["detach"]
    sessionId: str = Field(min_length=1)


CommandResult = Annotated[
    ListResult | CreateResult | AttachResult | DetachResult | PromptResult | SteerResult | AbortResult | SetModelResult | SetThinkingResult,
    Field(discriminator="command"),
]


# ---------------------------------------------------------------------------
# Server events
# ---------------------------------------------------------------------------


class ServerSnapshotEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["server_snapshot"]
    snapshot: ServerSnapshot


class SessionSnapshotEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["session_snapshot"]
    snapshot: SessionSnapshot


class SessionProgressEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["session_progress"]
    sessionId: str = Field(min_length=1)
    progress: TranscriptProgress


class SessionRemovedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["session_removed"]
    sessionId: str = Field(min_length=1)


ServerEvent = Annotated[
    ServerSnapshotEvent | SessionSnapshotEvent | SessionProgressEvent | SessionRemovedEvent,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Envelopes + handshake
# ---------------------------------------------------------------------------


class ClientHello(BaseModel):
    """First message from a client: declare protocol version + auth token."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["hello"]
    version: int = Field(ge=0)
    token: str = Field(min_length=1)


class ServerHello(BaseModel):
    """Server greeting: pins the protocol version, connection id, snapshot."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["hello"]
    version: int
    connectionId: str = Field(min_length=1)
    snapshot: ServerSnapshot

    @model_validator(mode="after")
    def _pin_protocol_version(self) -> ServerHello:
        if self.version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version {self.version}")
        return self


class ServerHelloError(BaseModel):
    """Sent instead of ``ServerHello`` when the handshake fails, then close."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["hello_error"]
    error: ProtocolError


class RequestEnvelope(BaseModel):
    """A client request carrying one ``Command``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["request"]
    id: str = Field(min_length=1)
    request: Command


class ResponseEnvelope(BaseModel):
    """A server response: ``ok`` carries the result, else ``error``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["response"]
    id: str = Field(min_length=1)
    ok: bool
    result: CommandResult | None = None
    error: ProtocolError | None = None

    @model_validator(mode="after")
    def _ok_matches_payload(self) -> ResponseEnvelope:
        if self.ok and (self.result is None or self.error is not None):
            raise ValueError("ok=True requires result and no error")
        if not self.ok and (self.error is None or self.result is not None):
            raise ValueError("ok=False requires error and no result")
        return self


class EventEnvelope(BaseModel):
    """A server-initiated event carrying one ``ServerEvent``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["event"]
    event: ServerEvent


ClientMessage = Annotated[
    ClientHello | RequestEnvelope, Field(discriminator="type")
]
ServerMessage = Annotated[
    ServerHello | ServerHelloError | ResponseEnvelope | EventEnvelope,
    Field(discriminator="type"),
]


__all__ = [
    # version
    "PROTOCOL_VERSION",
    "is_supported_protocol_version",
    # errors + enums
    "ProtocolErrorCode",
    "ProtocolError",
    "BackendError",
    "ThinkingLevel",
    "SessionPhase",
    "AssistantStopReason",
    "ModelRef",
    "ModelDescriptor",
    # content + transcript
    "TextContent",
    "ToolCallContent",
    "AssistantContent",
    "UserContent",
    "ToolContent",
    "UserTranscriptItem",
    "AssistantTranscriptItem",
    "ToolTranscriptItem",
    "TranscriptItem",
    # progress
    "ItemStarted",
    "AssistantDelta",
    "ItemUpdated",
    "ItemFinished",
    "TranscriptProgress",
    # snapshots
    "SessionSummary",
    "SessionSnapshot",
    "ServerSnapshot",
    # commands + results
    "ListCommand",
    "CreateCommand",
    "AttachCommand",
    "DetachCommand",
    "PromptCommand",
    "SteerCommand",
    "AbortCommand",
    "SetModelCommand",
    "SetThinkingCommand",
    "Command",
    "ListResult",
    "CreateResult",
    "AttachResult",
    "DetachResult",
    "PromptResult",
    "SteerResult",
    "AbortResult",
    "SetModelResult",
    "SetThinkingResult",
    "CommandResult",
    # events
    "ServerSnapshotEvent",
    "SessionSnapshotEvent",
    "SessionProgressEvent",
    "SessionRemovedEvent",
    "ServerEvent",
    # envelopes + handshake
    "ClientHello",
    "ServerHello",
    "ServerHelloError",
    "RequestEnvelope",
    "ResponseEnvelope",
    "EventEnvelope",
    "ClientMessage",
    "ServerMessage",
]
