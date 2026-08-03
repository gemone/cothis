"""ACP-ready protocol message/event types (clean-room seed).

Mirrors the *architecture* of pi's ``packages/protocol`` schemas — a small
set of top-level wire messages discriminated by a ``type`` field — in
idiomatic pydantic v2. This is the envelope spine; transcript/progress/
command unions and the binary wire format (CBOR + framing) land in later
iterations alongside the ACP server. Nothing outside this package's own
tests imports it yet, so it is safe to evolve.

Clean-room: no pi TypeScript is copied; only the layering/shape ideas are
reflected in idiomatic Python.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

# cothis protocol version (integer, mirroring pi's explicit versioning).
# Will align with the ACP spec version as the server lands.
PROTOCOL_VERSION: int = 1


def is_supported_protocol_version(version: int) -> bool:
    """True only for the protocol version this build speaks."""
    return version == PROTOCOL_VERSION


ProtocolErrorCode = Literal[
    "auth",
    "version",
    "busy",
    "session_locked",
    "not_found",
    "invalid_request",
]


class ProtocolError(BaseModel):
    """A structured protocol error (mirrors pi's ``ProtocolError``)."""

    model_config = ConfigDict(extra="forbid")

    code: ProtocolErrorCode
    message: str
    details: JsonValue | None = None


class ClientHello(BaseModel):
    """First message from a client: declare protocol version + auth token."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["hello"]
    version: int = Field(ge=0)
    token: str = Field(min_length=1)


class ServerHello(BaseModel):
    """Server greeting: pins the supported protocol version + connection id."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["hello"]
    version: int
    connection_id: str

    @model_validator(mode="after")
    def _pin_protocol_version(self) -> ServerHello:
        if self.version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version {self.version}")
        return self


class RequestEnvelope(BaseModel):
    """A client request. ``request`` is loosely typed now (the Command union
    lands with the ACP server) and narrowed later."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["request"]
    id: str
    request: JsonValue


class ResponseEnvelope(BaseModel):
    """A server response: exactly one of ``result`` (ok) / ``error``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["response"]
    id: str
    result: JsonValue | None = None
    error: ProtocolError | None = None

    @model_validator(mode="after")
    def _exactly_one_of_result_error(self) -> ResponseEnvelope:
        if (self.result is None) == (self.error is None):
            raise ValueError("exactly one of result/error must be set")
        return self


class EventEnvelope(BaseModel):
    """A server-initiated event. ``event`` is loosely typed now (the
    ServerEvent union lands with the ACP server)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["event"]
    event: JsonValue


ClientMessage = Annotated[
    ClientHello | RequestEnvelope, Field(discriminator="type")
]
ServerMessage = Annotated[
    ServerHello | ResponseEnvelope | EventEnvelope, Field(discriminator="type")
]


__all__ = [
    "PROTOCOL_VERSION",
    "is_supported_protocol_version",
    "ProtocolErrorCode",
    "ProtocolError",
    "ClientHello",
    "ServerHello",
    "RequestEnvelope",
    "ResponseEnvelope",
    "EventEnvelope",
    "ClientMessage",
    "ServerMessage",
]
