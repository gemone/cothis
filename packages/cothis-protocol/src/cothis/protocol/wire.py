"""``cothis.protocol.wire`` — length-prefixed framing + message codec.

Wire format: each message is a *frame* = a 4-byte big-endian unsigned
32-bit length header followed by that many payload bytes. The payload is
UTF-8 JSON in this iteration; the codec is an isolated seam
(:func:`_encode_payload` / :func:`_decode_payload`) so a CBOR payload is a
drop-in follow-up that touches only those two functions.

Two layers, matching the reference design:

* :class:`FrameDecoder` — incremental, transport-agnostic. Feed it arbitrary
  byte chunks; it yields complete payloads, enforcing a max-frame bound.
* :func:`encode_frame` — the inverse: prefix a payload with its length.
* Message codec — :func:`encode_client_message` / :func:`encode_server_message`
  validate a message against the schema (pydantic), frame it, and return
  bytes ready to send. :class:`ClientMessageDecoder` /
  :class:`ServerMessageDecoder` wrap a :class:`FrameDecoder` and parse each
  payload back into a validated model instance.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter, ValidationError

from cothis.protocol.messages import (
    ClientMessage,
    ServerMessage,
)

FRAME_HEADER_LENGTH = 4
_MAX_UINT32 = 0xFFFF_FFFF
#: Default upper bound for one framed payload (16 MiB).
DEFAULT_MAX_FRAME_LENGTH = 16 * 1024 * 1024


class FrameError(Exception):
    """A framing violation (truncated frame, oversize length, …)."""


class ProtocolValidationError(Exception):
    """A payload that does not satisfy the protocol schema."""


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def _resolve_max_frame_length(max_frame_length: int | None) -> int:
    value = DEFAULT_MAX_FRAME_LENGTH if max_frame_length is None else max_frame_length
    if not isinstance(value, int) or value < 0 or value > _MAX_UINT32:
        raise ValueError(
            f"max_frame_length must be an integer between 0 and {_MAX_UINT32}"
        )
    return value


def encode_frame(payload: bytes, *, max_frame_length: int | None = None) -> bytes:
    """Prefix *payload* with its unsigned 32-bit big-endian byte length."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("Frame payload must be bytes")
    length = len(payload)
    if length > _MAX_UINT32:
        raise ValueError("Frame payload exceeds the unsigned 32-bit length limit")
    if max_frame_length is not None and length > max_frame_length:
        raise FrameError(
            f"Frame length {length} exceeds configured limit of {max_frame_length}"
        )
    return length.to_bytes(FRAME_HEADER_LENGTH, "big") + bytes(payload)


class FrameDecoder:
    """Incrementally split arbitrary byte chunks into length-prefixed payloads.

    Stateful: call :meth:`push` for each inbound chunk and collect the
    complete payloads it returns. :meth:`end` asserts the stream finished
    on a frame boundary.
    """

    def __init__(self, *, max_frame_length: int | None = None) -> None:
        self._max = _resolve_max_frame_length(max_frame_length)
        self._header = bytearray()
        self._expected: int | None = None
        self._payload = bytearray()
        self._ended = False
        self._failed = False

    def push(self, chunk: bytes) -> list[bytes]:
        if self._ended:
            raise FrameError("Frame decoder has ended")
        if self._failed:
            raise FrameError("Frame decoder has failed")
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError("Frame chunk must be bytes")

        frames: list[bytes] = []
        buf = memoryview(chunk)
        offset = 0
        while offset < len(buf):
            if self._expected is None:
                # Accumulate header bytes until we have all 4.
                need = FRAME_HEADER_LENGTH - len(self._header)
                take = min(need, len(buf) - offset)
                self._header += buf[offset : offset + take]
                offset += take
                if len(self._header) < FRAME_HEADER_LENGTH:
                    break
                length = int.from_bytes(self._header, "big")
                self._header.clear()
                if length > self._max:
                    self._fail(
                        f"Frame length {length} exceeds configured limit of {self._max}"
                    )
                self._expected = length
                self._payload = bytearray()
                if length == 0:
                    frames.append(b"")
                    self._expected = None
                    continue

            assert self._expected is not None
            need = self._expected - len(self._payload)
            take = min(need, len(buf) - offset)
            self._payload += buf[offset : offset + take]
            offset += take
            if len(self._payload) == self._expected:
                frames.append(bytes(self._payload))
                self._payload = bytearray()
                self._expected = None
        return frames

    def end(self) -> None:
        if self._ended:
            raise FrameError("Frame decoder has ended")
        if self._failed:
            raise FrameError("Frame decoder has failed")
        if self._header or self._expected is not None or self._payload:
            self._fail("Truncated frame at end of stream")
        self._ended = True

    def _fail(self, message: str) -> None:
        self._failed = True
        self._header.clear()
        self._payload = bytearray()
        self._expected = None
        raise FrameError(message)


# ---------------------------------------------------------------------------
# Payload codec (JSON; CBOR is a follow-up that swaps only these two)
# ---------------------------------------------------------------------------


def _encode_payload(model: Any) -> bytes:
    return json.dumps(model, separators=(",", ":")).encode("utf-8")


def _decode_payload(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"))


# ---------------------------------------------------------------------------
# Message codec
# ---------------------------------------------------------------------------

_client_ta: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
_server_ta: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def _validate_client(value: Any) -> ClientMessage:
    try:
        return _client_ta.validate_python(value)
    except ValidationError as exc:
        raise ProtocolValidationError("Invalid client protocol message") from exc


def _validate_server(value: Any) -> ServerMessage:
    try:
        return _server_ta.validate_python(value)
    except ValidationError as exc:
        raise ProtocolValidationError("Invalid server protocol message") from exc


def encode_client_message(
    message: Any, *, max_frame_length: int | None = None
) -> bytes:
    """Validate *message* as a client message and return a framed frame."""
    validated = _validate_client(message)
    payload = _encode_payload(validated.model_dump(mode="json"))
    return encode_frame(payload, max_frame_length=max_frame_length)


def encode_server_message(
    message: Any, *, max_frame_length: int | None = None
) -> bytes:
    """Validate *message* as a server message and return a framed frame."""
    validated = _validate_server(message)
    payload = _encode_payload(validated.model_dump(mode="json"))
    return encode_frame(payload, max_frame_length=max_frame_length)


class _MessageDecoder:
    """Frames + parses + schema-validates inbound messages of one kind."""

    def __init__(
        self,
        kind: str,
        validate: Any,
        *,
        max_frame_length: int | None = None,
    ) -> None:
        self._frames = FrameDecoder(max_frame_length=max_frame_length)
        self._kind = kind
        self._validate = validate
        self._failed = False

    def push(self, chunk: bytes) -> list[Any]:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            messages: list[Any] = []
            for payload in self._frames.push(chunk):
                messages.append(self._validate(_decode_payload(payload)))
            return messages
        except (FrameError, ProtocolValidationError):
            self._failed = True
            raise
        except Exception as exc:  # JSON decode errors, etc.
            self._failed = True
            raise ProtocolValidationError(
                f"Invalid {self._kind} protocol frame: {str(exc)[:500]}"
            ) from exc

    def end(self) -> None:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            self._frames.end()
        except FrameError as exc:
            self._failed = True
            raise ProtocolValidationError(
                f"Invalid {self._kind} protocol framing: {exc}"
            ) from exc


class ClientMessageDecoder(_MessageDecoder):
    """Incrementally decode + validate framed client messages."""

    def __init__(self, *, max_frame_length: int | None = None) -> None:
        super().__init__("client", _validate_client, max_frame_length=max_frame_length)


class ServerMessageDecoder(_MessageDecoder):
    """Incrementally decode + validate framed server messages."""

    def __init__(self, *, max_frame_length: int | None = None) -> None:
        super().__init__("server", _validate_server, max_frame_length=max_frame_length)


__all__ = [
    "FRAME_HEADER_LENGTH",
    "DEFAULT_MAX_FRAME_LENGTH",
    "FrameError",
    "ProtocolValidationError",
    "encode_frame",
    "FrameDecoder",
    "encode_client_message",
    "encode_server_message",
    "ClientMessageDecoder",
    "ServerMessageDecoder",
]
