"""Tests for ``cothis.protocol.wire`` — framing + JSON message codec.

Hermetic: no sockets, no network. Exercises the frame decoder's incremental
reassembly, the max-frame guard, and the client/server message round-trip.
"""

from __future__ import annotations

import pytest

from cothis.protocol.messages import PROTOCOL_VERSION
from cothis.protocol.wire import (
    DEFAULT_MAX_FRAME_LENGTH,
    ClientMessageDecoder,
    FrameDecoder,
    FrameError,
    ProtocolValidationError,
    ServerMessageDecoder,
    encode_client_message,
    encode_frame,
    encode_server_message,
)

# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def test_encode_frame_prepends_big_endian_length() -> None:
    frame = encode_frame(b"hello")
    # 4-byte BE length prefix (5) + payload.
    assert frame[:4] == (5).to_bytes(4, "big")
    assert frame[4:] == b"hello"


def test_frame_decoder_reassembles_split_chunks() -> None:
    payload = b"x" * 10
    frame = encode_frame(payload)
    dec = FrameDecoder()
    assert dec.push(frame[:2]) == []  # partial header
    assert dec.push(frame[2:5]) == []  # header done, partial payload
    assert dec.push(frame[5:]) == [payload]  # remainder completes it


def test_frame_decoder_yields_multiple_frames_per_chunk() -> None:
    two = encode_frame(b"a") + encode_frame(b"bb")
    assert FrameDecoder().push(two) == [b"a", b"bb"]


def test_frame_decoder_empty_payload_round_trips() -> None:
    assert FrameDecoder().push(encode_frame(b"")) == [b""]


def test_encode_frame_rejects_oversize_payload() -> None:
    with pytest.raises(FrameError):
        encode_frame(b"x" * 8, max_frame_length=4)


def test_frame_decoder_rejects_oversize_length_header() -> None:
    # Hand-craft a frame whose length header exceeds the configured limit.
    bad = (999).to_bytes(4, "big") + b""
    with pytest.raises(FrameError):
        FrameDecoder(max_frame_length=16).push(bad)


def test_frame_decoder_end_detects_truncation() -> None:
    dec = FrameDecoder()
    dec.push(encode_frame(b"ok"))
    # Feed a partial frame then end -> truncated.
    dec.push((5).to_bytes(4, "big"))
    with pytest.raises(FrameError):
        dec.end()


def test_default_max_frame_length_constant() -> None:
    assert DEFAULT_MAX_FRAME_LENGTH == 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Message codec
# ---------------------------------------------------------------------------


def test_client_message_round_trip_hello() -> None:
    frame = encode_client_message(
        {"type": "hello", "version": PROTOCOL_VERSION, "token": "t"}
    )
    [msg] = ClientMessageDecoder().push(frame)
    assert msg.type == "hello" and msg.token == "t"


def test_client_message_round_trip_request() -> None:
    frame = encode_client_message(
        {
            "type": "request",
            "id": "r1",
            "request": {"command": "prompt", "sessionId": "s", "text": "hi"},
        }
    )
    [msg] = ClientMessageDecoder().push(frame)
    assert msg.type == "request"
    assert msg.request.command == "prompt" and msg.request.text == "hi"


def test_server_message_round_trip_hello_with_snapshot() -> None:
    snapshot = {
        "serverId": "srv",
        "protocolVersion": PROTOCOL_VERSION,
        "revision": 0,
        "sessions": [],
        "models": [],
    }
    frame = encode_server_message(
        {"type": "hello", "version": PROTOCOL_VERSION, "connectionId": "c1", "snapshot": snapshot}
    )
    [msg] = ServerMessageDecoder().push(frame)
    assert msg.type == "hello" and msg.connectionId == "c1"
    assert msg.snapshot.serverId == "srv"


def test_server_message_round_trip_event() -> None:
    frame = encode_server_message(
        {
            "type": "event",
            "event": {"type": "session_removed", "sessionId": "s"},
        }
    )
    [msg] = ServerMessageDecoder().push(frame)
    assert msg.type == "event" and msg.event.type == "session_removed"


def test_invalid_payload_raises_protocol_validation_error() -> None:
    bad = encode_frame(b'{"not":"a valid message"}')
    with pytest.raises(ProtocolValidationError):
        ClientMessageDecoder().push(bad)


def test_decoder_is_poisoned_after_failure() -> None:
    dec = ClientMessageDecoder()
    bad = encode_frame(b'{"not":"valid"}')
    with pytest.raises(ProtocolValidationError):
        dec.push(bad)
    # A second push on a failed decoder raises immediately.
    with pytest.raises(ProtocolValidationError):
        dec.push(encode_frame(b'{"type":"hello","version":1,"token":"t"}'))


def test_client_decoder_handles_split_frame() -> None:
    frame = encode_client_message(
        {"type": "hello", "version": PROTOCOL_VERSION, "token": "t"}
    )
    dec = ClientMessageDecoder()
    assert dec.push(frame[:3]) == []
    [msg] = dec.push(frame[3:])
    assert msg.token == "t"
