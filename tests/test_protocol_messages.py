"""Tests for the ``cothis.protocol`` ACP-ready type seed (clean-room).

Pure model validation — no sockets, no network.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from cothis.protocol import (
    PROTOCOL_VERSION,
    ClientHello,
    ClientMessage,
    EventEnvelope,
    ProtocolError,
    RequestEnvelope,
    ResponseEnvelope,
    ServerHello,
    ServerMessage,
    is_supported_protocol_version,
)


def test_protocol_version_supported() -> None:
    assert is_supported_protocol_version(PROTOCOL_VERSION)
    assert not is_supported_protocol_version(PROTOCOL_VERSION + 1)


def test_client_hello_roundtrip_and_validation() -> None:
    msg = ClientHello.model_validate(
        {"type": "hello", "version": PROTOCOL_VERSION, "token": "abc"}
    )
    assert msg.token == "abc"
    assert ClientHello.model_validate(msg.model_dump()).token == "abc"
    with pytest.raises(ValidationError):  # negative version
        ClientHello.model_validate({"type": "hello", "version": -1, "token": "abc"})
    with pytest.raises(ValidationError):  # empty token
        ClientHello.model_validate({"type": "hello", "version": 1, "token": ""})
    with pytest.raises(ValidationError):  # extra field
        ClientHello.model_validate(
            {"type": "hello", "version": 1, "token": "abc", "extra": 1}
        )


def test_server_hello_version_is_pinned() -> None:
    ServerHello.model_validate(
        {"type": "hello", "version": PROTOCOL_VERSION, "connection_id": "c1"}
    )
    with pytest.raises(ValidationError):
        ServerHello.model_validate(
            {"type": "hello", "version": PROTOCOL_VERSION + 1, "connection_id": "c1"}
        )


def test_response_envelope_ok_and_error_variants() -> None:
    ok = ResponseEnvelope.model_validate(
        {"type": "response", "id": "r1", "result": {"x": 1}}
    )
    assert ok.result == {"x": 1}
    err = ResponseEnvelope.model_validate(
        {"type": "response", "id": "r1", "error": {"code": "auth", "message": "no"}}
    )
    assert err.error is not None and err.error.code == "auth"
    with pytest.raises(ValidationError):  # neither set
        ResponseEnvelope.model_validate({"type": "response", "id": "r1"})
    with pytest.raises(ValidationError):  # both set
        ResponseEnvelope.model_validate(
            {
                "type": "response",
                "id": "r1",
                "result": 1,
                "error": {"code": "auth", "message": "x"},
            }
        )


def test_protocol_error_code_literal() -> None:
    ProtocolError.model_validate({"code": "busy", "message": "occupied"})
    with pytest.raises(ValidationError):  # unknown code
        ProtocolError.model_validate({"code": "bogus", "message": "x"})


def test_client_message_discriminated_dispatch() -> None:
    ta = TypeAdapter(ClientMessage)
    assert isinstance(
        ta.validate_python({"type": "hello", "version": 1, "token": "t"}), ClientHello
    )
    assert isinstance(
        ta.validate_python({"type": "request", "id": "r", "request": {}}), RequestEnvelope
    )


def test_server_message_discriminated_dispatch() -> None:
    ta = TypeAdapter(ServerMessage)
    assert isinstance(
        ta.validate_python(
            {"type": "hello", "version": PROTOCOL_VERSION, "connection_id": "c"}
        ),
        ServerHello,
    )
    assert isinstance(
        ta.validate_python({"type": "response", "id": "r", "result": 1}),
        ResponseEnvelope,
    )
    assert isinstance(
        ta.validate_python({"type": "event", "event": {"k": 1}}),
        EventEnvelope,
    )
