"""Tests for the ``cothis.protocol`` ACP message model.

Pure model validation — no sockets, no network. Covers the handshake
envelopes, the Command / CommandResult / ServerEvent discriminated unions,
and the transcript/progress shapes.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from cothis.protocol import (
    PROTOCOL_VERSION,
    BackendError,
    ClientHello,
    ClientMessage,
    Command,
    CommandResult,
    EventEnvelope,
    PromptCommand,
    ProtocolError,
    RequestEnvelope,
    ResponseEnvelope,
    ServerEvent,
    ServerHello,
    ServerMessage,
    is_supported_protocol_version,
)


def _snapshot() -> dict:
    return {
        "serverId": "srv",
        "protocolVersion": PROTOCOL_VERSION,
        "revision": 0,
        "sessions": [],
        "models": [],
    }


# ---------------------------------------------------------------------------
# Version + handshake
# ---------------------------------------------------------------------------


def test_protocol_version_supported() -> None:
    assert is_supported_protocol_version(PROTOCOL_VERSION)
    assert not is_supported_protocol_version(PROTOCOL_VERSION + 1)


def test_client_hello_roundtrip_and_validation() -> None:
    msg = ClientHello.model_validate(
        {"type": "hello", "version": PROTOCOL_VERSION, "token": "abc"}
    )
    assert msg.token == "abc"
    with pytest.raises(ValidationError):  # negative version
        ClientHello.model_validate({"type": "hello", "version": -1, "token": "abc"})
    with pytest.raises(ValidationError):  # empty token
        ClientHello.model_validate({"type": "hello", "version": 1, "token": ""})
    with pytest.raises(ValidationError):  # extra field
        ClientHello.model_validate(
            {"type": "hello", "version": 1, "token": "abc", "extra": 1}
        )


def test_server_hello_requires_snapshot_and_pins_version() -> None:
    ServerHello.model_validate(
        {
            "type": "hello",
            "version": PROTOCOL_VERSION,
            "connectionId": "c1",
            "snapshot": _snapshot(),
        }
    )
    with pytest.raises(ValidationError):  # wrong version
        ServerHello.model_validate(
            {
                "type": "hello",
                "version": PROTOCOL_VERSION + 1,
                "connectionId": "c1",
                "snapshot": _snapshot(),
            }
        )
    with pytest.raises(ValidationError):  # missing snapshot
        ServerHello.model_validate(
            {"type": "hello", "version": PROTOCOL_VERSION, "connectionId": "c1"}
        )


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------


def test_response_envelope_ok_consistency() -> None:
    ok = ResponseEnvelope.model_validate(
        {"type": "response", "id": "r1", "ok": True, "result": {"command": "list", "sessions": []}}
    )
    assert ok.ok and ok.result is not None
    err = ResponseEnvelope.model_validate(
        {"type": "response", "id": "r1", "ok": False, "error": {"code": "auth", "message": "no"}}
    )
    assert not err.ok and err.error is not None and err.error.code == "auth"
    with pytest.raises(ValidationError):  # ok True but no result
        ResponseEnvelope.model_validate({"type": "response", "id": "r1", "ok": True})
    with pytest.raises(ValidationError):  # ok True but also error
        ResponseEnvelope.model_validate(
            {
                "type": "response", "id": "r1", "ok": True,
                "result": {"command": "list", "sessions": []},
                "error": {"code": "auth", "message": "x"},
            }
        )
    with pytest.raises(ValidationError):  # ok False but no error
        ResponseEnvelope.model_validate({"type": "response", "id": "r1", "ok": False})


def test_protocol_error_code_literal_and_backend_error() -> None:
    ProtocolError.model_validate({"code": "busy", "message": "occupied"})
    with pytest.raises(ValidationError):  # unknown code
        ProtocolError.model_validate({"code": "bogus", "message": "x"})
    # BackendError wraps a ProtocolError and is a real Exception.
    be = BackendError(ProtocolError(code="not_found", message="missing"))
    assert isinstance(be, Exception)
    assert be.error.code == "not_found"


# ---------------------------------------------------------------------------
# Discriminated unions
# ---------------------------------------------------------------------------


def test_request_envelope_narrows_to_command() -> None:
    env = RequestEnvelope.model_validate(
        {
            "type": "request",
            "id": "r1",
            "request": {"command": "prompt", "sessionId": "s", "text": "hi"},
        }
    )
    assert isinstance(env.request, PromptCommand)
    assert env.request.text == "hi"
    with pytest.raises(ValidationError):  # unknown command
        RequestEnvelope.model_validate(
            {"type": "request", "id": "r1", "request": {"command": "bogus"}}
        )


def test_command_union_dispatch() -> None:
    ta = TypeAdapter(Command)
    listed = ta.validate_python({"command": "list"})
    assert listed.command == "list"
    # Prompt carries its payload.
    cmd = ta.validate_python({"command": "prompt", "sessionId": "s", "text": "x"})
    assert cmd.command == "prompt" and cmd.sessionId == "s"  # type: ignore[attr-defined]


def test_command_result_union_dispatch() -> None:
    ta = TypeAdapter(CommandResult)
    res = ta.validate_python({"command": "list", "sessions": []})
    assert res.command == "list" and res.sessions == []  # type: ignore[attr-defined]


def test_server_event_union_dispatch() -> None:
    ta = TypeAdapter(ServerEvent)
    ev = ta.validate_python({"type": "session_removed", "sessionId": "s"})
    assert ev.type == "session_removed" and ev.sessionId == "s"
    progress = ta.validate_python(
        {
            "type": "session_progress",
            "sessionId": "s",
            "progress": {"type": "item_started", "item": {"role": "user", "id": "u", "content": [{"type": "text", "text": "hi"}], "timestamp": 0}},
        }
    )
    assert progress.progress.type == "item_started"


def test_client_message_discriminated_dispatch() -> None:
    ta = TypeAdapter(ClientMessage)
    assert isinstance(
        ta.validate_python({"type": "hello", "version": 1, "token": "t"}), ClientHello
    )
    req = ta.validate_python(
        {"type": "request", "id": "r", "request": {"command": "list"}}
    )
    assert isinstance(req, RequestEnvelope)


def test_server_message_discriminated_dispatch() -> None:
    ta = TypeAdapter(ServerMessage)
    assert isinstance(
        ta.validate_python(
            {"type": "hello", "version": PROTOCOL_VERSION, "connectionId": "c", "snapshot": _snapshot()}
        ),
        ServerHello,
    )
    assert isinstance(
        ta.validate_python(
            {"type": "response", "id": "r", "ok": True, "result": {"command": "list", "sessions": []}}
        ),
        ResponseEnvelope,
    )
    assert isinstance(
        ta.validate_python(
            {"type": "event", "event": {"type": "session_removed", "sessionId": "s"}}
        ),
        EventEnvelope,
    )


def test_event_envelope_requires_valid_server_event() -> None:
    EventEnvelope.model_validate(
        {"type": "event", "event": {"type": "session_removed", "sessionId": "s"}}
    )
    with pytest.raises(ValidationError):  # malformed event payload
        EventEnvelope.model_validate({"type": "event", "event": {"k": 1}})
