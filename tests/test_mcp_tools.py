"""Tests for the MCP stdio adapter (issue #16).

Every test that needs a live session drives a real in-memory MCP server via
the SDK's ``create_connected_server_and_client_session`` transport — no
subprocess, no network, deterministic. The server is a ``FastMCP`` with a
handful of tools; ``MCPServer``'s ``connect`` seam swaps the production
stdio transport for this in-memory one, so the adapter code under test is
exactly what production runs (only the transport differs).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from mcp import StdioServerParameters
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from cothis.agent import Agent
from cothis.tools import (
    MCPClientTool,
    MCPServer,
    load_tools_from_layer,
    load_yaml_tools,
    tool,
)
from cothis.tools.core import _HookableTool, _ShellTool
from cothis.tools.mcp import (
    _build_mcp_http_server,
    _build_mcp_stdio_server,
    _flatten_exc,
    _normalize_mcp_result,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


# --- fixtures / helpers ------------------------------------------------


def _make_server() -> FastMCP:
    """A minimal in-memory MCP server with an ``add`` and a ``boom`` tool."""
    server = FastMCP("test-server")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    def boom() -> str:
        """Always raises."""
        raise ValueError("kaboom")

    return server


def _in_memory(server: FastMCP) -> Callable[[], Any]:
    """A ``connect`` seam that yields an in-memory session bound to ``server``."""
    return lambda: create_connected_server_and_client_session(server)


def _mcp_server(server: FastMCP | None = None) -> MCPServer:
    # ``mcp:`` prefix matches production handles (``_make_mcp_server``);
    # ``start()`` strips it to get the label ``test``, so produced tools are
    # ``test.add`` / ``test.boom`` (ADR-0006 prefix).
    return MCPServer(name="mcp:test", connect=_in_memory(server or _make_server()))


class _FailingCM:
    """A connection context whose ``__aenter__`` always fails.

    Simulates a server that can't launch (bad command, connection refused) so
    the load-failure path (story 30) is exercised without a real subprocess.
    """

    async def __aenter__(self) -> Any:
        raise RuntimeError("connect refused")

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FailingGroupCM:
    """A connection context whose ``__aenter__`` raises an ``ExceptionGroup``,
    mirroring how anyio's TaskGroup surfaces an http/transport failure."""

    async def __aenter__(self) -> Any:
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup (1 sub-exception)",
            [ConnectionError("Name or service not known")],
        )

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeSession:
    """A ``ClientSession`` stand-in that lists zero tools.

    Lets a transport-delivery test drive ``_default_connect`` end-to-end
    (open transport → wrap session → ``list_tools``) while isolating the
    transport-argument capture from the real MCP protocol.
    """

    def __init__(self, read: Any, write: Any) -> None:
        self._read, self._write = read, write

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=[])


def _patch_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``mcp.ClientSession`` (imported lazily inside ``_default_connect``)
    with ``_FakeSession`` so no real session/protocol is exercised."""
    import mcp

    monkeypatch.setattr(mcp, "ClientSession", _FakeSession, raising=False)


def _mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``AnyLLM.create`` so ``Agent(...)`` needs no provider/network."""
    import any_llm

    monkeypatch.setattr(
        any_llm.AnyLLM, "create", staticmethod(lambda *a, **kw: MagicMock())
    )


# --- YAML routing (discovery) ------------------------------------------


def test_yaml_type_mcp_stdio_routes_to_server() -> None:
    """``type: mcp.stdio`` produces an ``MCPServer``, not a shell tool."""
    yaml_text = (
        "type: mcp.stdio\nname: browser\ncommand: uvx\nargs: [browser-use, --mcp]\n"
    )
    tools = load_yaml_tools(yaml_text, source="browser.yaml")
    assert len(tools) == 1
    server = tools[0]
    assert isinstance(server, MCPServer)
    assert not isinstance(server, _ShellTool)
    assert server.__name__ == "mcp:browser"
    # ``command``/``args`` are parsed into the safe (secret-free) diagnostic.
    assert "uvx" in server._diagnostic
    assert "browser-use" in server._diagnostic


def test_yaml_mcp_env_absent_from_diagnostic() -> None:
    """``env:`` is parsed but kept out of the loggable diagnostic (story 32)."""
    yaml_text = "type: mcp.stdio\ncommand: srv\nenv:\n  API_KEY: s3cr3t\n"
    server = load_yaml_tools(yaml_text)[0]
    assert isinstance(server, MCPServer)
    assert "s3cr3t" not in server._diagnostic
    assert "API_KEY" not in server._diagnostic


def test_yaml_mcp_name_defaults_to_file_stem() -> None:
    tools = load_yaml_tools(
        "type: mcp.stdio\ncommand: foo\n", source="/x/myserver.yaml"
    )
    assert tools[0].__name__ == "mcp:myserver"


def test_yaml_mcp_handle_name_prefixed_to_avoid_tool_collision() -> None:
    """A server named like a real tool gets an ``mcp:`` prefix, so it can't
    shadow that tool in the discovery registry."""
    server = load_yaml_tools("type: mcp.stdio\nname: fs.read\ncommand: foo\n")[0]
    assert server.__name__ == "mcp:fs.read"


def test_yaml_mcp_unknown_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        load_yaml_tools("type: mcp.stdio\ncommand: foo\nbogus: 1\n")


def test_yaml_mcp_missing_command_rejected() -> None:
    with pytest.raises(ValueError, match="must define 'command'") as exc_info:
        load_yaml_tools("type: mcp.stdio\nname: x\n", source="srv.yaml")
    assert "srv.yaml" in str(exc_info.value)


def test_yaml_mcp_args_must_be_list() -> None:
    with pytest.raises(ValueError, match="'args' must be a list"):
        load_yaml_tools("type: mcp.stdio\ncommand: foo\nargs: nope\n")


def test_yaml_mcp_env_non_string_value_rejected() -> None:
    """A non-string ``env`` value is rejected with file + field + type (AC #9,
    story 30) — not silently coerced to str."""
    with pytest.raises(ValueError, match="'env.API_KEY' must be a string") as exc_info:
        load_yaml_tools(
            "type: mcp.stdio\ncommand: foo\nenv:\n  API_KEY: 123\n",
            source="srv.yaml",
        )
    msg = str(exc_info.value)
    assert "srv.yaml" in msg
    assert "int" in msg


def test_yaml_mcp_stdio_warns_when_command_not_on_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stdio server whose ``command`` is not on PATH logs a WARNING (story 30)."""
    with caplog.at_level(logging.WARNING, logger="cothis.tools"):
        server = load_yaml_tools(
            "type: mcp.stdio\nname: ghost\ncommand: definitely-not-on-path-xyz\n"
        )[0]
    assert isinstance(server, MCPServer)
    assert "definitely-not-on-path-xyz" in caplog.text
    assert "not on PATH" in caplog.text


def test_yaml_mcp_stdio_no_warning_when_command_on_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stdio server whose ``command`` IS on PATH emits no PATH warning."""
    import sys

    exe = sys.executable  # always resolvable by shutil.which
    with caplog.at_level(logging.WARNING, logger="cothis.tools"):
        load_yaml_tools(f"type: mcp.stdio\nname: ok\ncommand: {exe}\n")
    assert "not on PATH" not in caplog.text


@pytest.mark.parametrize(
    "yaml_text",
    [
        "type: mcp.stdio\nname: a:b\ncommand: foo\n",
        "type: mcp.http\nname: a:b\nurl: https://x/mcp\n",
    ],
    ids=["stdio", "http"],
)
def test_yaml_mcp_label_with_colon_rejected(yaml_text: str) -> None:
    """A label containing ``:`` would break the ``:``-is-not-valid-in-tool-names
    invariant — refused at build time for both transports."""
    with pytest.raises(ValueError, match="contains ':'"):
        load_yaml_tools(yaml_text, source="bad.yaml")


# --- YAML routing: http transport --------------------------------------


def test_yaml_type_mcp_http_routes_to_server() -> None:
    """``type: mcp.http`` produces an ``MCPServer`` (reusing the stdio path),
    not a shell tool."""
    yaml_text = "type: mcp.http\nname: remote\nurl: https://example.com/mcp\n"
    tools = load_yaml_tools(yaml_text, source="remote.yaml")
    assert len(tools) == 1
    server = tools[0]
    assert isinstance(server, MCPServer)
    assert not isinstance(server, _ShellTool)
    assert server.__name__ == "mcp:remote"
    # ``url`` is parsed into the safe diagnostic; ``headers`` never are.
    assert "https://example.com/mcp" in server._diagnostic


def test_yaml_mcp_http_name_defaults_to_file_stem() -> None:
    tools = load_yaml_tools(
        "type: mcp.http\nurl: https://x/mcp\n", source="/x/remote.yaml"
    )
    assert tools[0].__name__ == "mcp:remote"


def test_yaml_mcp_http_headers_absent_from_diagnostic() -> None:
    """``headers:`` secrets (e.g. Authorization) never reach the diagnostic."""
    yaml_text = (
        "type: mcp.http\nurl: https://x/mcp\nheaders:\n  Authorization: Bearer s3cr3t\n"
    )
    server = load_yaml_tools(yaml_text)[0]
    assert isinstance(server, MCPServer)
    assert "s3cr3t" not in server._diagnostic
    assert "Authorization" not in server._diagnostic


def test_yaml_mcp_http_missing_url_rejected() -> None:
    with pytest.raises(ValueError, match="must define 'url'") as exc_info:
        load_yaml_tools("type: mcp.http\nname: x\n", source="remote.yaml")
    assert "remote.yaml" in str(exc_info.value)


def test_yaml_mcp_http_unknown_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        load_yaml_tools("type: mcp.http\nurl: https://x/mcp\nbogus: 1\n")


def test_http_headers_must_be_mapping() -> None:
    with pytest.raises(ValueError, match="'headers' must be a mapping"):
        load_yaml_tools("type: mcp.http\nurl: https://x/mcp\nheaders: nope\n")


def test_http_url_scrubbed_in_diagnostic() -> None:
    """Userinfo and query-string secrets never reach the loggable diagnostic
    (story 32 — the diagnostic is the only url-derived string that's logged)."""
    yaml_text = (
        "type: mcp.http\n"
        "url: 'https://token:hunter2@host.example.com/mcp?api_key=leak'\n"
    )
    server = load_yaml_tools(yaml_text)[0]
    assert isinstance(server, MCPServer)
    # Userinfo and query are stripped — both can carry secrets.
    assert "token" not in server._diagnostic
    assert "hunter2" not in server._diagnostic
    assert "leak" not in server._diagnostic
    assert "api_key" not in server._diagnostic
    # …but the host + path survive (still useful to debug a bad endpoint).
    assert "host.example.com" in server._diagnostic
    assert "/mcp" in server._diagnostic


def test_http_url_ipv6_brackets_preserved_in_diagnostic() -> None:
    """IPv6 endpoints keep their brackets after scrubbing — rebuilding netloc
    from ``hostname`` would drop them and emit a malformed url in the log."""
    server = load_yaml_tools(
        "type: mcp.http\nurl: 'https://[::1]:8000/mcp?api_key=leak'\n"
    )[0]
    assert isinstance(server, MCPServer)
    assert "[::1]:8000" in server._diagnostic  # brackets + port preserved
    assert "leak" not in server._diagnostic
    assert "https://::1" not in server._diagnostic  # not the malformed form


def test_unknown_type_rejected_with_valid_options() -> None:
    """An unknown ``type:`` value names the file + the bad value + valid
    options (story 30), instead of falling through to shell-tool compile."""
    with pytest.raises(ValueError, match="unknown tool type") as exc_info:
        load_yaml_tools("type: bogus\nname: x\ncommand: echo\n", source="bad.yaml")
    msg = str(exc_info.value)
    assert "'bogus'" in msg
    assert "bad.yaml" in msg
    assert "mcp.stdio" in msg
    assert "mcp.http" in msg


def test_no_type_falls_through_to_shell_tool() -> None:
    """A declaration with no ``type:`` is a shell-template tool — unchanged
    backward-compatible behavior (#18 AC)."""
    tools = load_yaml_tools("name: hi\ncommand: [echo, hello]\n", source="ok.yaml")
    assert len(tools) == 1
    assert isinstance(tools[0], _ShellTool)
    assert tools[0].__name__ == "hi"


def test_mixed_directory_loads_all_three_types(tmp_path: Any) -> None:
    """A single discovery directory loads shell, mcp.stdio, and mcp.http tools
    side by side (#18 AC: mixed declarations)."""
    (tmp_path / "shell.yaml").write_text(
        'name: my.shell\ncommand: ["echo", "hi"]\n', encoding="utf-8"
    )
    (tmp_path / "stdio.yaml").write_text(
        "type: mcp.stdio\nname: local\ncommand: echo\n", encoding="utf-8"
    )
    (tmp_path / "http.yaml").write_text(
        "type: mcp.http\nname: remote\nurl: https://example.com/mcp\n",
        encoding="utf-8",
    )
    tools = load_tools_from_layer(tmp_path)
    by_name = {t.__name__: t for t in tools}
    # Shell tool is a dispatchable _ShellTool.
    assert "my.shell" in by_name
    assert isinstance(by_name["my.shell"], _ShellTool)
    # Both MCP kinds are MCPServer handles (mcp: prefixed, not yet connected).
    assert "mcp:local" in by_name
    assert isinstance(by_name["mcp:local"], MCPServer)
    assert "mcp:remote" in by_name
    assert isinstance(by_name["mcp:remote"], MCPServer)


# --- normalization (pure) ----------------------------------------------


def test_normalize_single_block() -> None:
    result = SimpleNamespace(content=[SimpleNamespace(text="7")], isError=False)
    assert _normalize_mcp_result(result) == "7"


def test_normalize_multiple_blocks_joined() -> None:
    result = SimpleNamespace(
        content=[SimpleNamespace(text="a"), SimpleNamespace(text="b")],
        isError=False,
    )
    assert _normalize_mcp_result(result) == "a\nb"


def test_normalize_empty_content() -> None:
    result = SimpleNamespace(content=[], isError=False)
    assert _normalize_mcp_result(result) == "(no output)"


def test_normalize_error_prefixed() -> None:
    result = SimpleNamespace(content=[SimpleNamespace(text="bad")], isError=True)
    assert _normalize_mcp_result(result) == "Error: bad"


def test_normalize_nontext_only_content_is_no_output() -> None:
    """A non-empty result carrying only non-text blocks collapses to
    ``(no output)`` for a text-only agent (documented ceiling in
    ``_normalize_mcp_result``: text-less == nothing-to-say)."""
    # A block with no ``.text`` attribute (e.g. an image block).
    result = SimpleNamespace(content=[SimpleNamespace(data="<bytes>")], isError=False)
    assert _normalize_mcp_result(result) == "(no output)"


# --- session lifecycle + dispatch --------------------------------------


@pytest.mark.asyncio
async def test_start_discovers_tools_with_schema() -> None:
    """``start`` lists remote tools, each wrapped with its ``inputSchema``, with
    a label-prefixed ``__name__`` and a bare ``_remote_name`` (ADR-0006)."""
    server = _mcp_server()
    tools = await server.start()
    try:
        by_name = {t.__name__: t for t in tools}
        assert "test.add" in by_name
        add = by_name["test.add"]
        assert isinstance(add, MCPClientTool)
        assert isinstance(add, _HookableTool)
        assert add.__name__ == "test.add"
        assert add._remote_name == "add"  # bare server-side name for call_tool
        params = add.__cothis_schema__["function"]["parameters"]
        assert add.__cothis_schema__["function"]["name"] == "test.add"
        assert "a" in params["properties"]
        assert "b" in params["properties"]
    finally:
        await server.aclose()


@pytest.mark.asyncio
async def test_mcp_tool_call_uses_bare_remote_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``call_tool`` receives the bare ``_remote_name``, not the prefixed name."""
    server = _mcp_server()
    tools = {t.__name__: t for t in await server.start()}
    add = tools["test.add"]
    captured: dict[str, Any] = {}
    orig_call = server._session.call_tool

    async def spy_call(name: str, args: Any) -> Any:
        captured["name"] = name
        return await orig_call(name, args)

    monkeypatch.setattr(server._session, "call_tool", spy_call)
    try:
        assert await add(a=2, b=3) == "5"
    finally:
        await server.aclose()
    assert captured["name"] == "add"  # bare, not "test.add"


@pytest.mark.asyncio
async def test_call_returns_normalized_string() -> None:
    server = _mcp_server()
    tools = {t.__name__: t for t in await server.start()}
    try:
        assert await tools["test.add"](a=3, b=4) == "7"
    finally:
        await server.aclose()


@pytest.mark.asyncio
async def test_call_error_prefixed() -> None:
    """A remote tool that raises comes back as an ``Error:`` string."""
    server = _mcp_server()
    tools = {t.__name__: t for t in await server.start()}
    try:
        result = await tools["test.boom"]()
        assert result.startswith("Error:")
        assert "kaboom" in result
    finally:
        await server.aclose()


@pytest.mark.asyncio
async def test_session_persistent_no_reconnect() -> None:
    """Multiple calls reuse one session — connect happens exactly once."""
    server_obj = _make_server()
    connect_calls: list[int] = []

    def connect() -> Any:
        connect_calls.append(1)
        return create_connected_server_and_client_session(server_obj)

    server = MCPServer(name="mcp:t", connect=connect)
    tools = {t.__name__: t for t in await server.start()}
    try:
        first_session = server._session
        assert await tools["t.add"](a=1, b=1) == "2"
        assert await tools["t.add"](a=2, b=2) == "4"
        assert server._session is first_session
        assert len(connect_calls) == 1
    finally:
        await server.aclose()


@pytest.mark.asyncio
async def test_aclose_clears_session_and_tools_fail_after() -> None:
    server = _mcp_server()
    tools = {t.__name__: t for t in await server.start()}
    await server.aclose()
    assert server._session is None
    assert server._cm is None
    with pytest.raises(RuntimeError, match="session is not active"):
        await tools["test.add"](a=1, b=1)


@pytest.mark.asyncio
async def test_aclose_safe_if_never_started() -> None:
    """Closing a server that never started is a no-op, not an error."""
    server = _mcp_server()
    await server.aclose()
    assert server._session is None


@pytest.mark.asyncio
async def test_stdio_transport_delivers_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production stdio transport (built by ``_build_mcp_stdio_server``, no
    ``connect`` seam) hands the declared ``env`` straight to ``stdio_client``
    — so a server's secrets reach its subprocess (story 32). Patches the lazy
    transport imports so nothing is actually spawned."""
    import mcp.client.stdio as mcp_stdio

    captured: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_stdio_client(params: Any) -> AsyncIterator[tuple[str, str]]:
        captured["params"] = params
        yield ("read", "write")

    monkeypatch.setattr(mcp_stdio, "stdio_client", fake_stdio_client, raising=False)
    _patch_session(monkeypatch)

    server = _build_mcp_stdio_server(
        {
            "type": "mcp.stdio",
            "name": "t",
            "command": "srv",
            "args": ["--x"],
            "env": {"API_KEY": "s3cr3t"},
        },
        source=None,
    )
    try:
        assert await server.start() == []
    finally:
        await server.aclose()
    assert captured["params"].env == {"API_KEY": "s3cr3t"}
    assert captured["params"].command == "srv"
    assert captured["params"].args == ["--x"]


@pytest.mark.asyncio
async def test_http_transport_delivers_url_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production http transport (built by ``_build_mcp_http_server``) hands
    ``url`` + ``headers`` straight to ``streamablehttp_client`` — so auth
    secrets reach the remote (story 32). Patches the lazy transport import so
    no network call is made."""
    import mcp.client.streamable_http as mcp_http

    captured: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_streamablehttp_client(
        url: str, headers: Any = None
    ) -> AsyncIterator[tuple[str, str, None]]:
        captured["url"] = url
        captured["headers"] = headers
        yield ("read", "write", None)

    monkeypatch.setattr(
        mcp_http, "streamablehttp_client", fake_streamablehttp_client, raising=False
    )
    _patch_session(monkeypatch)

    server = _build_mcp_http_server(
        {
            "type": "mcp.http",
            "name": "t",
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer s3cr3t"},
        },
        source=None,
    )
    try:
        assert await server.start() == []
    finally:
        await server.aclose()
    assert captured["url"] == "https://example.com/mcp"
    assert captured["headers"] == {"Authorization": "Bearer s3cr3t"}


def test_streamablehttp_client_signature_unchanged() -> None:
    """Pin the SDK's real ``streamablehttp_client`` call shape.

    The http transport calls it as ``(url, headers=...)`` and unpacks its
    yield as a 3-tuple ``(read, write, get_session_id)``. An SDK rename or
    arity change would otherwise surface only at runtime against a live
    server; this smoke test fails fast on import/upgrade. (The transport
    test above mocks the function, so it asserts cothis's call shape, not
    the SDK's — this one guards the SDK side.)"""
    import inspect

    from mcp.client.streamable_http import streamablehttp_client

    sig = inspect.signature(streamablehttp_client)
    params = list(sig.parameters)
    assert params[0] == "url", f"SDK renamed first param: {params}"
    assert "headers" in params, f"SDK dropped headers param: {params}"
    # ``headers`` must default to None — cothis passes ``headers or None``.
    assert sig.parameters["headers"].default is None


@pytest.mark.asyncio
async def test_start_failure_logs_warning_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A launch failure logs at WARNING (command/args, never env) and yields [].

    Built via the real builder, so the assertion that ``env`` is absent proves
    the *builder* keeps secrets out of the diagnostic (story 32)."""
    server = _build_mcp_stdio_server(
        {
            "type": "mcp.stdio",
            "name": "broken",
            "command": "badcmd",
            "args": ["--flag"],
            "env": {"SECRET_KEY": "topsecret"},
        },
        source=None,
    )
    server._connect = lambda: _FailingCM()  # force a launch failure
    with caplog.at_level(logging.WARNING, logger="cothis.tools"):
        tools = await server.start()
    assert tools == []
    assert server._session is None
    assert "broken" in caplog.text
    assert "connect refused" in caplog.text
    # Safe diagnostics are present…
    assert "badcmd" in caplog.text
    assert "--flag" in caplog.text
    # …but env secrets are never logged (story 32).
    assert "topsecret" not in caplog.text
    assert "SECRET_KEY" not in caplog.text


@pytest.mark.asyncio
async def test_http_start_failure_logs_url_never_headers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An http connection failure logs at WARNING naming the url — never the
    ``headers`` (Authorization secret, story 32) — and yields []."""
    server = _build_mcp_http_server(
        {
            "type": "mcp.http",
            "name": "broken",
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer topsecret"},
        },
        source=None,
    )
    server._connect = lambda: _FailingCM()  # force a connection failure
    with caplog.at_level(logging.WARNING, logger="cothis.tools"):
        tools = await server.start()
    assert tools == []
    assert "connect refused" in caplog.text
    # Safe diagnostic (url) is present…
    assert "https://example.com/mcp" in caplog.text
    # …but header secrets are never logged (story 32).
    assert "topsecret" not in caplog.text
    assert "Authorization" not in caplog.text


def test_flatten_exc_plain() -> None:
    assert _flatten_exc(ValueError("boom")) == "ValueError: boom"


def test_flatten_exc_unwraps_exception_group() -> None:
    """A TaskGroup ExceptionGroup is unwrapped to its leaf cause — not the
    opaque 'unhandled errors in a TaskGroup' wrapper (actionable errors)."""
    group = ExceptionGroup(
        "unhandled errors in a TaskGroup (1 sub-exception)",
        [ConnectionError("Name or service not known")],
    )
    msg = _flatten_exc(group)
    assert "ConnectionError" in msg
    assert "Name or service not known" in msg
    assert "TaskGroup" not in msg


@pytest.mark.asyncio
async def test_start_failure_unwraps_taskgroup_in_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the transport fails inside an anyio TaskGroup, the WARNING names
    the real cause, not 'unhandled errors in a TaskGroup'."""
    server = _build_mcp_http_server(
        {"type": "mcp.http", "name": "remote", "url": "https://example.com/mcp"},
        source=None,
    )
    server._connect = lambda: _FailingGroupCM()
    with caplog.at_level(logging.WARNING, logger="cothis.tools"):
        assert await server.start() == []
    assert "Name or service not known" in caplog.text
    assert "unhandled errors in a TaskGroup" not in caplog.text


# --- hooks (inheritance from _HookableTool) ----------------------------


@pytest.mark.asyncio
async def test_pre_and_after_execute_hooks_run() -> None:
    """pre_execute/after_execute pipelines wrap the async MCP call."""
    server = _mcp_server()
    tools = {t.__name__: t for t in await server.start()}
    add = tools["test.add"]
    seen: dict[str, bool] = {}

    @add.pre_execute()
    def _pre(args: dict[str, Any]) -> dict[str, Any]:
        seen["pre"] = True
        return {"a": args["a"] + 10, "b": args["b"]}

    @add.after_execute()
    def _post(result: Any, args: dict[str, Any]) -> Any:
        seen["post"] = True
        return f"[{result}]"

    try:
        args = add._run_pre_execute({"a": 1, "b": 2})
        result = await add(**args)
        result = add._run_after_execute(result, args)
    finally:
        await server.aclose()

    assert seen == {"pre": True, "post": True}
    assert result == "[13]"  # (1 + 10) + 2, then wrapped


@pytest.mark.asyncio
async def test_on_error_hook_fires() -> None:
    server = _mcp_server()
    tools = {t.__name__: t for t in await server.start()}
    add = tools["test.add"]
    observed: list[tuple[str, str]] = []

    @add.on_error()
    def _obs(exc: Exception, phase: str, args: Any, result: Any) -> None:
        observed.append((type(exc).__name__, phase))

    try:
        add._run_on_error(ValueError("x"), "tool", {"a": 1})
    finally:
        await server.aclose()
    assert observed == [("ValueError", "tool")]


# --- Agent integration -------------------------------------------------


@pytest.mark.asyncio
async def test_agent_separates_server_and_resolves_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server handle is kept out of ``_tool_map``; its tools join after resolve."""
    _mock_llm(monkeypatch)
    server = _mcp_server()
    agent = Agent(model="x", provider="openrouter", tools=[server])

    # Before resolution: the server is separated out, not dispatchable.
    assert server in agent._mcp_servers
    assert agent._tool_map == {}
    assert agent._tool_schemas() is None

    await agent._ensure_mcp()
    try:
        assert "test.add" in agent._tool_map
        schemas = agent._tool_schemas()
        assert schemas is not None
        names = [s["function"]["name"] for s in schemas]
        assert "test.add" in names
    finally:
        await agent.aclose()
    assert server._session is None


@pytest.mark.asyncio
async def test_agent_dispatches_mcp_tool_via_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full ``_execute`` round-trip through an MCP tool returns its output."""
    _mock_llm(monkeypatch)
    agent = Agent(model="x", provider="openrouter", tools=[_mcp_server()])
    await agent._ensure_mcp()
    tc = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="test.add", arguments='{"a": 5, "b": 6}')
    )
    try:
        assert await agent._execute(tc) == "11"
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_agent_mcp_failure_keeps_other_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that fails to start contributes nothing; other tools survive."""
    _mock_llm(monkeypatch)

    @tool("noop")
    def noop() -> str:
        """Does nothing."""
        return "ok"

    broken = MCPServer(name="broken", connect=lambda: _FailingCM())
    agent = Agent(model="x", provider="openrouter", tools=[noop, broken])
    await agent._ensure_mcp()
    try:
        assert "noop" in agent._tool_map
        assert "test.add" not in agent._tool_map
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_ensure_mcp_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_ensure_mcp`` connects at most once, however many times it's called."""
    _mock_llm(monkeypatch)
    server_obj = _make_server()
    connect_calls: list[int] = []

    def connect() -> Any:
        connect_calls.append(1)
        return create_connected_server_and_client_session(server_obj)

    agent = Agent(
        model="x", provider="openrouter", tools=[MCPServer(name="t", connect=connect)]
    )
    await agent._ensure_mcp()
    await agent._ensure_mcp()
    try:
        assert len(connect_calls) == 1
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_agent_reconnects_after_aclose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reusing an Agent after ``aclose`` reconnects instead of dispatching
    against dead sessions.

    ``aclose`` must drop the resolved MCP tools and re-arm resolution so a
    later ``_ensure_mcp`` connects afresh — otherwise the stale
    ``MCPClientTool`` entries would fail with "session is not active".
    """
    _mock_llm(monkeypatch)
    server_obj = _make_server()
    connect_calls: list[int] = []

    def connect() -> Any:
        connect_calls.append(1)
        return create_connected_server_and_client_session(server_obj)

    agent = Agent(
        model="x",
        provider="openrouter",
        tools=[MCPServer(name="mcp:t", connect=connect)],
    )
    tc = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="t.add", arguments='{"a": 1, "b": 2}')
    )

    await agent._ensure_mcp()
    assert await agent._execute(tc) == "3"
    await agent.aclose()
    # State is reset: no stale MCP tools, guard re-armed.
    assert "t.add" not in agent._tool_map
    assert agent._mcp_started is False

    # Second cycle reconnects and works again.
    await agent._ensure_mcp()
    try:
        assert "t.add" in agent._tool_map
        assert await agent._execute(tc) == "3"
        assert len(connect_calls) == 2
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_same_server_duplicate_tool_names_first_wins(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-remote-name duplicates from one server: first-write-wins + ERROR."""
    _mock_llm(monkeypatch)

    # Build a server that lists two tools both named "dup".
    from mcp.server.fastmcp import FastMCP

    dup_server = FastMCP("dup-server")

    @dup_server.tool()
    def dup() -> str:
        """First definition."""
        return "first"

    # Inject a second tool of the same name by patching start to return a
    # duplicate of the first tool (same prefixed name).
    agent = Agent(
        model="x",
        provider="openrouter",
        tools=[MCPServer(name="mcp:dup", connect=_in_memory(dup_server))],
    )

    orig_start = type(agent._mcp_servers[0]).start

    async def patched_start(self: Any) -> Any:
        tools = await orig_start(self)
        return tools + tools[:1]  # duplicate the first tool (same prefixed name)

    monkeypatch.setattr(type(agent._mcp_servers[0]), "start", patched_start)

    with caplog.at_level(logging.ERROR, logger="cothis.agent"):
        await agent._ensure_mcp()
    try:
        # Only one "dup" tool registered (first-write-wins).
        assert "dup.dup" in agent._tool_map
        # The duplicate was logged at ERROR.
        assert "already registered" in caplog.text
        assert "dup.dup" in caplog.text
    finally:
        await agent.aclose()
