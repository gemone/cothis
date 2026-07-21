"""Tests for the MCP subsystem (stdio + http) built on the SDK's
``ClientSessionGroup``.

Every test that needs a live session drives a real in-memory MCP server via
the SDK's ``create_connected_server_and_client_session`` transport — no
subprocess, no network, deterministic. The production path
(``MCPServer.connect_into`` → ``group.connect_to_server``) is exercised
end-to-end; only ``connect_to_server`` is swapped for the in-memory transport
(via ``in_memory_group`` for unit tests, or a class-level monkeypatch for
Agent integration tests where ``_ensure_mcp`` builds its own group).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from mcp import ClientSessionGroup
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, ImageContent, Implementation, TextContent

from cothis.agent import Agent
from cothis.tools import (
    MCPClientTool,
    MCPServer,
    tool,
)
from cothis.tools.core import _HookableTool, load_tools_from_layer
from cothis.tools.mcp import (
    _build_mcp_http_server,
    _build_mcp_stdio_server,
    _flatten_exc,
    _normalize_mcp_result,
)
from cothis.tools.yaml import _ShellTool, load_yaml_tools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# --- fixtures / helpers ------------------------------------------------


def _make_server() -> FastMCP:
    """A minimal in-memory MCP server with an ``add`` and a ``boom`` tool.

    The server's self-reported name is ``test-server`` — that's what the
    ``component_name_hook`` prefixes tool names with (``test-server.add``),
    NOT cothis's YAML ``name:`` field. Tests assert the prefixed form.
    """
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


class _FailingParams:
    """A stand-in for SDK ``ServerParameters`` that signals failure.

    Simulates a server that can't launch (bad command, connection refused) so
    the connect-failure path is exercised without a real subprocess. The
    exception is read by a monkeypatched ``connect_to_server`` (the params
    object itself can't raise — it's the transport that would). The exception
    type is injected (plain RuntimeError or an ExceptionGroup, mirroring how
    anyio surfaces transport failures).
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc


@asynccontextmanager
async def in_memory_group(server: FastMCP) -> AsyncIterator[ClientSessionGroup]:
    """Yield a ``ClientSessionGroup`` whose ``connect_to_server`` is in-memory.

    Wraps the group context and patches its ``connect_to_server`` to drive an
    in-memory session (via ``connect_with_session``) instead of spawning a
    subprocess or opening a socket. The session's CM is registered on the
    group's own exit stack so it's torn down with the group. The group carries
    a ``component_name_hook`` that prefixes tool names with the server's
    self-reported name, matching production behaviour.

    With this group, the REAL ``MCPServer.connect_into`` runs (snapshot before
    → ``connect_to_server`` → wrap new tools) — only the transport is faked.
    """

    def _prefix(name: str, server_info: Any) -> str:
        return f"{server_info.name}.{name}"

    async with ClientSessionGroup(component_name_hook=_prefix) as group:

        async def _in_memory_connect_to_server(
            params: Any, session_params: Any = None
        ) -> Any:
            session = await group._exit_stack.enter_async_context(
                create_connected_server_and_client_session(server)
            )
            await group.connect_with_session(
                Implementation(name=server.name or "server", version="1.0"), session
            )
            return session

        # Monkeypatch the group's connect method to use the in-memory transport
        # instead of spawning a real subprocess.
        setattr(group, "connect_to_server", _in_memory_connect_to_server)
        yield group


def _patch_in_memory_transport(
    monkeypatch: pytest.MonkeyPatch,
    fastmcp: FastMCP,
    calls: list[int] | None = None,
) -> None:
    """Monkeypatch ``ClientSessionGroup.connect_to_server`` class-wide to be in-memory.

    Used by Agent integration tests where ``_ensure_mcp`` builds its own group
    internally (so an instance-level patch can't reach it). The real
    ``MCPServer.connect_into`` snapshot+wrap logic still runs against whatever
    group the Agent creates. ``calls``, if given, records one entry per
    connect (so ``runs_once`` / ``reconnects`` tests can count).
    """

    async def _in_memory_connect_to_server(
        self: ClientSessionGroup, params: Any, session_params: Any = None
    ) -> Any:
        if calls is not None:
            calls.append(1)
        session = await self._exit_stack.enter_async_context(  # noqa: SLF001
            create_connected_server_and_client_session(fastmcp)
        )
        await self.connect_with_session(
            Implementation(name=fastmcp.name or "server", version="1.0"), session
        )
        return session

    monkeypatch.setattr(
        ClientSessionGroup, "connect_to_server", _in_memory_connect_to_server
    )


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


def test_mcp_server_label_strips_handle_prefix() -> None:
    """``_label`` strips the ``mcp:`` discovery-handle prefix from
    ``__name__``, yielding the raw YAML ``name:`` value used as the
    tool-name prefix fallback when a server reports an empty
    ``Implementation.name`` (ADR-0005)."""
    assert MCPServer(name="mcp:context7", params=None)._label == "context7"
    # A name without the handle prefix passes through unchanged.
    assert MCPServer(name="custom", params=None)._label == "custom"


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
    result = CallToolResult(content=[TextContent(type="text", text="7")], isError=False)
    assert _normalize_mcp_result(result) == "7"


def test_normalize_multiple_blocks_joined() -> None:
    result = CallToolResult(
        content=[
            TextContent(type="text", text="a"),
            TextContent(type="text", text="b"),
        ],
        isError=False,
    )
    assert _normalize_mcp_result(result) == "a\nb"


def test_normalize_empty_content() -> None:
    result = CallToolResult(content=[], isError=False)
    assert _normalize_mcp_result(result) == "(no output)"


def test_normalize_error_prefixed() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="bad")], isError=True
    )
    assert _normalize_mcp_result(result) == "Error: bad"


def test_normalize_image_only_returns_size_placeholder() -> None:
    """Image-only result returns a placeholder describing the bytes (#92).

    The agent loop is text-only; the model can't see image bytes. But
    it needs to know the tool *did* return an image so it can describe
    the call, ask for a path, or proceed accordingly. The placeholder
    names the mime type + base64 byte count.
    """
    result = CallToolResult(
        content=[ImageContent(type="image", data="<bytes>", mimeType="image/png")],
        isError=False,
    )
    out = _normalize_mcp_result(result)
    assert "image" in out
    assert "image/png" in out
    assert "7 bytes base64" in out  # len("<bytes>") == 7


def test_normalize_resource_block_returns_placeholder() -> None:
    """EmbeddedResource blocks get a placeholder naming the resource."""
    from mcp.types import EmbeddedResource, TextResourceContents
    from pydantic import AnyUrl

    resource = TextResourceContents(
        uri=AnyUrl("https://example.com/hostname.txt"),
        mimeType="text/plain",
        text="example.com",
    )
    result = CallToolResult(
        content=[EmbeddedResource(type="resource", resource=resource)],
        isError=False,
    )
    out = _normalize_mcp_result(result)
    assert "embedded resource" in out
    # Resource URI travels in the placeholder so the model can ask for it.
    assert "example.com/hostname.txt" in out


def test_normalize_mixed_text_and_image_keeps_both() -> None:
    """Text blocks join as before; non-text blocks get inline placeholders."""
    result = CallToolResult(
        content=[
            TextContent(type="text", text="screenshot saved:"),
            ImageContent(type="image", data="<bytes>", mimeType="image/png"),
        ],
        isError=False,
    )
    out = _normalize_mcp_result(result)
    lines = out.splitlines()
    assert lines[0] == "screenshot saved:"
    assert "image/png" in lines[1]
    assert "7 bytes base64" in lines[1]


# --- connect_into + dispatch -------------------------------------------


@pytest.mark.asyncio
async def test_connect_into_discovers_tools_with_schema() -> None:
    """``connect_into`` lists remote tools, each wrapped with its ``inputSchema``
    and a server-prefixed ``__name__`` (ADR-0005). ``_remote_name`` is the same
    prefixed name — the group routes ``call_tool`` by it."""
    server = MCPServer(name="mcp:test-server", params=None)
    async with in_memory_group(_make_server()) as group:
        tools = (await server.connect_into(group))[0]
        by_name = {t.__name__: t for t in tools}
        assert "test-server.add" in by_name
        add = by_name["test-server.add"]
        assert isinstance(add, MCPClientTool)
        assert isinstance(add, _HookableTool)
        assert add.__name__ == "test-server.add"
        # The prefixed name is what ``call_tool`` expects (group keys by it).
        assert add._remote_name == "test-server.add"
        params = add.__cothis_schema__["input_schema"]
        assert add.__cothis_schema__["name"] == "test-server.add"
        assert "a" in params["properties"]
        assert "b" in params["properties"]


@pytest.mark.asyncio
async def test_mcp_tool_call_routes_by_prefixed_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``call_tool`` receives the prefixed ``_remote_name`` — the group's
    ``_tool_to_session`` index is keyed by the prefixed name."""
    server = MCPServer(name="mcp:test-server", params=None)
    async with in_memory_group(_make_server()) as group:
        tools = {t.__name__: t for t in (await server.connect_into(group))[0]}
        add = tools["test-server.add"]
        captured: dict[str, Any] = {}
        orig_call = group.call_tool

        async def spy_call(name: str, arguments: Any = None, **kw: Any) -> Any:
            captured["name"] = name
            return await orig_call(name, arguments, **kw)

        monkeypatch.setattr(group, "call_tool", spy_call)
        assert await add(a=2, b=3) == "5"
    assert captured["name"] == "test-server.add"


@pytest.mark.asyncio
async def test_call_returns_normalized_string() -> None:
    """A successful call returns the result text; two calls prove the session
    is persistent (group-owned, not reconnected per call)."""
    server = MCPServer(name="mcp:test-server", params=None)
    async with in_memory_group(_make_server()) as group:
        tools = {t.__name__: t for t in (await server.connect_into(group))[0]}
        assert await tools["test-server.add"](a=3, b=4) == "7"
        assert await tools["test-server.add"](a=10, b=0) == "10"


@pytest.mark.asyncio
async def test_call_error_prefixed() -> None:
    """A remote tool that raises comes back as an ``Error:`` string."""
    server = MCPServer(name="mcp:test-server", params=None)
    async with in_memory_group(_make_server()) as group:
        tools = {t.__name__: t for t in (await server.connect_into(group))[0]}
        result = await tools["test-server.boom"]()
        assert result.startswith("Error:")
        assert "kaboom" in result


@pytest.mark.asyncio
async def test_connect_into_returns_only_new_tools() -> None:
    """``connect_into`` returns only the tools THIS server contributed — not
    tools already in the group from another server (the before/after snapshot
    is what lets one group host many servers side by side)."""
    from mcp.types import Tool as McpTool

    server = MCPServer(name="mcp:test-server", params=None)
    async with in_memory_group(_make_server()) as group:
        # Pretend another server already registered a tool in this group.
        group._tools[  # noqa: SLF001 — inject a sentinel to test the snapshot
            "other-server.x"
        ] = McpTool(name="x", description="d", inputSchema={"type": "object"})
        tools = {t.__name__: t for t in (await server.connect_into(group))[0]}
        assert "other-server.x" not in tools  # pre-existing tool excluded
        assert "test-server.add" in tools  # new tool included
        assert "test-server.boom" in tools


@pytest.mark.asyncio
async def test_stdio_params_carry_env_to_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder's ``StdioServerParameters`` (with ``env``) reaches
    ``connect_to_server`` unchanged — so a server's secrets reach its
    subprocess (story 32). ``connect_to_server`` is faked to capture params
    without spawning."""
    captured: dict[str, Any] = {}

    async def capture(
        self: ClientSessionGroup, params: Any, session_params: Any = None
    ) -> Any:
        captured["params"] = params
        return None  # don't aggregate any tools

    monkeypatch.setattr(ClientSessionGroup, "connect_to_server", capture)
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
    async with ClientSessionGroup() as group:
        assert (await server.connect_into(group))[0] == []
    assert captured["params"].env == {"API_KEY": "s3cr3t"}
    assert captured["params"].command == "srv"
    assert captured["params"].args == ["--x"]


@pytest.mark.asyncio
async def test_http_params_carry_url_and_headers_to_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder's ``StreamableHttpParameters`` (with ``headers``) reaches
    ``connect_to_server`` unchanged — so auth secrets reach the remote
    (story 32)."""
    captured: dict[str, Any] = {}

    async def capture(
        self: ClientSessionGroup, params: Any, session_params: Any = None
    ) -> Any:
        captured["params"] = params
        return None

    monkeypatch.setattr(ClientSessionGroup, "connect_to_server", capture)
    server = _build_mcp_http_server(
        {
            "type": "mcp.http",
            "name": "t",
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer s3cr3t"},
        },
        source=None,
    )
    async with ClientSessionGroup() as group:
        assert (await server.connect_into(group))[0] == []
    assert captured["params"].url == "https://example.com/mcp"
    assert captured["params"].headers == {"Authorization": "Bearer s3cr3t"}


@pytest.mark.asyncio
async def test_connect_into_failure_returns_empty(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connect failure logs at WARNING (with diagnostic, never secrets) and
    yields ``[]`` so the rest of the agent's tools still load (story 30).

    ``_FailingParams`` carries the exception; the monkeypatched
    ``connect_to_server`` reads it off ``params._exc`` and raises — so the
    exception type is injected (plain RuntimeError here, ExceptionGroup below)."""

    async def boom(
        self: ClientSessionGroup, params: Any, session_params: Any = None
    ) -> Any:
        raise params._exc

    monkeypatch.setattr(ClientSessionGroup, "connect_to_server", boom)
    server = MCPServer(
        name="mcp:broken",
        params=_FailingParams(RuntimeError("connect refused")),
        diagnostic="command='badcmd'",
    )
    async with ClientSessionGroup() as group:
        with caplog.at_level(logging.WARNING, logger="cothis.tools"):
            tools = (await server.connect_into(group))[0]
    assert tools == []
    assert "broken" in caplog.text
    assert "connect refused" in caplog.text
    assert "badcmd" in caplog.text


@pytest.mark.asyncio
async def test_stdio_connect_failure_logs_warning_returns_empty(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch failure logs at WARNING (command/args, never env) and yields [].

    Built via the real builder, so the assertion that ``env`` is absent proves
    the *builder* keeps secrets out of the diagnostic (story 32)."""

    async def boom(
        self: ClientSessionGroup, params: Any, session_params: Any = None
    ) -> Any:
        raise RuntimeError("connect refused")

    monkeypatch.setattr(ClientSessionGroup, "connect_to_server", boom)
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
    async with ClientSessionGroup() as group:
        with caplog.at_level(logging.WARNING, logger="cothis.tools"):
            tools = (await server.connect_into(group))[0]
    assert tools == []
    assert "broken" in caplog.text
    assert "connect refused" in caplog.text
    # Safe diagnostics are present…
    assert "badcmd" in caplog.text
    assert "--flag" in caplog.text
    # …but env secrets are never logged (story 32).
    assert "topsecret" not in caplog.text
    assert "SECRET_KEY" not in caplog.text


@pytest.mark.asyncio
async def test_http_connect_failure_logs_url_never_headers(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An http connection failure logs at WARNING naming the url — never the
    ``headers`` (Authorization secret, story 32) — and yields []."""

    async def boom(
        self: ClientSessionGroup, params: Any, session_params: Any = None
    ) -> Any:
        raise RuntimeError("connect refused")

    monkeypatch.setattr(ClientSessionGroup, "connect_to_server", boom)
    server = _build_mcp_http_server(
        {
            "type": "mcp.http",
            "name": "broken",
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer topsecret"},
        },
        source=None,
    )
    async with ClientSessionGroup() as group:
        with caplog.at_level(logging.WARNING, logger="cothis.tools"):
            tools = (await server.connect_into(group))[0]
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


def test_flatten_exc_scrubs_url_secrets() -> None:
    """httpx errors embed the full request URL — query string included — so
    the flattened text must pass every URL through ``_scrub_url`` (story 32).
    Uses a real ``raise_for_status`` error, wrapped in an ExceptionGroup the
    way anyio surfaces transport failures."""
    import httpx

    response = httpx.Response(
        401,
        request=httpx.Request("GET", "https://mcp.example.com/mcp?api_key=SECRET123"),
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        group = ExceptionGroup("unhandled errors in a TaskGroup (1 sub-exception)", [exc])
    msg = _flatten_exc(group)
    assert "SECRET123" not in msg
    assert "api_key" not in msg
    # The scrubbed URL itself survives — the log stays actionable.
    assert "https://mcp.example.com/mcp" in msg
    assert "HTTPStatusError" in msg


@pytest.mark.asyncio
async def test_connect_failure_unwraps_taskgroup_in_warning(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the transport fails inside an anyio TaskGroup, the WARNING names
    the real cause, not 'unhandled errors in a TaskGroup'."""

    async def boom(
        self: ClientSessionGroup, params: Any, session_params: Any = None
    ) -> Any:
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup (1 sub-exception)",
            [ConnectionError("Name or service not known")],
        )

    monkeypatch.setattr(ClientSessionGroup, "connect_to_server", boom)
    server = _build_mcp_http_server(
        {"type": "mcp.http", "name": "remote", "url": "https://example.com/mcp"},
        source=None,
    )
    async with ClientSessionGroup() as group:
        with caplog.at_level(logging.WARNING, logger="cothis.tools"):
            assert (await server.connect_into(group))[0] == []
    assert "Name or service not known" in caplog.text
    assert "unhandled errors in a TaskGroup" not in caplog.text


# --- hooks (inheritance from _HookableTool) ----------------------------


@pytest.mark.asyncio
async def test_pre_and_after_execute_hooks_run() -> None:
    """pre_execute/after_execute pipelines wrap the async MCP call."""
    server = MCPServer(name="mcp:test-server", params=None)
    async with in_memory_group(_make_server()) as group:
        tools = {t.__name__: t for t in (await server.connect_into(group))[0]}
        add = tools["test-server.add"]
        seen: dict[str, bool] = {}

        @add.pre_execute()
        def _pre(args: dict[str, Any]) -> dict[str, Any]:
            seen["pre"] = True
            return {"a": args["a"] + 10, "b": args["b"]}

        @add.after_execute()
        def _post(result: Any, args: dict[str, Any]) -> Any:
            seen["post"] = True
            return f"[{result}]"

        args = add._run_pre_execute({"a": 1, "b": 2})
        result = await add(**args)
        result = add._run_after_execute(result, args)

    assert seen == {"pre": True, "post": True}
    assert result == "[13]"  # (1 + 10) + 2, then wrapped


@pytest.mark.asyncio
async def test_on_error_hook_fires() -> None:
    server = MCPServer(name="mcp:test-server", params=None)
    async with in_memory_group(_make_server()) as group:
        tools = {t.__name__: t for t in (await server.connect_into(group))[0]}
        add = tools["test-server.add"]
        observed: list[tuple[str, str]] = []

        @add.on_error()
        def _obs(exc: Exception, phase: str, args: Any, result: Any) -> None:
            observed.append((type(exc).__name__, phase))

        add._run_on_error(ValueError("x"), "tool", {"a": 1})
    assert observed == [("ValueError", "tool")]


# --- Agent integration -------------------------------------------------


@pytest.mark.asyncio
async def test_agent_separates_server_and_resolves_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server handle is kept out of ``_tool_map``; its tools join after resolve."""
    _mock_llm(monkeypatch)
    _patch_in_memory_transport(monkeypatch, _make_server())
    server = MCPServer(name="mcp:test-server", params=None)
    agent = Agent(model="x", provider="openrouter", tools=[server])

    # Before resolution: the server is separated out, not dispatchable.
    assert server in agent._mcp_servers
    assert agent._tool_map == {}
    assert agent._tool_schemas() is None

    await agent._ensure_mcp()
    try:
        assert "test-server_add" in agent._tool_map
        schemas = agent._tool_schemas()
        assert schemas is not None
        names = [s["name"] for s in schemas]
        assert "test-server_add" in names
    finally:
        await agent.aclose()
    # aclose tears the group down and re-arms resolution.
    assert agent._mcp_group is None


@pytest.mark.asyncio
async def test_agent_dispatches_mcp_tool_via_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full ``_execute`` round-trip through an MCP tool returns its output."""
    _mock_llm(monkeypatch)
    _patch_in_memory_transport(monkeypatch, _make_server())
    agent = Agent(
        model="x",
        provider="openrouter",
        tools=[MCPServer(name="mcp:test-server", params=None)],
    )
    await agent._ensure_mcp()
    tu = {"name": "test-server_add", "input": {"a": 5, "b": 6}}
    try:
        assert await agent._execute_tool(tu) == (False, "11")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_agent_mcp_failure_keeps_other_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that fails to connect contributes nothing; other tools survive."""

    async def boom(
        self: ClientSessionGroup, params: Any, session_params: Any = None
    ) -> Any:
        raise RuntimeError("connect refused")

    monkeypatch.setattr(ClientSessionGroup, "connect_to_server", boom)
    _mock_llm(monkeypatch)

    @tool("noop")
    def noop() -> str:
        """Does nothing."""
        return "ok"

    broken = MCPServer(name="mcp:broken", params=None)
    agent = Agent(model="x", provider="openrouter", tools=[noop, broken])
    await agent._ensure_mcp()
    try:
        assert "noop" in agent._tool_map
        assert "test-server_add" not in agent._tool_map
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_ensure_mcp_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_ensure_mcp`` connects at most once, however many times it's called."""
    _mock_llm(monkeypatch)
    calls: list[int] = []
    _patch_in_memory_transport(monkeypatch, _make_server(), calls=calls)
    agent = Agent(
        model="x",
        provider="openrouter",
        tools=[MCPServer(name="mcp:test-server", params=None)],
    )
    await agent._ensure_mcp()
    await agent._ensure_mcp()
    try:
        assert len(calls) == 1
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_agent_aclose_safe_if_never_ensured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``aclose`` before any ``_ensure_mcp`` is a no-op, not an error.

    The group is ``None`` until ``_ensure_mcp`` runs; ``aclose`` guards on that
    (idempotent teardown)."""
    _mock_llm(monkeypatch)
    agent = Agent(
        model="x",
        provider="openrouter",
        tools=[MCPServer(name="mcp:test-server", params=None)],
    )
    await agent.aclose()
    assert agent._mcp_group is None
    assert agent._mcp_started is False


@pytest.mark.asyncio
async def test_agent_reconnects_after_aclose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reusing an Agent after ``aclose`` reconnects instead of dispatching
    against a torn-down group.

    ``aclose`` must drop the resolved MCP tools and re-arm resolution so a
    later ``_ensure_mcp`` connects afresh — otherwise the stale
    ``MCPClientTool`` entries would dispatch against a closed group.
    """
    _mock_llm(monkeypatch)
    calls: list[int] = []
    _patch_in_memory_transport(monkeypatch, _make_server(), calls=calls)
    agent = Agent(
        model="x",
        provider="openrouter",
        tools=[MCPServer(name="mcp:test-server", params=None)],
    )
    tu = {"name": "test-server_add", "input": {"a": 1, "b": 2}}

    await agent._ensure_mcp()
    assert await agent._execute_tool(tu) == (False, "3")
    await agent.aclose()
    # State is reset: no stale MCP tools, guard re-armed.
    assert "test-server_add" not in agent._tool_map
    assert agent._mcp_started is False

    # Second cycle reconnects and works again.
    await agent._ensure_mcp()
    try:
        assert "test-server_add" in agent._tool_map
        assert await agent._execute_tool(tu) == (False, "3")
        assert len(calls) == 2
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_duplicate_prefixed_tool_name_first_wins(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two ``MCPClientTool``s resolving to the same prefixed name: first-write-wins,
    the duplicate is logged at ERROR (ADR-0005).

    Simulated by patching one server's ``connect_into`` to return its first
    tool twice (same ``__name__``). The Agent's ``_ensure_mcp`` keeps the
    existing entry and skips the collision."""
    _mock_llm(monkeypatch)
    _patch_in_memory_transport(monkeypatch, _make_server())
    server = MCPServer(name="mcp:test-server", params=None)
    original_connect_into = server.connect_into

    async def duplicating_connect_into(
        group: ClientSessionGroup,
    ) -> tuple[list[MCPClientTool], Any]:
        tools, session = await original_connect_into(group)
        return tools + tools[:1], session  # duplicate the first tool (same prefixed name)

    setattr(server, "connect_into", duplicating_connect_into)
    agent = Agent(model="x", provider="openrouter", tools=[server])

    with caplog.at_level(logging.ERROR, logger="cothis.agent"):
        await agent._ensure_mcp()
    try:
        # Only one ``test-server.add`` registered (first-write-wins).
        assert sum(1 for n in agent._tool_map if n == "test-server_add") == 1
        # The duplicate was logged at ERROR.
        assert "already registered" in caplog.text
        assert "test-server.add" in caplog.text
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_prefix_falls_back_to_yaml_label_when_server_name_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a server reports an empty ``Implementation.name``, the tool prefix
    falls back to the YAML ``name:`` label (ADR-0005). Defensive against a
    non-conformant server that sends an empty string where the spec requires a
    non-empty name."""
    _mock_llm(monkeypatch)
    fastmcp = _make_server()

    async def _empty_name_connect(
        self: ClientSessionGroup, params: Any, session_params: Any = None
    ) -> Any:
        session = await self._exit_stack.enter_async_context(  # noqa: SLF001
            create_connected_server_and_client_session(fastmcp)
        )
        # Server reports an empty name — the non-conformant case.
        await self.connect_with_session(Implementation(name="", version="1.0"), session)
        return session

    monkeypatch.setattr(ClientSessionGroup, "connect_to_server", _empty_name_connect)
    # MCPServer label is "my-label" (from YAML name:); the server reports "".
    server = MCPServer(name="mcp:my-label", params=None)
    agent = Agent(model="x", provider="openrouter", tools=[server])

    await agent._ensure_mcp()
    try:
        # Prefix falls back to the YAML label, not the empty server name.
        assert "my-label_add" in agent._tool_map
        assert "my-label_boom" in agent._tool_map
        # No bare or dot-prefixed name leaked through.
        assert ".add" not in agent._tool_map
        assert "add" not in agent._tool_map
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_empty_name_prefix_stable_across_reacquire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-acquire after reclaim must reuse the server's OWN fallback label.

    Two servers both report an empty ``Implementation.name``. After the
    first server's session is reclaimed and re-acquired, its tools must
    re-register under ``aaa.`` (their advertised prefix), not under the
    last-connected server's label — otherwise ``call_tool`` routes by a
    name the group no longer has and raises ``KeyError`` for the rest of
    the session (ADR-0005 §4).
    """
    _mock_llm(monkeypatch)
    fastmcp = _make_server()

    async def _empty_name_connect(
        self: ClientSessionGroup, params: Any, session_params: Any = None
    ) -> Any:
        session = await self._exit_stack.enter_async_context(  # noqa: SLF001
            create_connected_server_and_client_session(fastmcp)
        )
        # Server reports an empty name — the non-conformant case.
        await self.connect_with_session(Implementation(name="", version="1.0"), session)
        return session

    monkeypatch.setattr(ClientSessionGroup, "connect_to_server", _empty_name_connect)
    aaa = MCPServer(name="mcp:aaa", params=None)
    zzz = MCPServer(name="mcp:zzz", params=None)
    agent = Agent(model="x", provider="openrouter", tools=[aaa, zzz])

    await agent._ensure_mcp()
    try:
        assert "aaa_add" in agent._tool_map
        assert "zzz_add" in agent._tool_map
        tool_a = agent._tool_map["aaa_add"]
        cls_a = getattr(tool_a, "_handle_cls")

        # Reclaim aaa's session only; zzz stays live.
        import time

        agent._handle_manager._last_used[cls_a] = time.time() - 9999
        await agent._handle_manager.reclaim_idle()
        assert cls_a not in agent._handle_manager._live

        # Re-acquire: tools must come back under aaa.* and dispatch works.
        await agent._handle_manager.ensure_acquired(tool_a)
        assert cls_a in agent._handle_manager._live
        assert await tool_a(a=2, b=3) == "5"
    finally:
        await agent.aclose()


# --- MCP session handle lifecycle (ADR-0005) ---------------------------


def test_yaml_mcp_keepalive_parsed() -> None:
    """``keepalive:`` is parsed into ``MCPServer.keepalive``."""
    server = load_yaml_tools("type: mcp.stdio\ncommand: srv\nkeepalive: 42\n")[0]
    assert isinstance(server, MCPServer)
    assert server.keepalive == 42.0


def test_yaml_mcp_pin_parsed() -> None:
    """``pin: true`` is parsed into ``MCPServer.pin``."""
    server = load_yaml_tools("type: mcp.stdio\ncommand: srv\npin: true\n")[0]
    assert isinstance(server, MCPServer)
    assert server.pin is True


def test_yaml_mcp_keepalive_default() -> None:
    """Without ``keepalive:``, defaults to 600s."""
    server = load_yaml_tools("type: mcp.stdio\ncommand: srv\n")[0]
    assert isinstance(server, MCPServer)
    assert server.keepalive == 600.0


def test_yaml_mcp_keepalive_invalid_raises() -> None:
    """Non-numeric ``keepalive`` raises with an actionable message."""
    with pytest.raises(ValueError, match="keepalive"):
        load_yaml_tools("type: mcp.stdio\ncommand: srv\nkeepalive: abc\n")


def test_yaml_mcp_keepalive_negative_raises() -> None:
    """Zero/negative ``keepalive`` raises."""
    with pytest.raises(ValueError, match="keepalive"):
        load_yaml_tools("type: mcp.stdio\ncommand: srv\nkeepalive: 0\n")


def test_yaml_mcp_pin_non_bool_raises() -> None:
    """Non-boolean ``pin`` raises with an actionable message."""
    with pytest.raises(ValueError, match="pin"):
        load_yaml_tools("type: mcp.stdio\ncommand: srv\npin: 1\n")


@pytest.mark.asyncio
async def test_mcp_startup_adopts_session_into_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """After ``_ensure_mcp``, the MCP session handle is live in the pool."""
    _mock_llm(monkeypatch)
    calls: list[int] = []
    _patch_in_memory_transport(monkeypatch, _make_server(), calls=calls)
    server = MCPServer(name="mcp:test-server", params=None)
    agent = Agent(model="x", provider="openrouter", tools=[server])

    await agent._ensure_mcp()
    try:
        # One connect at startup.
        assert len(calls) == 1
        # The MCP tool is registered.
        assert "test-server_add" in agent._tool_map
        # Its handle class is live in the pool.
        mcp_tool = agent._tool_map["test-server_add"]
        handle_cls = getattr(mcp_tool, "_handle_cls")
        assert handle_cls is not None
        assert handle_cls in agent._handle_manager._live
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_mcp_keepalive_reclaims_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """An idle MCP session past keepalive is released (disconnected)."""
    _mock_llm(monkeypatch)
    _patch_in_memory_transport(monkeypatch, _make_server())
    server = MCPServer(name="mcp:test-server", params=None, keepalive=0.01)
    agent = Agent(model="x", provider="openrouter", tools=[server])

    await agent._ensure_mcp()
    try:
        mcp_tool = agent._tool_map["test-server_add"]
        handle_cls = getattr(mcp_tool, "_handle_cls")
        assert handle_cls in agent._handle_manager._live

        import time

        agent._handle_manager._last_used[handle_cls] = time.time() - 100
        reclaimed = await agent._handle_manager.reclaim_idle()
        assert reclaimed >= 1
        assert handle_cls not in agent._handle_manager._live
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_mcp_self_heal_reconnects_on_next_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After reclamation, the next call re-acquires (reconnects) the session."""
    _mock_llm(monkeypatch)
    calls: list[int] = []
    _patch_in_memory_transport(monkeypatch, _make_server(), calls=calls)
    server = MCPServer(name="mcp:test-server", params=None, keepalive=0.01)
    agent = Agent(model="x", provider="openrouter", tools=[server])

    await agent._ensure_mcp()
    try:
        mcp_tool = agent._tool_map["test-server_add"]
        handle_cls = getattr(mcp_tool, "_handle_cls")

        # Reclaim the session.
        import time

        agent._handle_manager._last_used[handle_cls] = time.time() - 100
        await agent._handle_manager.reclaim_idle()
        assert handle_cls not in agent._handle_manager._live
        connect_count_after_reclaim = len(calls)

        # Self-heal: ensure_acquired re-connects.
        await agent._handle_manager.ensure_acquired(mcp_tool)
        assert handle_cls in agent._handle_manager._live
        assert len(calls) == connect_count_after_reclaim + 1
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_mcp_pin_session_not_reclaimed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pinned MCP server's session survives ``reclaim_idle``."""
    _mock_llm(monkeypatch)
    _patch_in_memory_transport(monkeypatch, _make_server())
    server = MCPServer(name="mcp:test-server", params=None, pin=True)
    agent = Agent(model="x", provider="openrouter", tools=[server])

    await agent._ensure_mcp()
    try:
        mcp_tool = agent._tool_map["test-server_add"]
        handle_cls = getattr(mcp_tool, "_handle_cls")

        import time

        agent._handle_manager._last_used[handle_cls] = time.time() - 9999
        reclaimed = await agent._handle_manager.reclaim_idle()
        assert reclaimed == 0
        assert handle_cls in agent._handle_manager._live
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_mcp_connect_into_returns_session() -> None:
    """``connect_into`` returns ``(tools, session)``, not just tools."""
    async with in_memory_group(_make_server()) as group:
        server = MCPServer(name="mcp:test-server", params=None)
        tools, session = await server.connect_into(group)
        assert len(tools) > 0
        assert session is not None


@pytest.mark.asyncio
async def test_mcp_self_heal_dispatch_after_reclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: after a session is reclaimed, the next ``_execute`` call
    transparently reconnects and succeeds — the LLM never sees a failure."""
    _mock_llm(monkeypatch)
    _patch_in_memory_transport(monkeypatch, _make_server())
    server = MCPServer(name="mcp:test-server", params=None, keepalive=0.01)
    agent = Agent(model="x", provider="openrouter", tools=[server])

    await agent._ensure_mcp()
    try:
        # Call 1 — works.
        tu = {"name": "test-server_add", "input": {"a": 2, "b": 3}}
        is_error, result = await agent._execute_tool(tu)
        assert is_error is False
        assert "5" in result

        # Reclaim the session (idle past keepalive).
        import time

        mcp_tool = agent._tool_map["test-server_add"]
        handle_cls = getattr(mcp_tool, "_handle_cls")
        agent._handle_manager._last_used[handle_cls] = time.time() - 100
        await agent._handle_manager.reclaim_idle()
        assert handle_cls not in agent._handle_manager._live

        # Call 2 — self-heal: _execute_tool's ensure_handle_ready reconnects.
        tu = {"name": "test-server_add", "input": {"a": 10, "b": 20}}
        is_error, result = await agent._execute_tool(tu)
        assert is_error is False
        assert "30" in result
    finally:
        await agent.aclose()
