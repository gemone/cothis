"""Subprocess round-trip for :func:`cothis.protocol.acp_client.connect_stdio`.

Spawns an in-test fake ACP server (a child ``python -c`` process) and drives
a real :class:`ACPClient` through it: handshake, list, create, prompt, clean
shutdown. Proves the ``_SubprocessByteConnection`` + ``connect_stdio`` path
works against a real child process boundary. Skipped when the child cannot
import ``cothis`` (e.g. environments that build the workspace differently).
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest

from cothis.protocol.acp_client import ACPClient, connect_stdio
from cothis.protocol.messages import ModelDescriptor

# An inline fake ACP server: speaks the protocol over stdio using a fake
# backend (no LLM, no API keys). The token is fixed to keep the script
# self-contained. Run as ``sys.executable -c SERVER_SCRIPT``.
_SERVER_SCRIPT = """
import asyncio
import os
import sys

from cothis.protocol.acp import ACPServer
from cothis.protocol.messages import (
    AssistantDelta,
    ModelDescriptor,
    ModelRef,
    SessionSnapshot,
)

class FakeBackend:
    async def models(self):
        return [ModelDescriptor(provider="openrouter", id="openai/gpt-oss-120b")]
    async def list_sessions(self):
        return []
    async def create_session(self, cwd, name, model, thinking_level):
        return SessionSnapshot(
            id="s1", cwd=cwd or "/", phase="idle",
            model=ModelRef(provider="p", id="m"), thinkingLevel="off",
            createdAt=0, updatedAt=0, revision=0, transcript=[],
        )
    async def prompt(self, sid, text, emit):
        await emit(AssistantDelta(
            type="assistant_delta", messageId="m1", contentIndex=0,
            kind="text", delta="Hi",
        ))
        return SessionSnapshot(
            id=sid, cwd="/", phase="idle",
            model=ModelRef(provider="p", id="m"), thinkingLevel="off",
            createdAt=0, updatedAt=0, revision=1, transcript=[],
        )

class StdioConn:
    def __init__(self):
        self.closed = False
    async def send(self, chunk):
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    async def close(self, final_chunk=None):
        if final_chunk is not None:
            sys.stdout.buffer.write(final_chunk)
            sys.stdout.buffer.flush()
        self.closed = True
    def __aiter__(self):
        return self
    async def __anext__(self):
        chunk = await asyncio.to_thread(os.read, sys.stdin.fileno(), 65536)
        if not chunk:
            raise StopAsyncIteration
        return bytes(chunk)

async def main():
    await ACPServer(FakeBackend(), token="secret").serve_connection(StdioConn())

asyncio.run(main())
"""


def _child_can_import_cothis() -> bool:
    """True if a fresh ``sys.executable`` child can import the protocol package."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import cothis.protocol.acp, cothis.protocol.acp_client"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _child_can_import_cothis(),
    reason="cothis is not importable in a child process (workspace not installed)",
)


@pytest.mark.asyncio
async def test_connect_stdio_handshake_list_create_prompt() -> None:
    client = await connect_stdio(
        [sys.executable, "-c", _SERVER_SCRIPT],
        "secret",
        start_timeout=20.0,
    )
    assert isinstance(client, ACPClient)
    try:
        # connect_stdio already completed the handshake.
        assert client.snapshot is not None
        assert client.snapshot.models == [
            ModelDescriptor(provider="openrouter", id="openai/gpt-oss-120b")
        ]

        sessions = await client.list_sessions()
        assert sessions == []

        created = await client.create_session(cwd="/tmp")
        assert created.id == "s1"

        progress = [p async for p in client.prompt(created.id, "hello")]
        assert len(progress) == 1
        assert progress[0].type == "assistant_delta"
        assert progress[0].delta == "Hi"
        assert client.last_prompt_snapshot is not None
        assert client.last_prompt_snapshot.revision == 1
    finally:
        await client.aclose()
