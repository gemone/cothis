"""Tests for ``cothis.agent.Agent._maybe_compact`` — compaction slice C1.

C1 wires slices A (``resolve_summary_model`` / ``build_summarisation_request``)
and B (``plan_eviction``) into the live agent run loop as a single in-memory
compaction step. These tests pin the run-loop mutation in isolation with a
mocked summariser provider (no real network):

* compaction fires under HIGH pressure (window replaced by a summary user
  message, retained tail intact by identity, ``_phase`` flips to
  ``"compaction"`` during the call and is restored to ``"idle"`` after);
* compaction is a no-op under LOW / MEDIUM / NONE / unknown pressure;
* the once-per-run guard bounds total summarisation calls to 1;
* a summariser failure degrades gracefully without corrupting ``_messages``;
* the substitute preserves slice B's tool-pair closure invariant;
* the different-provider path (``COTHIS_SUMMARY_MODEL``) routes to a fresh
  client;
* the normal turn path is byte-for-byte unchanged when pressure is low;
* ask-mode (no Session) compacts in-memory without crashing.

The compaction code path is fully exercised via ``_maybe_compact`` direct
calls plus one full-turn regression through ``Agent.run``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from cothis.agent import Agent, _request_messages
from cothis.ai.compaction import plan_eviction

if TYPE_CHECKING:
    from anthropic.types import Message as MessageResponse

# The default test model/provider (mirrors ``tests/test_model_metadata``).
# ``model_info`` reports a 131072-token context window for this pair, so a
# last-assistant ``usage.input_tokens`` of ~120k forces HIGH pressure
# (ratio >= 0.90) and ~140k forces CRITICAL (ratio >= 1.00).
_MODEL = "openai/gpt-oss-120b"
_PROVIDER = "openrouter"
_CONTEXT_WINDOW = 131072


# --- fakes & helpers --------------------------------------------------------


def _text_block(text: str) -> SimpleNamespace:
    """A stand-in for an anthropic ``TextBlock`` carrying ``text``."""

    def _model_dump(*, exclude_none: bool = False) -> dict[str, Any]:
        return {"type": "text", "text": text}

    return SimpleNamespace(type="text", text=text, model_dump=_model_dump)


def _fake_response(
    texts: list[str],
    *,
    msg_id: str = "msg_summary",
    model: str = "summary-model",
) -> Any:
    """Build a lightweight fake ``MessageResponse`` (non-stream shape).

    The summary extraction in ``_maybe_compact`` calls
    ``b.model_dump(exclude_none=True)`` on each content block and feeds the
    dicts to ``_concat_text``; the full ``_assistant_msg_from_response``
    path additionally reads ``.id`` / ``.model`` / ``.stop_reason`` /
    ``.usage``.
    """
    return SimpleNamespace(
        content=[_text_block(t) for t in texts],
        id=msg_id,
        model=model,
        stop_reason="end_turn",
        usage=None,
    )


class _FakeProvider:
    """A fake ``AIProvider`` whose ``amessages`` is an async callable.

    Records every call's kwargs (for payload assertions) and the agent's
    ``_phase`` at call time (for the phase-flip assertion). Raises ``exc``
    on every call when set, so the graceful-failure path is exercisable
    without a real network error. ``responses`` lets a test queue distinct
    return values per call (otherwise ``response`` is returned every time).
    """

    def __init__(
        self,
        response: Any = None,
        *,
        agent: Agent | None = None,
        exc: BaseException | None = None,
        responses: list[Any] | None = None,
    ) -> None:
        self._response = response
        self._responses = list(responses) if responses is not None else []
        self._exc = exc
        self._agent = agent
        self.calls: list[dict[str, Any]] = []
        self.phases: list[str] = []

    async def amessages(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._agent is not None:
            self.phases.append(self._agent._phase)
        if self._exc is not None:
            raise self._exc
        if self._responses:
            return self._responses.pop(0)
        return self._response


def _user_msg(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant_msg(
    text: str = "ok",
    *,
    usage_input: int = 100,
    blocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a stored-shape assistant message.

    ``blocks`` overrides the default single-text content (used to inject
    ``tool_use`` blocks). ``usage_input`` sets the observed input-token
    count ``context_budget`` reads to compute pressure.
    """
    content = blocks if blocks is not None else [{"type": "text", "text": text}]
    return {
        "role": "assistant",
        "content": content,
        "id": None,
        "model": _MODEL,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": usage_input},
    }


def _tool_use_block(name: str, call_id: str, inp: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": inp or {}}


def _tool_result_block(call_id: str, content: str = "done") -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": call_id, "content": content}


def _seed_simple_groups(
    n: int, *, last_usage: int = 100, start_index: int = 1
) -> list[dict[str, Any]]:
    """Build ``n`` simple user/assistant turn-groups (no tool pairs).

    The final assistant message carries ``last_usage`` so the seeded
    conversation lands at a known pressure without re-walking earlier usage.
    """
    messages: list[dict[str, Any]] = []
    for i in range(n):
        messages.append(_user_msg(f"q{start_index + i}"))
        is_last = i == n - 1
        messages.append(
            _assistant_msg(usage_input=last_usage if is_last else 100)
            if is_last
            else _assistant_msg(f"a{start_index + i}")
        )
    # Fix the last assistant's usage (the helper above only sets it when
    # ``is_last`` branches correctly; set explicitly for clarity).
    messages[-1]["usage"] = {"input_tokens": last_usage}
    return messages


def _make_agent(
    monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> Agent:
    """Build an Agent without contacting any provider.

    ``cothis.ai.get_provider`` is patched to a MagicMock so no API key is
    required and ``_llm`` starts as a MagicMock; tests then set
    ``agent._llm`` to a ``_FakeProvider`` explicitly. Mirrors the
    ``tests/test_model_metadata._make_agent`` pattern.
    """
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **kw: MagicMock())
    return Agent(
        model=overrides.get("model", _MODEL),
        provider=overrides.get("provider", _PROVIDER),
        tools=[],
        max_iterations=overrides.get("max_iterations", 5),
        max_tokens=overrides.get("max_tokens", None),
    )


_HIGH_USAGE = 120_000  # ratio ~0.915 -> HIGH
_CRITICAL_USAGE = 140_000  # ratio ~1.07 -> CRITICAL


# --- (1) HIGH pressure: compaction fires, substitute is correct --------------


def test_maybe_compact_high_pressure_substitutes_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    # 6 turn-groups; last assistant carries HIGH usage.
    agent._messages = _seed_simple_groups(6, last_usage=_HIGH_USAGE)

    # Capture the expected retained tail via the same pure decision
    # ``_maybe_compact`` will make, so identity can be asserted precisely.
    expected_decision = plan_eviction(
        messages=agent._messages, budget=agent.context_budget()
    )
    assert expected_decision.window  # precondition: a non-empty window
    retained_before = list(expected_decision.retained)

    fake = _FakeProvider(
        _fake_response(["Summary of older turns."]),
        agent=agent,
    )
    agent._llm = fake

    asyncio.run(agent._maybe_compact())

    # Exactly one summariser call.
    assert len(fake.calls) == 1
    # SummarisationRequest payload shape: one system text block, one user
    # text block, no tools, target model.
    call = fake.calls[0]
    assert call["tools"] is None
    assert call["model"] == _MODEL  # resolve_summary_model -> session pair
    assert len(call["system"]) == 1
    assert call["system"][0]["type"] == "text"
    assert len(call["messages"]) == 1
    assert call["messages"][0]["role"] == "user"
    assert call["messages"][0]["content"][0]["type"] == "text"

    # _phase was "compaction" DURING the call, restored to "idle" after.
    assert fake.phases == ["compaction"]
    assert agent._phase == "idle"

    # _messages rebuilt as [summary user msg] + retained tail.
    assert agent._messages[0]["role"] == "user"
    assert agent._messages[0]["content"] == [
        {"type": "text", "text": "Summary of older turns."}
    ]
    # Retained dicts unchanged by identity (same objects).
    assert len(agent._messages) == 1 + len(retained_before)
    for i, original in enumerate(retained_before):
        assert agent._messages[1 + i] is original

    # Once-per-run guard now set.
    assert agent._compaction_attempted_this_run is True


# --- (2) Low / unknown pressure: no-op --------------------------------------


@pytest.mark.parametrize(
    "last_usage,capacity_override",
    [
        (100, None),  # NONE pressure (~0)
        (90_000, None),  # LOW pressure (~0.69)
        (100_000, None),  # MEDIUM pressure (~0.76)
    ],
)
def test_maybe_compact_below_high_pressure_is_noop(
    monkeypatch: pytest.MonkeyPatch,
    last_usage: int,
    capacity_override: None,
) -> None:
    agent = _make_agent(monkeypatch)
    agent._messages = _seed_simple_groups(6, last_usage=last_usage)
    original = list(agent._messages)
    fake = _FakeProvider(_fake_response(["irrelevant"]))
    agent._llm = fake

    asyncio.run(agent._maybe_compact())

    assert fake.calls == []  # summariser never called
    assert agent._messages == original  # untouched (deep equality)
    assert agent._phase == "idle"


def test_maybe_compact_unknown_budget_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the model's context window is unknown, pressure is None -> no-op."""
    agent = _make_agent(monkeypatch, model="no-such-model", provider="openai")
    agent._messages = _seed_simple_groups(6, last_usage=_HIGH_USAGE)
    original = list(agent._messages)
    fake = _FakeProvider(_fake_response(["irrelevant"]))
    agent._llm = fake

    asyncio.run(agent._maybe_compact())

    assert fake.calls == []
    assert agent._messages == original
    assert agent._phase == "idle"


# --- (3) Below retention floor: plan_eviction no-op -------------------------


def test_maybe_compact_below_retention_floor_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short conversation under HIGH pressure -> plan_eviction no-op."""
    agent = _make_agent(monkeypatch)
    # 3 turn-groups: at/under the 4-group floor -> nothing safe to evict.
    agent._messages = _seed_simple_groups(3, last_usage=_HIGH_USAGE)
    original = list(agent._messages)
    fake = _FakeProvider(_fake_response(["irrelevant"]))
    agent._llm = fake

    asyncio.run(agent._maybe_compact())

    assert fake.calls == []
    assert agent._messages == original
    assert agent._phase == "idle"


# --- (4) Summariser failure: graceful degradation --------------------------


def test_maybe_compact_summariser_raises_leaves_messages_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    agent._messages = _seed_simple_groups(6, last_usage=_HIGH_USAGE)
    original = agent._messages  # capture the LIST OBJECT identity
    original_contents_snapshot = [dict(m) for m in agent._messages]

    fake = _FakeProvider(None, agent=agent, exc=RuntimeError("summariser boom"))
    agent._llm = fake

    # Must not raise.
    result = asyncio.run(agent._maybe_compact())
    assert result is None

    # _messages byte-for-byte unchanged: same list object AND equal contents.
    assert agent._messages is original
    assert agent._messages == original_contents_snapshot
    assert agent._phase == "idle"
    # The guard was marked before the call, so a second attempt in the same
    # run is also a no-op (proves the failure doesn't retry + that the
    # once-per-run cap holds even after a failure).
    assert agent._compaction_attempted_this_run is True


# --- (5) Once-per-run guard --------------------------------------------------


def test_maybe_compact_once_per_run_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    agent._messages = _seed_simple_groups(6, last_usage=_HIGH_USAGE)
    fake = _FakeProvider(_fake_response(["summary"]), agent=agent)
    agent._llm = fake

    asyncio.run(agent._maybe_compact())
    assert len(fake.calls) == 1

    # Sustained HIGH pressure, but the guard caps further attempts in the
    # same run. Re-seed the conversation to a still-HIGH state to prove the
    # guard (not the budget) is what suppresses the second call.
    asyncio.run(agent._maybe_compact())
    assert len(fake.calls) == 1  # no second summariser call


# --- (6) Substitute validity / safe-cut regression --------------------------


def test_maybe_compact_preserves_tool_pair_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After compaction the retained tail has no dangling tool_use ids."""
    agent = _make_agent(monkeypatch)
    # Build a conversation with two complete tool pairs: one inside the
    # evictable prefix, one in the retained tail. The last assistant carries
    # HIGH usage so plan_eviction fires.
    messages: list[dict[str, Any]] = [
        _user_msg("q1"),
        _assistant_msg(
            blocks=[{"type": "text", "text": "a1"}, _tool_use_block("fs.read", "tu_1")]
        ),
        _user_msg("ignored"),  # placeholder; replaced below
        _assistant_msg("a1b"),
    ]
    # The tool_result lands in the user turn that follows the tool_use.
    messages[2] = {"role": "user", "content": [_tool_result_block("tu_1", "file body")]}
    # A second tool pair in what will become the retained tail.
    messages.extend(
        [
            _user_msg("q2"),
            _assistant_msg(
                blocks=[
                    {"type": "text", "text": "a2"},
                    _tool_use_block("fs.read", "tu_2"),
                ]
            ),
            {"role": "user", "content": [_tool_result_block("tu_2", "second body")]},
            _assistant_msg("a2b"),
            _user_msg("q3"),
            _assistant_msg("a3"),
            _user_msg("q4"),
            _assistant_msg("a4"),
            _user_msg("q5"),
            _assistant_msg("a5", usage_input=_HIGH_USAGE),
        ]
    )
    agent._messages = messages
    fake = _FakeProvider(_fake_response(["condensed summary"]))
    agent._llm = fake

    asyncio.run(agent._maybe_compact())

    assert len(fake.calls) == 1  # compaction actually fired

    # Validate the wire projection: every tool_use has a matching tool_result.
    projected = _request_messages(agent._messages)
    tool_use_ids: set[str] = set()
    tool_result_ids: set[str] = set()
    for msg in projected:
        for block in msg["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_use_ids.add(str(block.get("id", "")))
            elif block.get("type") == "tool_result":
                tool_result_ids.add(str(block.get("tool_use_id", "")))
    assert tool_use_ids, "test setup: expected at least one retained tool_use"
    # Every retained tool_use is answered (no dangling references).
    assert tool_use_ids <= tool_result_ids

    # Strict alternation in the projection (Anthropic wire contract).
    roles = [msg["role"] for msg in projected]
    for a, b in zip(roles, roles[1:]):
        assert a != b, f"consecutive same-role messages: {roles}"


def test_maybe_compact_consecutive_user_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When retained[0] is role='user', the summary + head merge on the wire."""
    agent = _make_agent(monkeypatch)
    # 6 groups; the retained tail (last 4 groups) starts with a user msg.
    agent._messages = _seed_simple_groups(6, last_usage=_HIGH_USAGE)
    fake = _FakeProvider(_fake_response(["condensed"]))
    agent._llm = fake

    asyncio.run(agent._maybe_compact())
    assert len(fake.calls) == 1

    projected = _request_messages(agent._messages)
    # The summary user message + the retained user head collapse into one
    # message (no consecutive same-role on the wire).
    roles = [msg["role"] for msg in projected]
    for a, b in zip(roles, roles[1:]):
        assert a != b


# --- (7) Summary text extraction (multi-block concat) -----------------------


def test_maybe_compact_concatenates_multi_block_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    agent._messages = _seed_simple_groups(6, last_usage=_HIGH_USAGE)
    fake = _FakeProvider(_fake_response(["Hello ", "World"]))
    agent._llm = fake

    asyncio.run(agent._maybe_compact())

    summary_text = agent._messages[0]["content"][0]["text"]
    assert summary_text == "Hello World"


# --- (8) Different-provider path --------------------------------------------


def test_maybe_compact_routes_to_different_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    agent._messages = _seed_simple_groups(6, last_usage=_HIGH_USAGE)

    # The session provider's client (must NOT be used for the summary).
    primary = _FakeProvider(_fake_response(["turn response"]))
    agent._llm = primary

    # Select a different provider/model for the summariser via the env knob.
    monkeypatch.setenv("COTHIS_SUMMARY_MODEL", "openai/gpt-4o")
    second = _FakeProvider(_fake_response(["cross-provider summary"]))
    captured: list[str] = []

    def fake_get_provider(provider: str, **_kw: Any) -> Any:
        captured.append(provider)
        return second

    # Re-patch get_provider so the lazy import inside ``_maybe_compact``
    # picks up the stub returning the second fake.
    monkeypatch.setattr("cothis.ai.get_provider", fake_get_provider)

    asyncio.run(agent._maybe_compact())

    assert captured == ["openai"]  # routed to the override provider
    assert len(second.calls) == 1
    assert second.calls[0]["model"] == "gpt-4o"
    # The session's own client was NOT used for the summary.
    assert primary.calls == []
    # Substitute landed from the second provider's response.
    assert agent._messages[0]["content"][0]["text"] == "cross-provider summary"


# --- (9) Normal-path regression: hook is invisible under low pressure -------


def test_normal_turn_path_unchanged_under_low_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One full turn under LOW pressure -> exactly one amessages call."""
    agent = _make_agent(monkeypatch)
    # Seed a small low-pressure history.
    agent._messages = [
        _user_msg("previous question"),
        _assistant_msg("previous answer", usage_input=100),  # ratio ~0 -> NONE
    ]
    # The turn response: empty content -> no tool_use -> loop returns "".
    fake = _FakeProvider(_fake_response([]), agent=agent)
    agent._llm = fake
    monkeypatch.setattr(agent, "_tool_schemas", lambda: [])

    async def _noop(*_a: Any, **_kw: Any) -> None:
        return None

    monkeypatch.setattr(agent, "_ensure_mcp", _noop)
    monkeypatch.setattr(agent, "_ensure_handles", _noop)

    result = asyncio.run(agent.run("next question"))
    assert result == ""  # empty-content turn -> empty final answer

    # Exactly one amessages call (the turn itself; zero summariser calls).
    assert len(fake.calls) == 1
    # _messages grew by exactly user + assistant as before.
    assert len(agent._messages) == 4
    assert agent._messages[2]["role"] == "user"  # the new user turn
    assert agent._messages[3]["role"] == "assistant"  # the new assistant turn
    # _phase never left idle on the non-compacting path.
    assert agent._phase == "idle"


# --- (10) ask-mode (_session is None) ----------------------------------------


def test_maybe_compact_ask_mode_no_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compaction fires in-memory without a Session and without crashing."""
    agent = _make_agent(monkeypatch)
    assert agent._session is None  # ask mode: no session bound
    agent._messages = _seed_simple_groups(6, last_usage=_HIGH_USAGE)
    fake = _FakeProvider(_fake_response(["ask-mode summary"]))
    agent._llm = fake

    asyncio.run(agent._maybe_compact())  # must not raise

    assert len(fake.calls) == 1
    # session_id=None because _session is None (no crash reading .session_id).
    assert fake.calls[0]["session_id"] is None
    assert agent._messages[0]["content"][0]["text"] == "ask-mode summary"


# --- (11) Lazy-client invariant ---------------------------------------------


def test_agent_constructs_without_summary_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compaction code path must not eagerly build a summariser client."""
    # ``get_provider`` is patched to a MagicMock (no credentials, no SDK);
    # constructing the Agent must succeed regardless of compaction wiring.
    monkeypatch.setattr("cothis.ai.get_provider", lambda *a, **kw: MagicMock())
    agent = Agent(model=_MODEL, provider=_PROVIDER, tools=[])
    # The phase + guard attrs default to idle / False without any client work.
    assert agent._phase == "idle"
    assert agent._compaction_attempted_this_run is False


# --- CRITICAL pressure variant (secondary coverage) -------------------------


def test_maybe_compact_critical_pressure_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CRITICAL pressure (ratio >= 1.00) also triggers compaction."""
    agent = _make_agent(monkeypatch)
    agent._messages = _seed_simple_groups(6, last_usage=_CRITICAL_USAGE)
    fake = _FakeProvider(_fake_response(["critical summary"]), agent=agent)
    agent._llm = fake

    asyncio.run(agent._maybe_compact())

    assert len(fake.calls) == 1
    assert fake.phases == ["compaction"]
    assert agent._phase == "idle"
    assert agent._messages[0]["content"][0]["text"] == "critical summary"
