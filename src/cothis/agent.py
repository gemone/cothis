"""A basic agent loop built on top of any-llm.

The loop is the standard ReAct-style cycle:

1. Send the conversation + tool schemas to the model.
2. If the model asks for tool calls, execute them and append the results.
3. Repeat until the model produces a message without tool calls, or
   ``max_iterations`` is reached.

Example
-------
>>> from cothis.agent import Agent
>>> agent = Agent(model="mistral-small-latest", provider="mistral")
>>> print(agent.run("What is 47 * 83?"))
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from any_llm import AnyLLM
    from any_llm.types.completion import ChatCompletionMessage

Tool = Callable[..., Any]
Message = dict[str, Any]


def _safe_parse_args(raw: str | None) -> dict[str, Any]:
    """Best-effort JSON parse for a streamed tool-call arguments string.

    The accumulated ``arguments`` from ``_assemble_tool_calls`` should be a
    complete JSON object by the time the stream ends, but providers can emit
    malformed JSON (trailing commas, truncation). On failure we fall back to
    ``{"_raw": raw}`` so the CLI still has something to show the user
    instead of crashing mid-stream.

    cothis: empty/None ``raw`` returns ``{}`` rather than the spec's
    ``{"_raw": raw}`` fallback. This lets the display format a no-arg tool
    call as ``calling fs.read()`` instead of ``calling fs.read(_raw='')``.
    The trade-off: a tool can't distinguish "provider sent empty string"
    from "provider sent no arguments" — acceptable for cothis today since
    no built-in tool has ambiguous empty-arg semantics. Upgrade path: if a
    future tool needs the distinction, gate on ``raw is None`` vs
    ``raw == ''``.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"_raw": raw}
    except json.JSONDecodeError:
        return {"_raw": raw}


@dataclass(frozen=True)
class ToolCallEvent:
    """Streamed event: the agent is about to invoke a tool.

    Yielded by ``Agent.run_stream`` before each tool dispatch so the CLI
    can surface "calling fs.read(...)" inline. ``arguments`` is the parsed
    dict (matches what will be passed to the tool); the raw JSON string is
    dropped because the parsed form is what the user wants to read.
    """

    name: str
    arguments: dict[str, Any]


# cothis: string-form type alias so the module top level does not import
# any_llm.types.completion at runtime. ChatCompletionMessage resolves only
# under static type checkers (see TYPE_CHECKING above).
# noqa: UP040 — PEP 695 `type` syntax would eagerly evaluate the RHS and
# pull any_llm back in; the string-form TypeAlias is the only form that
# stays lazy.
CompletionInput: TypeAlias = "dict[str, Any] | ChatCompletionMessage"  # noqa: UP040


class MaxIterationsError(RuntimeError):
    """Raised when the agent exhausts its iteration budget before finishing."""


class Agent(BaseModel):
    """A minimal ReAct-style agent loop over any-llm.

    Parameters
    ----------
    model:
        Model identifier, e.g. ``"mistral-small-latest"``.
    provider:
        any-llm provider key, e.g. ``"mistral"``, ``"openai"``, ``"anthropic"``.
    tools:
        Python callables the agent can invoke. Each must have a docstring
        and type annotations; any-llm converts them into the provider's
        tool schema automatically.
    system_prompt:
        Optional system message prepended to every run.
    max_iterations:
        Safety cap on the number of LLM round-trips per ``run``.
    api_key / api_base:
        Forwarded to ``AnyLLM.create``. Default to the provider's env vars.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    provider: str
    tools: Sequence[Tool] = Field(default_factory=list)
    system_prompt: str | None = None
    max_iterations: int = 10
    api_key: str | None = None
    api_base: str | None = None

    # Runtime-only state: not validated, not serialised. Built in
    # ``model_post_init`` so the pydantic model stays construction-safe.
    _llm: AnyLLM = PrivateAttr()
    _tool_map: dict[str, Tool] = PrivateAttr(default_factory=dict)
    # Conversation memory. Empty on a fresh Agent; ``_ensure_messages`` fills
    # the first turn (system + user) and appends ``user`` on every turn after.
    # ``ask`` (one-shot) never observes cross-turn state because the Agent is
    # discarded after a single ``run``. ``chat`` reuses one Agent across turns
    # so this list accumulates the whole session.
    #
    # cothis: ceiling — messages grow without bound across a long ``chat``
    # session. Token cost per turn rises linearly. No windowing/summarisation
    # is planned; Ctrl-C and start a new session when it gets slow.
    _messages: list[CompletionInput] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        # cothis: lazy-import any_llm here (not at module top level) so
        # `import cothis.agent` stays cheap. Importing any_llm eagerly pulls
        # openai + anthropic types (~1s cold-start), and we only need it
        # once the Agent is actually constructed. The matching loading
        # spinner in cli.py wraps this exact call.
        from any_llm import AnyLLM

        self._llm = AnyLLM.create(
            self.provider,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        self._tool_map = {
            tool.__name__: tool  # ty: ignore[unresolved-attribute] — same as tools._named: real tools are `def`s, the Callable alias is just too wide.
            for tool in self.tools
        }

    async def run(self, user_input: str) -> str:
        """Run the agent loop to completion and return the final answer.

        Non-streaming: the full ReAct loop runs to completion before this
        returns. Use ``run_stream`` when the caller wants the final answer
        token-by-token (e.g. ``cothis chat``).

        Side effect: appends the user message, each assistant response, and
        every tool result to ``self._messages``. This is what lets ``chat``
        reuse one Agent across turns — but it also means calling ``run``
        twice on the same instance leaks the first conversation into the
        second. ``ask`` is unaffected because it discards the Agent after a
        single call.
        """
        self._ensure_messages(user_input)

        for _turn in range(self.max_iterations):
            response = await self._llm.acompletion(
                model=self.model,
                messages=self._messages,
                tools=list(self.tools) or None,
            )
            message = response.choices[0].message
            self._messages.append(message)

            if not message.tool_calls:
                return message.content or ""

            for tool_msg in self._execute_tool_calls(message.tool_calls):
                self._messages.append(tool_msg)

        raise MaxIterationsError(
            f"Agent did not finish within {self.max_iterations} iterations."
        )

    async def run_stream(self, user_input: str) -> AsyncIterator[str | ToolCallEvent]:
        """Run the ReAct loop, yielding content deltas and tool-call events.

        Yields:
            ``str``: a content delta from the model's final answer, as soon
                as it arrives. The CLI accumulates these into a Live-rendered
                Markdown view.
            ``ToolCallEvent``: emitted immediately before each individual
                tool dispatch (not batched), so multi-tool turns surface
                "calling X" → X runs → "calling Y" → Y runs in order.

        Side effect: same as ``run`` — mutates ``self._messages``.

        cothis: optimistic yield — every ``delta.content`` the provider
        sends is yielded immediately, without waiting to see whether the
        current turn will also emit ``tool_calls``. In practice
        (OpenAI/Anthropic streaming semantics) tool-call turns emit empty
        content, so content deltas only flow on the final turn. Ceiling:
        if a provider streams content *and* tool_calls in the same turn,
        that interim text reaches the caller as if it were a final-answer
        fragment. Upgrade path: buffer content until end-of-turn, then
        yield only if no tool_calls arrived (costs one turn of latency).
        """
        self._ensure_messages(user_input)

        for _turn in range(self.max_iterations):
            response = await self._llm.acompletion(
                model=self.model,
                messages=self._messages,
                tools=list(self.tools) or None,
                stream=True,
            )

            content_parts: list[str] = []
            tool_call_chunks: list[Any] = []
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                    yield delta.content
                if delta.tool_calls:
                    tool_call_chunks.extend(delta.tool_calls)

            if not tool_call_chunks:
                # Final turn: content was streamed above. Record the full
                # assistant message so the next turn sees consistent history.
                self._messages.append(
                    {"role": "assistant", "content": "".join(content_parts)}
                )
                return

            # Intermediate tool-call turn: append the assistant message that
            # requested the tool calls, then dispatch each call individually,
            # yielding its ToolCallEvent immediately before execution so the
            # caller sees per-call ordering on multi-tool turns.
            assembled = self._assemble_tool_calls(tool_call_chunks)
            self._messages.append(
                {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assembled
                    ],
                }
            )
            for tc in assembled:
                yield ToolCallEvent(
                    name=tc.function.name,
                    arguments=_safe_parse_args(tc.function.arguments),
                )
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": self._execute(tc),
                    }
                )

        raise MaxIterationsError(
            f"Agent did not finish within {self.max_iterations} iterations."
        )

    # --- internals -----------------------------------------------------

    def _ensure_messages(self, user_input: str) -> None:
        """Populate ``self._messages`` for this turn.

        First turn: seed with system prompt (if any) and the user message.
        Subsequent turns: append the user message to the existing history.
        """
        if self._messages:
            self._messages.append({"role": "user", "content": user_input})
            return
        if self.system_prompt:
            self._messages.append({"role": "system", "content": self.system_prompt})
        self._messages.append({"role": "user", "content": user_input})

    def _execute_tool_calls(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """Dispatch every tool call and return ready-to-append ``tool`` dicts.

        Accepts any object exposing ``.id``, ``.function.name`` and
        ``.function.arguments`` — that covers both the pydantic
        ``ChatCompletionMessageToolCall`` returned by the non-stream path and
        the ``SimpleNamespace`` produced by ``_assemble_tool_calls`` for the
        stream path.
        """
        results: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": self._execute(tool_call),
                }
            )
        return results

    def _assemble_tool_calls(self, chunks: list[Any]) -> list[SimpleNamespace]:
        """Merge streamed ``ChoiceDeltaToolCall`` fragments by index.

        Streaming tool calls arrive as many fragments per logical call: the
        first usually carries ``id`` + ``function.name``; the rest carry
        ``function.arguments`` fragments that must be string-concatenated.
        Multiple parallel tool calls are distinguished by ``index``.

        Returns objects with the same attribute shape as the non-stream
        ``ChatCompletionMessageToolCall`` (``.id``, ``.function.name``,
        ``.function.arguments``) so ``_execute_tool_calls`` can consume both.
        """
        by_index: dict[int, dict[str, str | None]] = {}
        for tc in chunks:
            entry = by_index.setdefault(
                tc.index,
                {"id": None, "name": None, "arguments": ""},
            )
            if tc.id:
                entry["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    entry["name"] = tc.function.name
                if tc.function.arguments:
                    entry["arguments"] += tc.function.arguments
        return [
            SimpleNamespace(
                id=by_index[i]["id"],
                function=SimpleNamespace(
                    name=by_index[i]["name"],
                    arguments=by_index[i]["arguments"],
                ),
            )
            for i in sorted(by_index)
        ]

    def _execute(self, tool_call: Any) -> str:
        """Dispatch a single tool call and return its result as a string."""
        name = tool_call.function.name
        raw_args = tool_call.function.arguments or "{}"
        try:
            args: dict[str, Any] = json.loads(raw_args)
        except json.JSONDecodeError:
            return f"Error: could not parse tool arguments for {name!r}: {raw_args}"

        tool = self._tool_map.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}."

        try:
            result = tool(**args)
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
            return f"Error calling {name}: {exc}"
        return str(result)
