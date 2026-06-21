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
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from any_llm import AnyLLM
    from any_llm.types.completion import ChatCompletionMessage

Tool = Callable[..., Any]
Message = dict[str, Any]
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
        self._tool_map = {tool.__name__: tool for tool in self.tools}

    async def run(self, user_input: str) -> str:
        """Run the agent loop to completion and return the final answer."""
        messages: list[CompletionInput] = self._initial_messages(user_input)

        for _turn in range(self.max_iterations):
            response = await self._llm.acompletion(
                model=self.model,
                messages=messages,
                tools=list(self.tools) or None,
            )
            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                return message.content or ""

            for tool_call in message.tool_calls:
                result = self._execute(tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        raise MaxIterationsError(
            f"Agent did not finish within {self.max_iterations} iterations."
        )

    # --- internals -----------------------------------------------------

    def _initial_messages(self, user_input: str) -> list[CompletionInput]:
        messages: list[CompletionInput] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_input})
        return messages

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
