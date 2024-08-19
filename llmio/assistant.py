import asyncio
import pprint
from typing import Callable, Type, Any, AsyncIterator, TypeVar
from dataclasses import dataclass
import textwrap
from inspect import signature, iscoroutinefunction

import pydantic
import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam as Message,
    ChatCompletionToolMessageParam as ToolMessage,
    ChatCompletionAssistantMessageParam as AssistantMessage,
    ChatCompletionUserMessageParam as UserMessage,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionMessageToolCallParam as ToolCall,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function

from llmio import function_parser


State = TypeVar("State")

_STATE_ARG_NAME = "_state"


@dataclass
class _Tool:
    function: Callable
    strict: bool = False

    @property
    def name(self) -> str:
        return self.function.__name__

    @property
    def params(self) -> Type[pydantic.BaseModel]:
        return function_parser.model_from_function(self.function)

    @property
    def description(self) -> str:
        if self.function.__doc__ is None:
            return ""
        return textwrap.dedent(self.function.__doc__).strip()

    async def execute(self, params: pydantic.BaseModel, state: State | None) -> str:
        kwargs = {}
        if _STATE_ARG_NAME in signature(self.function).parameters:
            kwargs[_STATE_ARG_NAME] = state

        if iscoroutinefunction(self.function):
            result = await self.function(**params.dict(), **kwargs)
        else:
            result = self.function(**params.dict(), **kwargs)

        return str(result)

    @property
    def tool_definition(self) -> dict:
        schema = self.params.schema()
        if self.strict:
            schema["additionalProperties"] = False
        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema,
            "strict": self.strict,
        }


class Assistant:
    def __init__(
        self,
        instruction: str,
        client: openai.AsyncOpenAI,
        model: str = "gpt-4o-mini",
    ):
        self._model = model
        self._client = client

        self._instruction = textwrap.dedent(instruction).strip()

        self._tools: list[_Tool] = []
        self._prompt_inspectors: list[Callable] = []
        self._output_inspectors: list[Callable] = []
        self._message_callbacks: list[Callable] = []

    def _get_system_prompt(self) -> SystemMessage:
        return self._create_system_message(self._instruction)

    def _create_prompt(self, message_history: list[Message]) -> list[Message]:
        return [
            self._get_system_prompt(),
            *message_history,
        ]

    def summary(self) -> str:
        lines = ["Tools:"]
        for tool in self._tools:
            lines.append(f"  - {tool.name}")
            lines.append("    Schema:")
            lines.append(
                textwrap.indent(pprint.pformat(tool.tool_definition), "      ")
            )
            lines.append("")
        return "\n".join(lines)

    def tool(self, strict: bool = False) -> Callable:
        def decorator(function: Callable) -> Callable:
            self._tools.append(
                _Tool(function=function, strict=strict),
            )
            return function

        return decorator

    def inspect_prompt(self, function: Callable) -> Callable:
        self._prompt_inspectors.append(function)
        return function

    def inspect_output(self, function: Callable) -> Callable:
        self._output_inspectors.append(function)
        return function

    def on_message(self, function: Callable) -> Callable:
        params = set(signature(function).parameters.keys())
        if params not in [
            {"message"},
            {_STATE_ARG_NAME, "message"},
        ]:
            raise ValueError(
                "The message inspector must accept only 'message' or '_state, message' as arguments."
            )
        self._message_callbacks.append(function)
        return function

    async def _run_prompt_inspectors(
        self, prompt: list[Message], state: State | None
    ) -> None:
        for inspector in self._prompt_inspectors:
            kwargs = {}
            if _STATE_ARG_NAME in signature(inspector).parameters:
                kwargs[_STATE_ARG_NAME] = state
            if iscoroutinefunction(inspector):
                await inspector(prompt, **kwargs)
            else:
                inspector(prompt, **kwargs)

    async def _run_output_inspectors(
        self, content: AssistantMessage, state: State | None
    ) -> None:
        for inspector in self._output_inspectors:
            kwargs = {}
            if _STATE_ARG_NAME in signature(inspector).parameters:
                kwargs[_STATE_ARG_NAME] = state
            inspector(content, **kwargs)
            if iscoroutinefunction(inspector):
                await inspector(content, **kwargs)
            else:
                inspector(content, **kwargs)

    async def _run_message_inspectors(self, content: str, state: State | None) -> None:
        for callback in self._message_callbacks:
            kwargs: dict[str, str | State | None] = {
                "message": content,
            }
            if _STATE_ARG_NAME in signature(callback).parameters:
                kwargs[_STATE_ARG_NAME] = state

            if iscoroutinefunction(callback):
                await callback(**kwargs)
            else:
                callback(**kwargs)

    @staticmethod
    def _parse_completion(
        completion: ChatCompletionMessage,
    ) -> AssistantMessage:
        result = AssistantMessage(
            {
                "role": completion.role,
                "content": completion.content,
            }
        )
        if completion.tool_calls:
            result["tool_calls"] = [
                ToolCall(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": Function(
                            {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            }
                        ),
                    }
                )
                for tool_call in completion.tool_calls
            ]
        return result

    def _get_tool_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if tool_definitions := [
            {"type": "function", "function": tool.tool_definition}
            for tool in self._tools
        ]:
            kwargs["tools"] = tool_definitions
        return kwargs

    async def _get_completion(
        self,
        messages: list[Message],
    ) -> ChatCompletion:
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **self._get_tool_kwargs(),
        )

    def _create_user_message(self, message: str) -> UserMessage:
        return UserMessage(
            {
                "role": "user",
                "content": message,
            }
        )

    def _create_tool_message(self, tool_call_id: str, content: str) -> ToolMessage:
        return ToolMessage(
            {
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id,
            }
        )

    def _create_system_message(self, message: str) -> SystemMessage:
        return SystemMessage(
            {
                "role": "system",
                "content": message,
            }
        )

    async def run(
        self,
        message: str,
        history: list[Message] | None = None,
        _state: State | None = None,
    ) -> list[Message]:
        async for _, history in self.speak(message, history=history, _state=_state):
            pass
        return history

    async def speak(
        self,
        message: str,
        history: list[Message] | None = None,
        _state: State | None = None,
    ) -> AsyncIterator[tuple[str, list[Message]]]:
        if not history:
            history = []
        else:
            history = history.copy()
        history.append(self._create_user_message(message))
        async for ans, hist in self._iterate(
            history=history,
            state=_state,
        ):
            yield ans, hist
        return

    async def _iterate(
        self,
        history: list[Message],
        state: State | None,
    ) -> AsyncIterator[tuple[str, list[Message]]]:
        prompt = self._create_prompt(history)
        await self._run_prompt_inspectors(prompt, state)

        completion = await self._get_completion(
            messages=prompt,
        )
        generated_message = completion.choices[0].message
        parsed_response = self._parse_completion(generated_message)
        await self._run_output_inspectors(parsed_response, state)

        history.append(parsed_response)

        if generated_message.content:
            await self._run_message_inspectors(generated_message.content, state)
            yield generated_message.content, history

        if not generated_message.tool_calls:
            return

        awaitables = []
        awaited_tool_calls = []
        for tool_call in generated_message.tool_calls:
            tool = [cmd for cmd in self._tools if cmd.name == tool_call.function.name][
                0
            ]

            try:
                params = tool.params.parse_raw(tool_call.function.arguments)
            except pydantic.ValidationError as e:
                error_message = (
                    f"The argument validation failed for the function call to {tool.name}: "
                    + str(e)
                )
                history.append(
                    self._create_tool_message(
                        tool_call_id=tool_call.id,
                        content=error_message,
                    )
                )
                continue

            awaitables.append(tool.execute(params, state=state))
            awaited_tool_calls.append(tool_call)

        tool_results = await asyncio.gather(*awaitables)
        for tool_call, tool_result in zip(awaited_tool_calls, tool_results):
            history.append(
                self._create_tool_message(
                    tool_call_id=tool_call.id,
                    content=tool_result,
                )
            )

        async for ans, hist in self._iterate(
            history=history,
            state=state,
        ):
            yield ans, hist
