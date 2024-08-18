from typing import Callable, Type, Any, AsyncIterator
from dataclasses import dataclass
import textwrap
from inspect import signature, iscoroutinefunction

import pydantic
import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function

from llmio import function_parser


@dataclass
class _Command:
    function: Callable
    strict: bool = False

    @property
    def name(self) -> str:
        return self.function.__name__

    @property
    def params(self) -> Type[pydantic.BaseModel]:
        return function_parser.model_from_function(self.function)

    @property
    def returns(self) -> Type[pydantic.BaseModel]:
        annotation = self.function.__annotations__["return"]
        return pydantic.create_model("Result", result=(annotation, ...))

    @property
    def description(self) -> str:
        if self.function.__doc__ is None:
            return ""
        return textwrap.dedent(self.function.__doc__).strip()

    async def execute(self, params: pydantic.BaseModel, state=None):
        kwargs = {}
        if "state" in signature(self.function).parameters:
            kwargs["state"] = state

        if iscoroutinefunction(self.function):
            result = await self.function(**params.dict(), **kwargs)
        else:
            result = self.function(**params.dict(), **kwargs)

        return self.returns(result=result)

    @property
    def tool_definition(self) -> dict:
        schema = self.params.schema()
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
        self.model = model
        self.client = client

        self.instruction = textwrap.dedent(instruction).strip()

        self.commands: list[_Command] = []
        self._prompt_inspectors: list[Callable] = []
        self._output_inspectors: list[Callable] = []

    def _get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return self._create_system_message(self.instruction)

    def _create_prompt(
        self, message_history: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        return [
            self._get_system_prompt(),
            *message_history,
        ]

    def command(self, strict: bool = False) -> Callable:
        def decorator(function: Callable) -> Callable:
            assert "return" in function.__annotations__
            self.commands.append(
                _Command(function=function, strict=strict),
            )
            return function

        return decorator

    def inspect_prompt(self, function: Callable) -> Callable:
        self._prompt_inspectors.append(function)
        return function

    def inspect_output(self, function: Callable) -> Callable:
        self._output_inspectors.append(function)
        return function

    def _run_prompt_inspectors(
        self, prompt: list[ChatCompletionMessageParam], state: Any
    ) -> None:
        for inspector in self._prompt_inspectors:
            kwargs = {}
            if "state" in signature(inspector).parameters:
                kwargs["state"] = state
            inspector(prompt, **kwargs)

    def _run_output_inspectors(self, content: Any, state: Any) -> None:
        for inspector in self._output_inspectors:
            kwargs = {}
            if "state" in signature(inspector).parameters:
                kwargs["state"] = state
            inspector(content, **kwargs)

    @staticmethod
    def _parse_completion(
        completion: ChatCompletionMessage,
    ) -> ChatCompletionAssistantMessageParam:
        result = ChatCompletionAssistantMessageParam(
            {
                "role": completion.role,
                "content": completion.content,
            }
        )
        if completion.tool_calls:
            result["tool_calls"] = [
                ChatCompletionMessageToolCallParam(
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
            {"type": "function", "function": command.tool_definition}
            for command in self.commands
        ]:
            kwargs["tools"] = tool_definitions
        return kwargs

    async def _get_completion(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._get_tool_kwargs(),
        )

    def _create_user_message(self, message: str) -> ChatCompletionUserMessageParam:
        return ChatCompletionUserMessageParam(
            {
                "role": "user",
                "content": message,
            }
        )

    def _create_function_message(
        self, tool_call_id: str, content: str
    ) -> ChatCompletionToolMessageParam:
        return ChatCompletionToolMessageParam(
            {
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id,
            }
        )

    def _create_system_message(self, message: str) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            {
                "role": "system",
                "content": message,
            }
        )

    async def speak(
        self,
        message: str,
        history: list[ChatCompletionMessageParam] | None = None,
        state: Any = None,
    ) -> AsyncIterator[tuple[str, list[ChatCompletionMessageParam]]]:
        if not history:
            history = []
        history.append(self._create_user_message(message))
        async for ans, hist in self._iterate(
            history=history,
            state=state,
        ):
            yield ans, hist
        return

    async def _iterate(
        self,
        history: list[ChatCompletionMessageParam],
        state: Any = None,
    ) -> AsyncIterator[tuple[str, list[ChatCompletionMessageParam]]]:
        prompt = self._create_prompt(history)
        self._run_prompt_inspectors(prompt, state)

        result = await self._get_completion(
            messages=prompt,
        )
        generated_message = result.choices[0].message
        self._run_output_inspectors(generated_message, state)

        history.append(self._parse_completion(generated_message))

        if generated_message.content:
            yield generated_message.content, history

        if not generated_message.tool_calls:
            return

        for tool_call in generated_message.tool_calls:
            command = [
                cmd for cmd in self.commands if cmd.name == tool_call.function.name
            ][0]

            try:
                params = command.params.parse_raw(tool_call.function.arguments)
            except pydantic.ValidationError as e:
                error_message = (
                    f"The argument validation failed for the function call to {command.name}: "
                    + str(e)
                )
                history.append(
                    self._create_function_message(
                        tool_call_id=tool_call.id,
                        content=error_message,
                    )
                )
                continue

            result = await command.execute(params, state=state)

            history.append(
                self._create_function_message(
                    tool_call_id=tool_call.id,
                    content=result.json(),
                )
            )

        async for ans, hist in self._iterate(
            history=history,
            state=state,
        ):
            yield ans, hist
