import asyncio
import pprint
from typing import Callable, Type, Any, AsyncIterator, TypeVar
from dataclasses import dataclass
import textwrap
from inspect import signature, iscoroutinefunction

from typing_extensions import assert_never
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


_Context = TypeVar("_Context")

_CONTEXT_ARG_NAME = "_context"


@dataclass
class _Tool:
    function: Callable
    strict: bool = False

    @property
    def name(self) -> str:
        return self.function.__name__

    @property
    def params(self) -> Type[pydantic.BaseModel]:
        """
        Returns a Pydantic model dynamically created from the function signature.
        The pydantic model is used both to validate input arguments
        and to generate the function schema that is sent to the OpenAI API.
        """
        return function_parser.model_from_function(self.function)

    @property
    def description(self) -> str:
        """
        Returns the function's docstring, which is used as the tool's description
        in the function schema sent to the OpenAI API.
        """
        if self.function.__doc__ is None:
            return ""
        return textwrap.dedent(self.function.__doc__).strip()

    async def execute(
        self, params: pydantic.BaseModel, context: _Context | None
    ) -> str:
        """
        Executes the tool with the parsed parameters received from the OpenAI API.
        If the function is a coroutine, it is awaited.
        """
        kwargs = {}
        if _CONTEXT_ARG_NAME in signature(self.function).parameters:
            kwargs[_CONTEXT_ARG_NAME] = context

        if iscoroutinefunction(self.function):
            result = await self.function(**params.dict(), **kwargs)
        else:
            result = self.function(**params.dict(), **kwargs)

        return str(result)

    def parse_args(self, args: str) -> pydantic.BaseModel:
        """
        Parses the arguments received from the OpenAI API using the Pydantic model.
        """
        return self.params.parse_raw(args)

    @property
    def tool_definition(self) -> dict:
        """
        Returns the tool schema that is sent to the OpenAI API.
        """
        schema = self.params.schema()
        if self.strict:
            schema["additionalProperties"] = False
        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema,
            "strict": self.strict,
        }


class Agent:
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
        """
        Creates a prompt by combining the system instruction with the message history.
        """
        return [
            self._get_system_prompt(),
            *message_history,
        ]

    def summary(self) -> str:
        """
        Returns a summary of the agent's tools and their schemas.
        """
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
        """
        Decorator to define a tool function.
        """

        def decorator(function: Callable) -> Callable:
            self._tools.append(
                _Tool(function=function, strict=strict),
            )
            return function

        return decorator

    def inspect_prompt(self, function: Callable) -> Callable:
        """
        Decorator to define a prompt inspector.
        The prompt inspector is called with the full prompt.
        """
        self._prompt_inspectors.append(function)
        return function

    def inspect_output(self, function: Callable) -> Callable:
        """
        Decorator to define an output inspector.
        The output inspector is called with the full generated message, including tool calls.
        """
        self._output_inspectors.append(function)
        return function

    def on_message(self, function: Callable) -> Callable:
        """
        Decorator to define a message callback.
        """
        params = set(signature(function).parameters.keys())
        if params not in [
            {"message"},
            {_CONTEXT_ARG_NAME, "message"},
        ]:
            raise ValueError(
                "The message inspector must accept only 'message' or '_context, message' as arguments."
            )
        self._message_callbacks.append(function)
        return function

    async def _run_prompt_inspectors(
        self, prompt: list[Message], context: _Context | None
    ) -> None:
        """
        Runs all prompt inspectors with the full prompt prior to sending it to the OpenAI API.
        """
        for inspector in self._prompt_inspectors:
            kwargs = {}
            if _CONTEXT_ARG_NAME in signature(inspector).parameters:
                kwargs[_CONTEXT_ARG_NAME] = context
            if iscoroutinefunction(inspector):
                await inspector(prompt, **kwargs)
            else:
                inspector(prompt, **kwargs)

    async def _run_output_inspectors(
        self, content: AssistantMessage, context: _Context | None
    ) -> None:
        """
        Runs all output inspectors with the full generated message, including tool calls.
        """
        for inspector in self._output_inspectors:
            kwargs = {}
            if _CONTEXT_ARG_NAME in signature(inspector).parameters:
                kwargs[_CONTEXT_ARG_NAME] = context
            if iscoroutinefunction(inspector):
                await inspector(content, **kwargs)
            else:
                inspector(content, **kwargs)

    async def _run_message_inspectors(
        self, content: str, context: _Context | None
    ) -> None:
        """
        Runs all message callbacks with the generated message content.
        """
        for callback in self._message_callbacks:
            kwargs: dict[str, str | _Context | None] = {
                "message": content,
            }
            if _CONTEXT_ARG_NAME in signature(callback).parameters:
                kwargs[_CONTEXT_ARG_NAME] = context

            if iscoroutinefunction(callback):
                await callback(**kwargs)
            else:
                callback(**kwargs)

    @staticmethod
    def _parse_completion(
        completion: ChatCompletionMessage,
    ) -> AssistantMessage:
        """
        Parses the completion received from the OpenAI API into a Message TypedDict.
        """
        result = AssistantMessage(
            role=completion.role,
            content=completion.content,
        )
        if completion.tool_calls:
            result["tool_calls"] = [
                ToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function=Function(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
                for tool_call in completion.tool_calls
            ]
        return result

    def _get_tool_kwargs(self) -> dict[str, Any]:
        """
        Returns the tools schema that is sent to the OpenAI API.
        Built dynamically because the API does not accept empty lists.
        """
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
        """
        Sends the prompt to the OpenAI API and returns the completion.
        """
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **self._get_tool_kwargs(),
        )

    def _create_user_message(self, message: str) -> UserMessage:
        return UserMessage(
            role="user",
            content=message,
        )

    def _create_tool_message(self, tool_call_id: str, content: str) -> ToolMessage:
        return ToolMessage(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
        )

    def _create_system_message(self, message: str) -> SystemMessage:
        return SystemMessage(
            role="system",
            content=message,
        )

    async def speak(
        self,
        message: str,
        history: list[Message] | None = None,
        _context: _Context | None = None,
    ) -> tuple[list[str], list[Message]]:
        """
        A full interaction loop with the agent.
        If tool calls are present in the completion, they are executed, and the loop continues.
        """
        if not history:
            history = []
        else:
            history = history.copy()
        history.append(self._create_user_message(message))

        new_messages: list[str] = []

        async for message, history in self._iterate(history=history, context=_context):
            new_messages.append(message)
        return new_messages, history

    def _get_tool_by_name(self, name: str) -> _Tool:
        for tool in self._tools:
            if tool.name == name:
                return tool
        raise ValueError(f"No tool with the name '{name}' found.")

    async def _iterate(
        self,
        history: list[Message],
        context: _Context | None,
    ) -> AsyncIterator[tuple[str, list[Message]]]:
        """
        The main loop that sends the prompt to the OpenAI API and processes the response.
        """
        prompt = self._create_prompt(history)
        await self._run_prompt_inspectors(prompt, context)

        completion = await self._get_completion(
            messages=prompt,
        )
        generated_message = completion.choices[0].message
        parsed_response = self._parse_completion(generated_message)
        await self._run_output_inspectors(parsed_response, context)

        history.append(parsed_response)

        if generated_message.content:
            await self._run_message_inspectors(generated_message.content, context)
            yield generated_message.content, history

        if not generated_message.tool_calls:
            return

        awaitables = []
        awaited_tool_calls = []
        for tool_call in generated_message.tool_calls:
            try:
                tool = self._get_tool_by_name(tool_call.function.name)
                params = tool.parse_args(tool_call.function.arguments)
            except (ValueError, pydantic.ValidationError) as e:
                match e:
                    case pydantic.ValidationError():
                        error_message = (
                            f"The argument validation failed for the function call to {tool.name}: "
                            + str(e)
                        )
                    case ValueError():
                        error_message = str(e)
                    case _:
                        assert_never(e)

                history.append(
                    self._create_tool_message(
                        tool_call_id=tool_call.id,
                        content=error_message,
                    )
                )
                continue

            awaitables.append(tool.execute(params, context=context))
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
            context=context,
        ):
            yield ans, hist
