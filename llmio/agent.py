import asyncio
import pprint
from typing import Callable, Generic, Type, Any, AsyncIterator, TypeVar
from dataclasses import dataclass
import textwrap
from inspect import signature, iscoroutinefunction
import re

from typing_extensions import assert_never
import pydantic

from llmio import function_parser, errors, types as T, models
from llmio.clients import BaseClient, AsyncOpenAI


_Context = TypeVar("_Context")

_CONTEXT_ARG_NAME = "_context"


@dataclass
class AgentResponse:
    messages: list[str]
    history: list[T.Message]


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
            result = await self.function(**params.model_dump(), **kwargs)
        else:
            result = self.function(**params.model_dump(), **kwargs)

        return str(result)

    def parse_args(self, args: str) -> pydantic.BaseModel:
        """
        Parses the arguments received from the OpenAI API using the Pydantic model.
        """
        return self.params.model_validate_json(args)

    @property
    def function_definition(self) -> T.FunctionDefinition:
        """
        Returns the tool schema that is sent to the OpenAI API.
        """
        schema = self.params.model_json_schema()

        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)

        if self.strict:
            schema["additionalProperties"] = False
        definition = T.FunctionDefinition(
            name=self.name,
            description=self.description,
            parameters=schema,
        )
        if self.strict:
            definition["strict"] = True

        return definition


_ResponseFormatT = TypeVar("_ResponseFormatT", bound=pydantic.BaseModel)


class StructuredAgentResponse(Generic[_ResponseFormatT]):
    messages: list[_ResponseFormatT]
    history: list[T.Message]

    def __init__(self, messages: list[_ResponseFormatT], history: list[T.Message]):
        self.messages = messages
        self.history = history


class BaseAgent:
    def __init__(
        self,
        instruction: str,
        client: BaseClient | AsyncOpenAI,
        model: str = "gpt-4o-mini",
        graceful_errors: bool = False,
    ):
        """
        Initializes the agent with an instruction, OpenAI client, and model.

        Args:
            instruction: The instruction that the agent will follow.
            client: The OpenAI client to use for API requests.
            model: The model to use for completions.
            graceful_errors: Whether the agent should handle invalid parameters / tool calls
                                and try to continue the interaction.
                             If set to False, the agent will raise an exception when an
                                uninterpretable tool call is returned.
                             If set to True, the agent will try to explain the error
                                to the model and continue the interaction.
        """
        self._model = model
        self._raw_instruction = textwrap.dedent(instruction).strip()
        if isinstance(client, AsyncOpenAI):
            # Backward compatibility
            self._client = BaseClient(client=client)
        else:
            self._client = client
        self._instruction = textwrap.dedent(instruction).strip()

        self._graceful_errors = graceful_errors

        self._tools: list[_Tool] = []
        self._variables: dict[str, Callable] = {}

        self._prompt_inspectors: list[Callable] = []
        self._output_inspectors: list[Callable] = []
        self._message_callbacks: list[Callable] = []

    async def _execute_variable(
        self, variable_name: str, context: _Context | None
    ) -> Any:
        """
        Executes a variable function by name.
        """
        variable_function = self._variables[variable_name]

        kwargs = {}
        if _CONTEXT_ARG_NAME in signature(variable_function).parameters:
            kwargs[_CONTEXT_ARG_NAME] = context

        return (
            await variable_function(**kwargs)
            if iscoroutinefunction(variable_function)
            else variable_function(**kwargs)
        )

    async def _get_instruction(self, context: _Context | None) -> str:
        """
        Returns the agent's instruction with variables replaced.
        """
        variables = re.findall(r"\{(\w+)\}", self._raw_instruction)
        for variable in variables:
            if variable not in self._variables:
                raise errors.MissingVariable(f"Variable '{variable}' is not defined.")

        variable_values = {
            variable: await self._execute_variable(variable, context)
            for variable in variables
        }
        instruction = self._raw_instruction.format(**variable_values)
        return instruction

    async def _get_system_prompt(self, context: _Context | None) -> T.SystemMessage:
        return self._create_system_message(await self._get_instruction(context))

    def summary(self) -> str:
        """
        Returns a summary of the agent's tools and their schemas.
        """
        lines = ["Tools:"]
        for tool in self._tools:
            lines.append(f"  - {tool.name}")
            lines.append("    Schema:")
            lines.append(
                textwrap.indent(pprint.pformat(tool.function_definition), "      ")
            )
            lines.append("")
        return "\n".join(lines)

    def tool(
        self, tool_function: Callable | None = None, strict: bool = False
    ) -> Callable:
        """
        Decorator to define a tool function.
        """

        def decorator(function: Callable) -> Callable:
            self._tools.append(
                _Tool(function=function, strict=strict),
            )
            return function

        if tool_function is not None:
            return decorator(tool_function)

        return decorator

    def variable(self, function: Callable) -> Callable:
        """
        Decorator to define a variable function.
        """
        self._variables[function.__name__] = function
        return function

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
        self, prompt: list[T.Message], context: _Context | None
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
        self, content: T.AssistantMessage, context: _Context | None
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

    def _parse_message_inspector_content(self, message: str) -> Any:
        """
        Parses the message content for the message inspectors.
        """
        return message

    async def _run_message_inspectors(
        self, content: str, context: _Context | None
    ) -> None:
        """
        Runs all message callbacks with the generated message content.
        """
        for callback in self._message_callbacks:
            kwargs: dict[str, str | _Context | None] = {
                "message": self._parse_message_inspector_content(content),
            }
            if _CONTEXT_ARG_NAME in signature(callback).parameters:
                kwargs[_CONTEXT_ARG_NAME] = context

            if iscoroutinefunction(callback):
                await callback(**kwargs)
            else:
                callback(**kwargs)

    @staticmethod
    def _parse_completion(
        completion: models.ChatCompletionMessage,
    ) -> T.AssistantMessage:
        """
        Parses the completion received from the OpenAI API into a Message TypedDict.
        """
        result = T.AssistantMessage(
            role=completion.role,
            content=completion.content,
        )
        if completion.tool_calls:
            result["tool_calls"] = [
                T.ToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function=T.ToolCallFunction(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
                for tool_call in completion.tool_calls
            ]
        return result

    @property
    def _tool_definitions(self) -> list[T.Tool]:
        return [
            T.Tool(
                function=tool.function_definition,
                type="function",
            )
            for tool in self._tools
        ]

    @property
    def response_format(self) -> dict[str, Any] | None:
        return None

    async def _get_completion(
        self,
        messages: list[T.Message],
    ) -> models.ChatCompletion:
        """
        Sends the prompt to the OpenAI API and returns the completion.
        """
        return await self._client.get_chat_completion(
            model=self._model,
            messages=messages,
            tools=self._tool_definitions,
            response_format=self.response_format,
        )

    def _create_user_message(self, message: str) -> T.UserMessage:
        return T.UserMessage(
            role="user",
            content=message,
        )

    def _create_tool_message(self, tool_call_id: str, content: str) -> T.ToolMessage:
        return T.ToolMessage(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
        )

    def _create_system_message(self, message: str) -> T.SystemMessage:
        return T.SystemMessage(
            role="system",
            content=message,
        )

    async def _speak(
        self,
        message: str,
        history: list[T.Message] | None = None,
        _context: _Context | None = None,
    ) -> AgentResponse:
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
        return AgentResponse(messages=new_messages, history=history)

    def _get_tool_by_name(self, name: str) -> _Tool:
        for tool in self._tools:
            if tool.name == name:
                return tool
        raise ValueError(f"No tool with the name '{name}' found.")

    async def _iterate(
        self,
        history: list[T.Message],
        context: _Context | None,
        system_message: T.SystemMessage | None = None,
    ) -> AsyncIterator[tuple[str, list[T.Message]]]:
        """
        The main loop that sends the prompt to the OpenAI API and processes the response.
        """
        system_message = system_message or await self._get_system_prompt(context)
        prompt = [
            system_message,
            *history,
        ]
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
                error_message: str
                if not self._graceful_errors:
                    match e:
                        case pydantic.ValidationError():
                            raise errors.BadToolCall(
                                f"Invalid tool call name '{tool_call.function.name}' received."
                            ) from e
                        case ValueError():
                            raise errors.BadToolCall(
                                f"Invalid tool call arguments '{tool_call.function.arguments}' received."
                            ) from e
                        case _:
                            assert_never(e)

                error_message = (
                    f"The argument validation failed for the function call to {tool.name}: {e}"
                    if isinstance(e, pydantic.ValidationError)
                    else str(e)
                )

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
            system_message=system_message,
        ):
            yield ans, hist


class Agent(BaseAgent):
    async def speak(
        self,
        message: str,
        history: list[T.Message] | None = None,
        _context: _Context | None = None,
    ) -> AgentResponse:
        return await self._speak(message, history=history, _context=_context)


class StructuredAgent(BaseAgent, Generic[_ResponseFormatT]):
    def __init__(
        self,
        instruction: str,
        client: BaseClient | AsyncOpenAI,
        response_format: Type[_ResponseFormatT],
        model: str = "gpt-4o-mini",
        graceful_errors: bool = False,
    ):
        super().__init__(
            instruction=instruction,
            client=client,
            model=model,
            graceful_errors=graceful_errors,
        )
        self._response_format = response_format

    async def speak(
        self,
        message: str,
        history: list[T.Message] | None = None,
        _context: _Context | None = None,
    ) -> StructuredAgentResponse[_ResponseFormatT]:
        assert self._response_format is not None
        response = await self._speak(message, history=history, _context=_context)
        parsed_messages = [
            self._response_format.model_validate_json(message)
            for message in response.messages
        ]
        return StructuredAgentResponse(
            messages=parsed_messages,
            history=response.history,
        )

    @property
    def response_format(self) -> dict[str, Any]:
        schema = self._response_format.model_json_schema()
        schema["additionalProperties"] = False
        return {
            "type": "json_schema",
            "json_schema": {
                "schema": schema,
                "name": self._response_format.__name__,
                "strict": True,
            },
        }

    def _parse_message_inspector_content(self, message: str) -> _ResponseFormatT:
        return self._response_format.model_validate_json(message)
