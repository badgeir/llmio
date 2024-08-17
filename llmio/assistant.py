from typing import Literal, Optional, Callable, Type, Any, AsyncIterator
from dataclasses import dataclass
import textwrap
from inspect import signature, iscoroutinefunction

import pydantic
import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import FunctionCall

from llmio import model


class CommandModel(pydantic.BaseModel):
    command: str
    params: Any


@dataclass
class Command:
    function: Callable

    @property
    def name(self) -> str:
        return self.function.__name__

    @property
    def params(self) -> Type[pydantic.BaseModel]:
        return model.model_from_function(self.function)

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
    def function_definition(self) -> dict:
        schema = self.params.schema()
        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema,
        }


class Assistant:
    def __init__(
        self,
        key: str,
        description: str,
        engine: str = "gpt-4",
        debug: bool = False,
    ):
        self.engine = engine
        self.description = textwrap.dedent(description).strip()
        self.commands: list[Command] = []
        self.debug = debug

        self._prompt_inspectors: list[Callable] = []
        self._output_inspectors: list[Callable] = []

        self.client = openai.AsyncOpenAI(api_key=key)

    def system_prompt(self) -> str:
        return self.description

    def _get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            {"role": "system", "content": self.system_prompt()}
        )

    def create_prompt(
        self, message_history: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        return [
            self._get_system_prompt(),
            *message_history,
        ]

    def command(self, function: Callable) -> Callable:
        assert "return" in function.__annotations__
        self.commands.append(
            Command(function=function),
        )
        return function

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

    def log(self, *message: Any) -> None:
        if self.debug:
            print(*message)

    @staticmethod
    def parse_completion(
        completion: ChatCompletionMessage,
    ) -> ChatCompletionMessageParam:
        match completion.role:
            case "user":
                return ChatCompletionUserMessageParam(
                    {
                        "role": completion.role,
                        "content": completion.message,
                    }
                )
            case "assistant":
                return ChatCompletionAssistantMessageParam(
                    {
                        "role": completion.role,
                        "content": completion.content,
                        "function_call": (
                            FunctionCall(
                                name=completion.function_call.name,
                                arguments=completion.function_call.arguments,
                            )
                            if completion.function_call
                            else None
                        ),
                    }
                )
            case "function":
                return ChatCompletionFunctionMessageParam(
                    {
                        "role": completion.role,
                        "content": completion.content,
                        "name": completion.name,
                    }
                )
            case _:
                raise ValueError(completion.role)

    def create_history(
        self,
        history: list[ChatCompletionMessageParam] | None,
        message: str | None,
        role: str,
        function_name: str | None,
    ) -> list[ChatCompletionMessageParam]:
        if history is None:
            history = []
        history = history[:]

        if message:
            new_message: ChatCompletionMessageParam
            match role:
                case "user":
                    new_message = ChatCompletionUserMessageParam(
                        {
                            "role": role,
                            "content": message,
                        }
                    )
                case "assistant":
                    new_message = ChatCompletionAssistantMessageParam(
                        {
                            "role": role,
                            "content": message,
                        }
                    )
                case "function":
                    assert function_name is not None
                    new_message = ChatCompletionFunctionMessageParam(
                        {
                            "role": role,
                            "content": message,
                            "name": function_name,
                        }
                    )
                case _:
                    raise ValueError(role)

            history.append(new_message)
        return history

    def get_function_kwargs(self) -> dict[str, Any]:
        function_definitions = [
            command.function_definition for command in self.commands
        ]

        kwargs: dict[str, Any] = {}
        if function_definitions:
            kwargs["functions"] = function_definitions
        return kwargs

    async def get_completion(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        return await self.client.chat.completions.create(
            model=self.engine,
            messages=messages,
            **self.get_function_kwargs(),
        )

    async def speak(
        self,
        message: str | None = None,
        history: Optional[list[ChatCompletionMessageParam]] = None,
        state: Any = None,
        role: Literal["user", "system", "function"] = "user",
        function_name: Optional[str] = None,
        retries=0,
    ) -> AsyncIterator[tuple[str, list[ChatCompletionMessageParam]]]:
        history = self.create_history(
            history,
            message,
            role,
            function_name,
        )

        prompt = self.create_prompt(history)
        self._run_prompt_inspectors(prompt, state)

        result = await self.get_completion(
            messages=prompt,
        )
        generated_message = result.choices[0].message
        self.log("Model output:", generated_message)
        self._run_output_inspectors(generated_message, state)

        new_message: ChatCompletionMessageParam = self.parse_completion(
            generated_message
        )
        history.append(new_message)

        if generated_message.content:
            yield generated_message.content, history

        if function_call := generated_message.function_call:
            function_name = function_call.name
            arguments = function_call.arguments

            command = [cmd for cmd in self.commands if cmd.name == function_name][0]

            try:
                params = command.params.parse_raw(arguments)
            except pydantic.ValidationError as e:
                if retries > 1:
                    raise e

                error_message = (
                    f"The argument validation failed for the function call to {command.name}: "
                    + str(e)
                )
                async for ans, hist in self.speak(
                    message=error_message,
                    role="system",
                    history=history,
                    state=state,
                    retries=retries + 1,
                ):
                    yield ans, hist
                return

            self.log(f"Executing command {command.name}({params})")
            result = await command.execute(params, state=state)
            self.log("Result:", result)

            async for ans, hist in self.speak(
                message=result.json(),
                role="function",
                function_name=function_name,
                history=history,
                state=state,
            ):
                yield ans, hist
