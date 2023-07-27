from typing import Literal, Optional, Callable, Type, Any, AsyncIterable
from dataclasses import dataclass
import textwrap
from inspect import signature, iscoroutinefunction

import pydantic
import openai

from llmio import model


ENGINES = {
    "gpt-3.5-turbo",
    "gpt-4",
}


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
        openai.api_key = key

        if engine not in ENGINES:
            raise ValueError(f"Unknown engine {engine}")

        self.engine = engine
        self.description = textwrap.dedent(description).strip()
        self.commands: list[Command] = []
        self.debug = debug

        self._prompt_inspectors: list[Callable] = []
        self._output_inspectors: list[Callable] = []

    def system_prompt(self) -> str:
        return self.description

    def _get_system_prompt(self) -> dict[str, str]:
        return {"role": "system", "content": self.system_prompt()}

    def create_prompt(
        self, message_history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
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

    def _run_prompt_inspectors(self, prompt: list[dict[str, str]], state: Any) -> None:
        for inspector in self._prompt_inspectors:
            kwargs = {}
            if "state" in signature(inspector).parameters:
                kwargs["state"] = state
            inspector(prompt, **kwargs)

    def _run_output_inspectors(self, content: str, state: Any) -> None:
        for inspector in self._output_inspectors:
            kwargs = {}
            if "state" in signature(inspector).parameters:
                kwargs["state"] = state
            inspector(content, **kwargs)

    def log(self, *message: str) -> None:
        if self.debug:
            print(*message)

    async def speak(
        self,
        message: str | None = None,
        history: Optional[list[dict[str, str]]] = None,
        state: Any = None,
        role: Literal["user", "system", "function"] = "user",
        function_name: Optional[str] = None,
        retries=0,
    ) -> AsyncIterable[tuple[str, list[dict[str, str]]]]:
        if history is None:
            history = []
        history = history[:]

        if message:
            new_message = {
                "role": role,
                "content": message,
            }
            if function_name:
                assert role == "function"
                new_message["name"] = function_name
            history.append(new_message)

        prompt = self.create_prompt(history)
        self._run_prompt_inspectors(prompt, state)

        function_definitions = [
            command.function_definition for command in self.commands
        ]

        kwargs = {}
        if function_definitions:
            kwargs["functions"] = function_definitions

        result = await openai.ChatCompletion.acreate(
            model=self.engine,
            messages=prompt,
            **kwargs,
        )
        generated_message = result["choices"][0]["message"]
        self.log("Model output:", generated_message)
        self._run_output_inspectors(generated_message, state)

        history.append(generated_message)

        if generated_message["content"]:
            yield generated_message["content"], history

        if function_call := generated_message.get("function_call"):
            function_name = function_call["name"]
            arguments = function_call["arguments"]

            command = [cmd for cmd in self.commands if cmd.name == function_name][0]

            try:
                params = command.params.parse_raw(arguments)
            except pydantic.ValidationError as e:
                if retries > 1:
                    yield "I am sorry, I encountered an error.", history
                    return

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
