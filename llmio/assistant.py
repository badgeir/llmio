from typing import Literal, Optional, Callable, Type, Any
from dataclasses import dataclass
import textwrap
from datetime import datetime
from inspect import isclass, signature

import jinja2
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
    def is_pydantic_input(self) -> bool:
        return bool(
            len(self.input_annotations) == 1
            and issubclass(self.pydantic_input, pydantic.BaseModel)
        )

    @property
    def is_pydantic_output(self) -> bool:
        annotation = self.function.__annotations__["return"]
        if isclass(annotation) and issubclass(annotation, pydantic.BaseModel):
            return True
        return False

    @property
    def input_annotations(self) -> dict[str, type]:
        return {
            name: annotation
            for name, annotation in self.function.__annotations__.items()
            if name not in {"return", "state"}
        }

    @property
    def pydantic_input(self) -> Type[pydantic.BaseModel]:
        input_type = list(self.input_annotations.values())[0]
        assert issubclass(input_type, pydantic.BaseModel)
        return input_type

    @property
    def params(self) -> Type[pydantic.BaseModel]:
        if self.is_pydantic_input:
            return self.pydantic_input

        return model.model_from_function(self.function)

    @property
    def returns(self) -> Type[pydantic.BaseModel]:
        annotation = self.function.__annotations__["return"]
        if isclass(annotation) and issubclass(annotation, pydantic.BaseModel):
            return annotation

        return pydantic.create_model("Result", result=(annotation, ...))

    @property
    def description(self) -> str:
        if self.function.__doc__ is None:
            return ""
        return textwrap.dedent(self.function.__doc__).strip()

    def execute(self, params: pydantic.BaseModel, state=None):
        kwargs = {}
        if "state" in signature(self.function).parameters:
            kwargs["state"] = state

        if self.is_pydantic_input:
            result = self.function(params, **kwargs)
        else:
            result = self.function(**params.dict(), **kwargs)

        if self.is_pydantic_output:
            return result
        return self.returns(result=result)

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
        self.description = description
        self.commands: list[Command] = []
        self.debug = debug

        self._prompt_inspectors: list[Callable] = []
        self._output_inspectors: list[Callable] = []

    def system_prompt(self) -> str:
        return jinja2.Template(
            self.description,
            lstrip_blocks=True,
            trim_blocks=True,
        ).render(current_time=datetime.now().isoformat())

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
        input_annotations = {
            name: annotation
            for name, annotation in function.__annotations__.items()
            if name not in {"return", "state"}
        }

        assert (
            len(input_annotations) == 1
            and issubclass(list(input_annotations.values())[0], pydantic.BaseModel)
        ) or (
            not any(
                issubclass(annotation, pydantic.BaseModel)
                for annotation in input_annotations.values()
            )
        )

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

    def _run_content_inspectors(self, content: str, state: Any) -> None:
        for inspector in self._output_inspectors:
            kwargs = {}
            if "state" in signature(inspector).parameters:
                kwargs["state"] = state
            inspector(content, **kwargs)

    def log(self, *message: str) -> None:
        if self.debug:
            print(*message)

    def speak(
        self,
        message: str,
        history: Optional[list[dict[str, str]]] = None,
        state: Any = None,
        role: Literal["user", "system", "function"] = "user",
        function_name: Optional[str] = None,
    ) -> tuple[str, list[dict[str, str]]]:
        if history is None:
            history = []
        history = history[:]

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
            command.function_definition() for command in self.commands
        ]

        result = openai.ChatCompletion.create(
            model=self.engine,
            messages=prompt,
            functions=function_definitions,
        )
        generated_message = result["choices"][0]["message"]
        content = generated_message["content"]

        self.log("Model output:", content)
        self._run_content_inspectors(content, state)

        history.append(generated_message)

        if function_call := generated_message.get("function_call"):
            function_name = function_call["name"]
            arguments = function_call["arguments"]

            command = [cmd for cmd in self.commands if cmd.name == function_name][0]

            params = command.params.parse_raw(arguments)

            self.log(f"Executing command {command.name}({params})")
            result = command.execute(params, state=state)
            self.log("Result:", result)

            return self.speak(
                message=result.json(),
                role="function",
                function_name=function_name,
                history=history,
                state=state,
            )

        return content, history
