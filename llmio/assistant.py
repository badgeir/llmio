from typing import Literal, Optional, Callable, Type, Any
from dataclasses import dataclass
import textwrap
from datetime import datetime
from inspect import isclass, signature

import jinja2
import pydantic
import openai
from polyfactory.factories.pydantic_factory import ModelFactory

from llmio import model, prompts


ENGINES = {
    "gpt-3.5-turbo",
    "gpt-4",
}


@dataclass
class Command:
    function: Callable

    @property
    def name(self):
        return self.function.__name__

    @property
    def is_pydantic_input(self):
        return bool(
            len(self.input_annotations) == 1
            and issubclass(list(self.input_annotations.values())[0], pydantic.BaseModel)
        )

    @property
    def is_pydantic_output(self):
        annotation = self.function.__annotations__["return"]
        if isclass(annotation) and issubclass(annotation, pydantic.BaseModel):
            return True
        return False

    @property
    def input_annotations(self):
        return {
            name: annotation
            for name, annotation in self.function.__annotations__.items()
            if name not in {"return", "state"}
        }

    @property
    def params(self) -> pydantic.BaseModel:
        if self.is_pydantic_input:
            return list(self.input_annotations.values())[0]

        return model.model_from_function(self.function)

    @property
    def returns(self) -> Type[pydantic.BaseModel]:
        annotation = self.function.__annotations__["return"]
        if isclass(annotation) and issubclass(annotation, pydantic.BaseModel):
            return annotation

        return pydantic.create_model("Result", result=(annotation, ...))

    @property
    def description(self):
        if self.function.__doc__ is None:
            return ""
        return textwrap.dedent(self.function.__doc__).strip()

    def explain(self):
        return jinja2.Template(
            prompts.DEFAULT_COMMAND_PROMPT,
            trim_blocks=True,
            lstrip_blocks=True,
        ).render(
            name=self.name,
            description=self.description,
            params=[
                (
                    key,
                    value["type"],
                    value.get("description", "-"),
                    key in self.params.schema()["required"],
                )
                for key, value in self.params.schema()["properties"].items()
            ],
            returns=[
                (key, value["type"], value.get("description", "-"))
                for key, value in self.returns.schema()["properties"].items()
            ],
            mock_data=self.mock_data().json(),
        )

    def command_model(self) -> Any:
        return pydantic.create_model(
            "Model", command=(Literal[self.name], ...), params=(self.params, ...)
        )

    def mock_data(self):
        class Mocker(ModelFactory):
            __model__ = self.command_model()

        return Mocker.build()

    def execute(self, params, state=None):
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


class Assistant:
    def __init__(
        self,
        key: str,
        description: str,
        engine: str = "gpt-4",
        command_header: Optional[str] = None,
    ):
        openai.api_key = key

        if engine not in ENGINES:
            raise ValueError(f"Unknown engine {engine}")

        self.engine = engine
        self.description = description
        self.commands: list[Command] = []

        if command_header is None:
            self.command_header = prompts.DEFAULT_COMMAND_HEADER
        else:
            self.command_header = command_header

        self._prompt_inspectors: list[Callable] = []
        self._output_inspectors: list[Callable] = []

    def system_prompt(self) -> str:
        return jinja2.Template(
            prompts.DEFAULT_SYSTEM_PROMPT,
            trim_blocks=True,
            lstrip_blocks=True,
        ).render(
            description=self.description,
            command_header=self.command_header,
            commands=self.commands,
            current_time=datetime.now().isoformat(),
        )

    def _get_system_prompt(self) -> dict[str, str]:
        return {"role": "system", "content": self.system_prompt()}

    def create_prompt(
        self, message_history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        return [
            self._get_system_prompt(),
            *message_history,
        ]

    def command(self, function: Callable):
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

    def inspect_prompt(self, function: Callable):
        self._prompt_inspectors.append(function)
        return function

    def inspect_output(self, function: Callable):
        self._output_inspectors.append(function)
        return function

    def _run_prompt_inspectors(self, prompt: list[dict[str, str]], state) -> None:
        for inspector in self._prompt_inspectors:
            kwargs = {}
            if "state" in signature(inspector).parameters:
                kwargs["state"] = state
            inspector(prompt, **kwargs)

    def _run_content_inspectors(self, content: str, state) -> None:
        for inspector in self._output_inspectors:
            kwargs = {}
            if "state" in signature(inspector).parameters:
                kwargs["state"] = state
            inspector(content, **kwargs)

    def speak(
        self,
        message: str,
        history: Optional[list[dict[str, str]]] = None,
        state=None,
        role="user",
    ) -> tuple[str, list[dict[str, str]]]:
        if history is None:
            history = []
        history = history[:]
        history.append(
            {
                "role": role,
                "content": message,
            }
        )

        prompt = self.create_prompt(history)
        self._run_prompt_inspectors(prompt, state)

        result = openai.ChatCompletion.create(
            model=self.engine,
            messages=prompt,
        )
        content = result["choices"][0]["message"]["content"]
        self._run_content_inspectors(content, state)

        history.append(
            {
                "role": "assistant",
                "content": content,
            }
        )
        for command in self.commands:
            cmd_model = command.command_model()
            try:
                inputs = cmd_model.parse_raw(content)
            except pydantic.ValidationError:
                continue
            result = command.execute(inputs.params, state=state)
            return self.speak(
                result.json(),
                history=history,
                role="system",
                state=state,
            )
        return content, history
