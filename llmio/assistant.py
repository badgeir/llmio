from typing import Literal, Optional
from dataclasses import dataclass
import jinja2
from datetime import datetime
from inspect import isclass

import pydantic
import openai

from llmio import model


ENGINES = {
    "gpt-3.5-turbo",
    "gpt-4",
}


@dataclass
class Command:
    function: callable

    @property
    def name(self):
        return self.function.__name__

    @property
    def is_pydantic_input(self):
        return bool(
            len(self.input_annotations) == 1 and
            issubclass(
                list(self.input_annotations.values())[0],
                pydantic.BaseModel
            )
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
            if name != "return"
        }

    @property
    def params(self):
        if self.is_pydantic_input:
            return list(self.input_annotations.values())[0]

        return model.model_from_function(self.function)

    @property
    def returns(self):
        annotation = self.function.__annotations__["return"]
        if isclass(annotation) and issubclass(annotation, pydantic.BaseModel):
            return annotation
        else:
            class Result(pydantic.BaseModel):
                result: annotation
            return Result

    @property
    def description(self):
        if self.function.__doc__ is None:
            return ""
        return self.function.__doc__.strip()

    def explain(self):
        return jinja2.Template("""
            Command: {{name}}
            Description: {{description}}
            Parameters:
            | Name | Type | Description |
            | ---- | ---- | ----------- | \
            {% for param_name, param_type, param_desc in params %}
            | {{param_name}} | {{param_type}} | {{param_desc}} | \
            {% endfor %}

            Returns:
            | Name | Type | Description |
            | ---- | ---- | ----------- | \
            {% for res_name, res_type, res_desc in returns %}
            | {{res_name}} | {{res_type}} | {{res_desc}} | \
            {% endfor %}

            Example usage:
            {{mock_data}}
        """).render(
            name=self.name,
            description=self.description,
            params=[(key, value["type"], value.get("description", "-")) for key, value in self.params.schema()["properties"].items()],
            returns=[(key, value["type"], value.get("description", "-")) for key, value in self.returns.schema()["properties"].items()],
            mock_data=self.mock_data().json(),
        )

    def command_model(self):
        class Model(pydantic.BaseModel):
            command: Literal[self.name]
            params: self.params
        return Model

    def mock_data(self):
        from polyfactory.factories.pydantic_factory import ModelFactory
        class Mocker(ModelFactory):
            __model__ = self.command_model()
        return Mocker.build()

    def execute(self, params):
        if self.is_pydantic_input:
            result = self.function(params)
        else:
            result = self.function(**params.dict())

        if self.is_pydantic_output:
            return result
        return self.returns(result=result)

class Assistant:
    def __init__(
            self,
            key: str,
            short_description: str,
            engine: str = "gpt-4",
            command_header: Optional[str] = None,
        ):
        openai.api_key = key

        if engine not in ENGINES:
            raise ValueError(f"Unknown engine {engine}")

        self.engine = engine
        self.short_description = short_description
        self._messages = []
        
        self.commands = []

        if command_header is None:
            self.command_header = """
                The following commands can be used.
                If you intend to execute a command, only write a valid command and nothing else.
                Do not try to both speak and execute a command at the same time, as it will not be accepted as a command.
                Also do not try to execute multiple commands at once.
                You can chain commands, but if so, only execute one command at a time, and then execute the next commands afterward.
                Every time a command is executed, the results will be shown as a system message, and you then get to either execute a new command or output a normal message intended to the user.
                Every time you return a normal text, this will stop the command iteration, and the text will be shown to the user. Because of this, do not hint that you will execute a command by saying something like "Ok, I will now do X". Instead, first execute the command, and then write a normal message to the user.
                Do not talk explicitly about the commands to the user, these are hidden and only serve as your interface to the application backend.
            """
        else:
            self.command_header = command_header

    @property
    def description(self) -> str:
        return self.short_description

    def system_prompt(self) -> str:
        return jinja2.Template("""
        {{short_description}}

        {{command_header}}

        {% for command in commands %}
        {{command.explain()}}
        {% endfor %}

        The current time is {{current_time}}
        """).render(
            short_description=self.short_description,
            command_header=self.command_header,
            commands=self.commands,
            current_time=datetime.now().isoformat()
        )

    def _get_system_prompt(self) -> str:
        return {
            "role": "system",
            "content": self.system_prompt()
        }

    @property
    def history(self):
        return [
            self._get_system_prompt(),
            *self._messages,
        ]

    def _run(self):
        result = openai.ChatCompletion.create(
            engine=self.engine,
            messages=self.history,
        )

        return result["choices"][0]["message"]["content"]

    def command(self):
        def wrapper(function):
            input_annotations = {
                name: annotation
                for name, annotation in function.__annotations__.items()
                if name != "return"
            }

            assert (
                len(input_annotations) == 1 and
                issubclass(
                    list(input_annotations.values())[0],
                    pydantic.BaseModel
                )
            ) or (
                not any(
                    issubclass(annotation, pydantic.BaseModel)
                    for annotation in input_annotations.values()
                )
            )

            assert "return" in function.__annotations__

            self.commands.append(
                Command(
                    function=function,
                )
            )
            return function
        return wrapper

    def speak(self, message: str, role="user") -> str:
        self._messages.append({
            "role": role,
            "content": message,
        })
        result = openai.ChatCompletion.create(
            model=self.engine,
            messages=self.history,
        )
        content = result["choices"][0]["message"]["content"]
        self._messages.append({
            "role": "assistant",
            "content": content,
        })
        for command in self.commands:
            model = command.command_model()
            try:
                inputs = model.parse_raw(content)
            except Exception as e:
                print(e)
                print("Not valid")
                continue
            print("Valid command!")
            result = command.execute(inputs.params)
            return self.speak(result.json(), role="system")
        return content
