from functools import wraps
from typing import Optional, Literal
from dataclasses import dataclass
import jinja2

import pydantic
import openai


ENGINES = {
    "gpt-3.5-turbo",
    "gpt-4",
}


@dataclass
class Command:
    name: str
    function: callable
    description: str
    params: pydantic.BaseModel
    returns: pydantic.BaseModel

    def explain(self):
        return jinja2.Template("""
            Command: {{name}}
            Description: {{description}}
            Parameters:
            | Name | Type | Description |
            | ---- | ---- | ----------- | \
            {% for param_name, param_type, param_desc in params %}
            | {{param_name}} | {{param_type}} | {{param_desc}} | \
            {% endfor %}

            Returns:
            | Name | Type | Description |
            | ---- | ---- | ----------- | \
            {% for res_name, res_type, res_desc in returns %}
            | {{res_name}} | {{res_type}} | {{res_desc}} | \
            {% endfor %}

            Example usage:
            {{mock_data}}
        """).render(
            name=self.name,
            description=self.description,
            params=[(key, value["type"], "") for key, value in self.params.schema()["properties"].items()],
            returns=[(key, value["type"], "") for key, value in self.returns.schema()["properties"].items()],
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

class Assistant:
    def __init__(self, key: str, short_description: str, engine: str = "gpt-4"):
        openai.api_key = key

        if engine not in ENGINES:
            raise ValueError(f"Unknown engine {engine}")

        self.engine = engine
        self.short_description = short_description
        self._messages = []
        
        self.commands = []

    @property
    def description(self) -> str:
        return self.short_description

    def system_prompt(self) -> str:
        return jinja2.Template("""
        {{short_description}}

        The following commands can be used.
        If you intend to execute a command, only write a valid command and nothing else.
        Do not try to both speak and execute a command at the same time, as it will not be accepted as a command.
        Also do not try to execute multiple commands at once.
        You can chain commands, but if so, only execute one command at a time, and then execute the next commands afterward.

        {% for command in commands %}
        {{command.explain()}}
        {% endfor %}
        """).render(
            short_description=self.short_description,
            commands=self.commands,
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

    def command(self, description: Optional[str] = None):
        def wrapper(function):
            assert len(function.__annotations__) == 2
            print(f"Creating command {function.__name__}")

            print(function.__annotations__)

            assert "return" in function.__annotations__
            assert issubclass(function.__annotations__["return"], pydantic.BaseModel)
            input_annotation = [(key, t) for key, t in function.__annotations__.items() if key != "return"]
            assert issubclass(input_annotation[0][1], pydantic.BaseModel)

            @wraps(function)
            def wrapped(*args, **kw):
                return function(*args, **kw)

            self.commands.append(
                Command(
                    name=function.__name__,
                    function=wrapped,
                    description=description,
                    params=input_annotation[0][1],
                    returns=function.__annotations__["return"]
                )
            )
            return wrapped
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
        print(content)

        for command in self.commands:
            model = command.command_model()
            print(model)
            try:
                inputs = model.parse_raw(content)
            except Exception as e:
                print(e)
                print("Not valid")
                continue
            print("Valid command!")
            result = command.function(inputs.params)
            return self.speak(result.json(), role="system")
        return content
