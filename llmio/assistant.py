from functools import wraps
from typing import Any
from dataclasses import dataclass

import openai


ENGINES = {
    "gpt-3.5-turbo",
    "gpt-4",
}


@dataclass
class Command:
    name: str
    description: str
    params: Any


class Assistant:
    def __init__(self, key: str, short_description: str, engine: str = "gpt-3.5-turbo"):
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

    def _get_system_prompt(self) -> str:
        return {
            "role": "system",
            "content": self.description
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

    def command(self, name: str, description: str):
        print(f"Creating command {name}")
        def wrapper(function):
            @wraps(function)
            def wrapped(*args, **kw):
                return function(*args, **kw)

            self.commands.append(
                Command(
                    name=name,
                    description=description,
                    params=[
                        (name, type)
                        for name, type in function.__annotations__.items()
                        if name != "return"
                    ]
                )
            )
            return wrapped
        return wrapper

    def run(self):
        pass
