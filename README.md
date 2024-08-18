# llmio
## Easily connect large language models into your application

![pylint](https://github.com/badgeir/llmio/actions/workflows/pylint.yml/badge.svg)
![mypy](https://github.com/badgeir/llmio/actions/workflows/mypy.yml/badge.svg)
![ruff](https://github.com/badgeir/llmio/actions/workflows/ruff.yml/badge.svg)
![tests](https://github.com/badgeir/llmio/actions/workflows/test.yml/badge.svg)

llmio uses type annotations to enable tool execution with OpenAI-compatible APIs.

# Setup

```
pip install llmio
```

# Examples

``` python

import asyncio
import os

import openai

from llmio import Assistant, Message


assistant = Assistant(
    instruction="""
        You are a calculating assistant.
        Always use tools to calculate things, never try to calculate things on your own.
        """,
    client=openai.AsyncOpenAI(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
)


@assistant.command()
async def add(num1: float, num2: float) -> float:
    print(f"Adding {num1} + {num2}")
    return num1 + num2


@assistant.command()
async def multiply(num1: float, num2: float) -> float:
    print(f"Multiplying {num1} * {num2}")
    return num1 * num2


async def main():
    history: list[Message] = []

    while True:
        async for reply, history in assistant.speak(input(">>"), history):
            print(reply)


if __name__ == "__main__":
    asyncio.run(main())
```

### More examples

For more examples, see `examples/`.


# Details

Under the hood, `llmio` uses type annotations to build function schemas compatible with OpenAI tools.

It also builds pydantic models in order to validate the input types of the arguments passed by the language model.

``` python
@assistant.command()
async def add(num1: float, num2: float) -> float:
    """
    The docstring is used as the description of the command.
    """
    return num1 + num2


print("Command name:", assistant.commands[-1].name)
print("Schema:")
pprint(assistant.commands[-1].tool_definition)
print(
    "Parsed arguments:",
    assistant.commands[-1].params.parse_raw('{"num1": 1, "num2": 2}'),
)
```

Output:
``` plaintext
Command name: add
Schema:
{'description': 'The docstring is used as the description of the command.',
 'name': 'add',
 'parameters': {'properties': {'num1': {'type': 'number'},
                               'num2': {'type': 'number'}},
                'required': ['num1', 'num2'],
                'type': 'object'},
 'strict': False}
Parsed arguments: num1=1.0 num2=2.0
```

#### Parameter descriptions

`pydantic.Field` can be used to describe parameters in detail.

``` python
@assistant.command()
async def book_flight(
    destination: str = Field(..., description="The destination airport"),
    origin: str = Field(..., description="The origin airport"),
    date: datetime = Field(
        ..., description="The date of the flight. ISO-format is expected."
    ),
) -> str:
    """Books a flight"""
    return f"Booked flight from {origin} to {destination} on {date}"

print("Schema:")
pprint(assistant.commands[-1].tool_definition)
print(
    "Parsed arguments:",
    assistant.commands[-1].params.parse_raw('{"destination": "Oslo", "origin": "Madrid", "date": "2024-12-24"}'),
)
```

Output:
``` plaintext
Schema:
{'description': 'Books a flight',
 'name': 'book_flight',
 'parameters': {'properties': {'date': {'description': 'The date of the '
                                                       'flight. ISO-format is '
                                                       'expected.',
                                        'format': 'date',
                                        'type': 'string'},
                               'destination': {'description': 'The destination '
                                                              'airport',
                                               'type': 'string'},
                               'origin': {'description': 'The origin airport',
                                          'type': 'string'}},
                'required': ['destination', 'origin', 'date'],
                'type': 'object'},
 'strict': False}
Parsed arguments: destination='Oslo' origin='Madrid' date=datetime.date(2024, 12, 24)
```

### Optional parameters

Optional parameters are supported.

``` python
@assistant.command()
async def create_task(name: str = "My task", description: str | None = None) -> str:
    return "Created task"
```

### Hooks

Add hooks to receive callbacks with prompts and outputs.

``` python
@assistant.inspect_prompt
def print_prompt(prompt: list[Message]):
    pprint(prompt)


@assistant.inspect_output
def print_model_output(output: Message):
    pprint(output)
``` 

### Pass a state to keep track of context in commands and hooks

Pass a state of any type to the assistant to keep track of context. This state will only be passed to commands and inspectors that include the special argument `_state`, not to the model itself.

``` python
@dataclass
class User:
    id: str
    name: str


@assistant.command()
async def create_task(task_name: str, _state: User) -> str:
    print(f"Created task {task_name} for user {_state.id}")
    return "Created task"



async def main() -> None:
    history = []
    async for reply, history in assistant.speak(
        "Create a task named 'Buy milk'",
        history,
        _state=User(id="1", name="Alice"),
    ):
        print(reply)

```
