
# llmio
## Seamless Tool Integration for LLM-powered Python Applications


![pylint](https://github.com/badgeir/llmio/actions/workflows/pylint.yml/badge.svg)
![mypy](https://github.com/badgeir/llmio/actions/workflows/mypy.yml/badge.svg)
![ruff](https://github.com/badgeir/llmio/actions/workflows/ruff.yml/badge.svg)
![tests](https://github.com/badgeir/llmio/actions/workflows/test.yml/badge.svg)
[![pypi](https://img.shields.io/pypi/v/llmio.svg)](https://pypi.python.org/pypi/llmio)
[![versions](https://img.shields.io/pypi/pyversions/llmio.svg)](https://github.com/badgeir/llmio)

#### *LLM I/O made easy*

llmio is a lightweight library that leverages type annotations to enable seamless tool execution with OpenAI-compatible APIs, including OpenAI, Azure OpenAI, AWS Bedrock Access Gateway, and Huggingface TGI.


# Setup

```
pip install llmio
```

# Examples

``` python
import asyncio
import os

import openai
from llmio.agent import Agent


# Define an agent that can add and multiply numbers using tools.
# The agent will also print any messages it receives.
agent = Agent(
    # Define the agent's instructions.
    instruction="""
        You are a calculating agent.
        Always use tools to calculate things.
        Never try to calculate things on your own.
        """,
    # Pass in an OpenAI client that will be used to interact with the model.
    # Any API that implements the OpenAI interface can be used.
    client=openai.AsyncOpenAI(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
)


# Define tools using the `@agent.tool()` decorator.
# Tools are automatically parsed by their type annotations
# and added to the agent's capabilities.
@agent.tool()
async def add(num1: float, num2: float) -> float:
    print(f"** Adding: {num1} + {num2}")
    return num1 + num2


# Tools can also be synchronous.
@agent.tool()
def multiply(num1: float, num2: float) -> float:
    print(f"** Multiplying: {num1} * {num2}")
    return num1 * num2


# Define a message handler using the `@agent.on_message` decorator.
# The handler is optional. The messages will also be returned by the `speak` method.
@agent.on_message
async def print_message(message: str):
    print(f"** Posting message: '{message}'")


async def main():
    # Run the agent with a message.
    # An empty history might also be passed in.
    # The agent will return the messages it generated and the updated history.
    messages, history = await agent.speak("Hi! how much is 1 + 1?")
    # The agent is stateless and does not remember previous messages.
    # The history must be passed in to maintain context.
    messages, history = await agent.speak(
        "and how much is that times two?", history=history
    )


if __name__ == "__main__":
    asyncio.run(main())
```

### More examples

For more examples, see `examples/`.


# Details

Under the hood, `llmio` uses type annotations to build function schemas compatible with OpenAI tools.

It also builds pydantic models in order to validate the input types of the arguments passed by the language model.

``` python
@agent.tool()
async def add(num1: float, num2: float) -> float:
    """
    The docstring is used as the description of the tool.
    """
    return num1 + num2


print(agent.summary())
```

Output:
``` plaintext
Tools:
  - add
    Schema:
      {'description': 'The docstring is used as the description of the tool.',
       'name': 'add',
       'parameters': {'properties': {'num1': {'type': 'number'},
                                     'num2': {'type': 'number'}},
                      'required': ['num1', 'num2'],
                      'type': 'object'},
       'strict': False}
```

#### Parameter descriptions

`pydantic.Field` can be used to describe parameters in detail. These descriptions will be included in the schema and help the language model understand the tool's requirements.

``` python
@agent.tool()
async def book_flight(
    destination: str = Field(..., description="The destination airport"),
    origin: str = Field(..., description="The origin airport"),
    date: datetime = Field(
        ..., description="The date of the flight. ISO-format is expected."
    ),
) -> str:
    """Books a flight"""
    return f"Booked flight from {origin} to {destination} on {date}"
```

### Optional parameters

Optional parameters are supported.

``` python
@agent.tool()
async def create_task(name: str = "My task", description: str | None = None) -> str:
    return "Created task"
```

### Supported parameter types

Types supported by pydantic are supported.
For documentation on supported types, see [pydantic's documentation](https://docs.pydantic.dev/latest/concepts/types).

### Hooks

Add hooks to receive callbacks with prompts and outputs. Note that llmio does not care what name you give to the hooks, as long as they are decorated with the correct decorator.

``` python
@agent.on_message
async def on_message(message: str):
    # on_message will be called with new messages from the model
    pprint(prompt)

@agent.inspect_prompt
async def inspect_prompt(prompt: list[llmio.Message]):
    # inspect_prompt will be called with the prompt before it is sent to the model
    pprint(prompt)


@agent.inspect_output
async def inspect_output(output: llmio.Message):
    # inspect_output will be called with the full model output
    pprint(output)
``` 

### Pass a context to keep track of context in tools and hooks

Pass an object of any type to the agent to keep track of context. This context will only be passed to tools and other hooks that include the special argument `_context`, not to the model itself.

``` python
@dataclass
class User:
    name: str


@agent.tool()
async def create_task(task_name: str, _context: User) -> str:
    print(f"** Created task {task_name} for user '{_context.name}'")
    return "Created task"


@agent.on_message
async def (message: str, _context: User) -> None:
    print(f"** Sending message to user {_context.name}: {message}")


async def main() -> None:
    _ = await agent.speak(
        "Create a task named 'Buy milk'",
        _context=User(name="Alice"),
    )
```

### Batched execution

Since the Agent class is stateless, asyncio.gather can be safely used to run multiple messages in parallel.

``` python
async def main() -> None:
    await asyncio.gather(
        agent.speak("Create a task named 'Buy milk'", history=[], _context=User(name="Alice")),
        agent.speak("Create a task named 'Buy bread'", history=[], _context=User(name="Bob")),
    )
```

### A simple example of looping

``` python
@agent.on_message
async def print_message(message: str):
    print(message)


async def main() -> None:
    history = []
    
    while True:
        _, history = await agent.speak(input(">>"), history=history)

```

### Or by using the messages returned by the agent

``` python
async def main() -> None:
    history = []
    
    while True:
        messages, history = await agent.speak(input(">>"), history=history)
        for message in messages:
            print(message)
```
