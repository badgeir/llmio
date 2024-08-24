
# llmio
## 🎈 A Lightweight Python Library for LLM I/O


![pylint](https://github.com/badgeir/llmio/actions/workflows/pylint.yml/badge.svg)
![mypy](https://github.com/badgeir/llmio/actions/workflows/mypy.yml/badge.svg)
![ruff](https://github.com/badgeir/llmio/actions/workflows/ruff.yml/badge.svg)
![tests](https://github.com/badgeir/llmio/actions/workflows/test.yml/badge.svg)
[![pypi](https://img.shields.io/pypi/v/llmio.svg)](https://pypi.python.org/pypi/llmio)
[![versions](https://img.shields.io/pypi/pyversions/llmio.svg)](https://github.com/badgeir/llmio)
[![Downloads](https://static.pepy.tech/badge/llmio/month)](https://pepy.tech/project/llmio)

Welcome to **llmio**! If you're looking for a simple, efficient way to build LLM-based agents, you've come to the right place.

**llmio** is a lightweight Python library that leverages type annotations to make tool execution with OpenAI-compatible APIs effortless. Whether you're working with OpenAI, Azure OpenAI, Google Gemini, AWS Bedrock, or Huggingface TGI, llmio has you covered.

## Why choose llmio?

- **Lightweight** 🪶: Designed to integrate smoothly into your project without adding unnecessary bulk.
- **Type Annotations** 🏷️: Easily define tools with Python's type annotations and let llmio handle the rest.
- **Broad API Compatibility** 🌍: Seamlessly works with major APIs like OpenAI, Azure, Google Gemini, AWS, and Huggingface.

## Overview

1. [Getting started](#getting-started-🚀)
2. [Examples](#examples)
    - [A simple calculator example](#-a-simple-calculator-example)
    - [More examples](#more-examples)
3. [Details](#details-)
    - [Tools](#tools)
    - [Parameter descriptions](#parameter-descriptions)
    - [Optional parameters](#optional-parameters)
    - [Supported parameter types](#supported-parameter-types)
    - [Hooks](#hooks)
    - [Keeping track of context](#keeping-track-of-context)
    - [Dynamic instructions](#dynamic-instructions)
    - [Batched execution](#batched-execution)
    - [A simple example of continuous interaction](#a-simple-example-of-continuous-interaction)
    - [Handling uninterpretable tool calls](#handling-uninterpretable-tool-calls)
    - [Strict tool mode](#strict-tool-mode)
    - [Structured output](#structured-output)
    - [Get involved](#get-involved-)

## Getting Started 🚀

Get started quickly with a simple installation:

``` bash
pip install llmio
```

**Set Up Your Agent**: Start building with a few lines of code:

``` python
import asyncio
from llmio import Agent, OpenAIClient


agent = Agent(
    instruction="You are a task manager.",
    client=OpenAIClient(api_key="your_openai_api_key"),
)

# Add tools and interact with your agent...
```

## Examples

### 💻 A simple calculator example

Let’s walk through a basic example where we create a simple calculator using llmio. This calculator can add and multiply numbers, leveraging AI to handle the operations. It’s a straightforward way to see how llmio can manage tasks while keeping the code clean and easy to follow.

``` python
import asyncio
import os

from llmio import Agent, OpenAIClient


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
    client=OpenAIClient(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
)


# Define tools using the `@agent.tool` decorator.
# Tools are automatically parsed by their type annotations
# and added to the agent's capabilities.
# The code itself is never seen by the LLM, only the function signature is exposed.
# When the agent invokes a tool, the corresponding function is executed locally.
@agent.tool
async def add(num1: float, num2: float) -> float:
    print(f"** Executing add({num1}, {num2}) -> {num1 + num2}")
    return num1 + num2


# Tools can also be synchronous.
@agent.tool
def multiply(num1: float, num2: float) -> float:
    print(f"** Executing multiply({num1}, {num2}) -> {num1 * num2}")
    return num1 * num2


# Define a message handler using the `@agent.on_message` decorator.
# The handler is optional. The messages will also be returned by the `speak` method.
@agent.on_message
async def print_message(message: str):
    print(f"** Posting message: '{message}'")


async def main():
    # Run the agent with a message.
    # The agent will return a response containing the messages it generated and the updated history.
    response = await agent.speak("Hi! how much is 1 + 1?")
    # The agent is stateless and does not remember previous messages by itself.
    # The history must be passed in to maintain context.
    response = await agent.speak(
        "and how much is that times two?", history=response.history
    )


if __name__ == "__main__":
    asyncio.run(main())

# Output:
# ** Executing add(1.0, 1.0) -> 2.0
# ** Posting message: '1 + 1 is 2.'
# ** Executing multiply(2.0, 2.0) -> 4.0
# ** Posting message: 'That times two is 4.'
```

### More examples

For more examples, see `examples/`.

For a notebook going throught how to create a simple AI task manager, see [examples/notebooks/simple_task_manager.ipynb](examples/notebooks/simple_task_manager.ipynb).


## Details 🔍

### Tools

Under the hood, llmio uses Python's type annotations to automatically generate function schemas that are compatible with OpenAI tools. It also leverages Pydantic models to validate the input types of arguments passed by the language model, ensuring robust and error-free execution.

``` python
@agent.tool
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

### Parameter descriptions

You can use pydantic.Field to describe parameters in detail. These descriptions will be included in the tool schema, guiding the language model to understand the tool's requirements better.

``` python
@agent.tool
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

llmio supports optional parameters seamlessly.

``` python
@agent.tool
async def create_task(name: str = "My task", description: str | None = None) -> str:
    return "Created task"
```

### Supported parameter types

llmio supports the types that are supported by Pydantic. For more details, refer to [Pydantic's documentation](https://docs.pydantic.dev/latest/concepts/types).

### Hooks

You can add hooks to receive callbacks with prompts and outputs. The names of the hooks are flexible as long as they are decorated appropriately.

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

### Keeping track of context

Pass an object of any type to the agent to maintain context across interactions. This context is available to tools and hooks via the special `_context` argument but is not passed to the language model itself.

``` python
@dataclass
class User:
    name: str


@agent.tool
async def create_task(task_name: str, _context: User) -> str:
    print(f"** Created task '{task_name}' for user '{_context.name}'")
    return "Created task"

@agent.on_message
async def on_message(message: str, _context: User) -> None:
    print(f"** Sending message to user '{_context.name}': {message}")


async def main() -> None:
    _ = await agent.speak(
        "Create a task named 'Buy milk'",
        _context=User(name="Alice"),
    )
```

### Dynamic instructions

`llmio` allows you to inject dynamic content into your instructions using variable hooks. These hooks act as placeholders, filling in values at runtime.

When an instruction contains a placeholder that matches the name of a variable hook, `llmio` will automatically replace it with the corresponding value returned by the hook. If a placeholder does not have a matching variable hook, a `MissingVariable` error will be raised.

``` python
agent = Agent(
    instruction="""
        You are a task manager for a user named {user_name}.
        The current time is {current_time}.
    """,
    ...
)

@agent.variable
def user_name(_context: User) -> str:
    return _context.name

@agent.variable
async def current_time() -> datetime:
    return datetime.now()

# Example of formatted instruction:
# "You are a task manager for a user named Alice.
#  The current time is 2024-08-25 10:17:04.606621."
```

### Batched execution

Since the `Agent` class is stateless, you can safely execute multiple messages in parallel using `asyncio.gather`.

``` python
async def main() -> None:
    await asyncio.gather(
        agent.speak("Create a task named 'Buy milk'", history=[], _context=User(name="Alice")),
        agent.speak("Create a task named 'Buy bread'", history=[], _context=User(name="Bob")),
    )
```

### A simple example of continuous interaction

``` python
@agent.on_message
async def print_message(message: str):
    print(message)


async def main() -> None:
    history = []
    while True:
        response = await agent.speak(input(">>"), history=history)
        history = response.history

```

Alternatively, use the messages returned by the agent:


``` python
async def main() -> None:
    history = []
    
    while True:
        response = await agent.speak(input(">>"), history=history)
        history = response.history
        for message in response.messages:
            print(message)
```

### Handling uninterpretable tool calls

`llmio` allows you to handle uninterpretable tool calls gracefully. By default, the agent will raise an exception if it encounters an unrecognized tool or invalid arguments. However, you can configure it to provide feedback to the model instead.

``` python
# Raises an exception for unrecognized tools or invalid arguments
agent = Agent(
    client=OpenAIClient(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
    graceful_errors=False,  # This is the default
)

# Provides feedback to the model for unrecognized tools or invalid arguments
agent = Agent(
    client=OpenAIClient(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
    graceful_errors=True,
)
```

### Strict tool mode

OpenAI supports a strict mode for tools, ensuring that only valid arguments are passed according to the function schema. Enable this by setting `strict=True` in the tool decorator.

``` python
@agent.tool(strict=True)
async def add_task(name: str, description: str | None = None) -> str:
    ...
```

### Structured output

`llmio` can return structured output from the messages it generates, ideal for more advanced use cases. This feature is currently supported by OpenAI and Azure OpenAI.

``` python
import asyncio
from pprint import pprint
from typing import Literal

import pydantic
import os

from llmio import StructuredAgent, OpenAIClient


class OutputFormat(pydantic.BaseModel):
    answer: str
    detected_sentiment: Literal["positive", "negative", "neutral"]


agent = StructuredAgent(
    instruction="Answer the questions and detect the user sentiment.",
    client=OpenAIClient(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
    response_format=OutputFormat,
)


@agent.on_message
async def print_message(message: OutputFormat):
    print(type(message))
    pprint(message.dict())


async def main() -> None:
    _ = await agent.speak("I am happy!")


if __name__ == "__main__":
    asyncio.run(main())

# Output:
# <class '__main__.OutputFormat'>
# {'answer': "That's great to hear! Happiness is a wonderful feeling.",
#  'detected_sentiment': 'positive'}
```

## Get involved 🎉

Your feedback, ideas, and contributions are welcome! Feel free to open an issue, submit a pull request, or start a discussion to help make `llmio` even better.
