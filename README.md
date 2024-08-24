
# llmio
## A Lightweight Python Library for LLM I/O 🎈


![pylint](https://github.com/badgeir/llmio/actions/workflows/pylint.yml/badge.svg)
![mypy](https://github.com/badgeir/llmio/actions/workflows/mypy.yml/badge.svg)
![ruff](https://github.com/badgeir/llmio/actions/workflows/ruff.yml/badge.svg)
![tests](https://github.com/badgeir/llmio/actions/workflows/test.yml/badge.svg)
[![pypi](https://img.shields.io/pypi/v/llmio.svg)](https://pypi.python.org/pypi/llmio)
[![versions](https://img.shields.io/pypi/pyversions/llmio.svg)](https://github.com/badgeir/llmio)
[![Downloads](https://static.pepy.tech/badge/llmio/month)](https://pepy.tech/project/llmio)

Welcome to **llmio**! If you're looking for a simple, efficient way to manage LLMs, you've come to the right place.

**llmio** is a lightweight Python library that leverages type annotations to make tool execution with OpenAI-compatible APIs effortless. Whether you're working with OpenAI, Azure OpenAI, Google Gemini, AWS Bedrock, or Huggingface TGI, llmio has you covered.

## Why choose llmio?

- **Lightweight** 🪶: Designed to integrate smoothly into your project without adding unnecessary bulk.
- **Type Annotations** 🏷️: Easily define tools with Python's type annotations and let llmio handle the rest.
- **Broad API Compatibility** 🌍: Seamlessly works with major APIs like OpenAI, Azure, Google Gemini, AWS, and Huggingface.

## Overview

1. [Getting started](#getting-started-🚀)
2. [Examples](#examples)
3. [Details](#details)
    - [Tools](#tools)
    - [Parameter descriptions](#parameter-descriptions)
    - [Optional parameters](#optional-parameters)
    - [Supported parameter types](#supported-parameter-types)
    - [Hooks](#hooks)
    - [Keeping track of context](#keeping-track-of-context)
    - [Batched execution](#batched-execution)
    - [A simple example of continuous interaction](#a-simple-example-of-continuous-interaction)
    - [Handling Uninterpretable Tool Calls](#handling-uninterpretable-tool-calls)
    - [Strict tool mode](#strict-tool-mode)
    - [Structured output](#structured-output)

## Getting Started 🚀

Get started quickly with a simple installation:

``` bash
pip install llmio
```

**Set Up Your Agent**: Start building with a few lines of code:

``` python
import asyncio
from llmio import Agent
from openai import AsyncOpenAI

agent = Agent(
    instruction="You are a task manager.",
    client=AsyncOpenAI(api_key="your_openai_api_key"),
)

# Add tools and interact with your agent...
```

## Examples

``` python
import asyncio
import os

import openai
from llmio import Agent


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


## Details

### Tools

Under the hood, `llmio` uses type annotations to build function schemas compatible with OpenAI tools.

It also builds pydantic models in order to validate the input types of the arguments passed by the language model.

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

`pydantic.Field` can be used to describe parameters in detail. These descriptions will be included in the schema and help the language model understand the tool's requirements.

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

Optional parameters are supported.

``` python
@agent.tool
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

### Keeping track of context

You can pass an object of any type to the agent to maintain context. This context will be available to tools and other hooks that include the special argument `_context`, but it will not be passed to the model itself.

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

### Batched execution

The Agent class is stateless, allowing you to safely use `asyncio.gather` to execute multiple messages in parallel.

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

#### Or by using the messages returned by the agent

``` python
async def main() -> None:
    history = []
    
    while True:
        response = await agent.speak(input(">>"), history=history)
        history = response.history
        for message in response.messages:
            print(message)
```

### Handling Uninterpretable Tool Calls

The agent can be set up to either raise an exception or provide feedback to the model when it makes an uninterpretable tool call. By default, the agent will raise an exception if the model attempts to call an unrecognized tool or passes invalid arguments.

``` python
# This will raise an exception if the model tries to call a tool
# that the agent does not recognize or if the arguments are not valid.
agent = Agent(
    client=openai.AsyncOpenAI(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
    graceful_errors=False,  # This is the default
)

# This will try to explain to the model what it did wrong
# if it tries to call a tool that the agent does not recognize
# or if the arguments are not valid.
agent = Agent(
    client=openai.AsyncOpenAI(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
    graceful_errors=True,
)
```

### Strict tool mode

OpenAI supports strict mode for tools, ensuring that tools are only called with arguments that adhere to the defined function schema. This can be enabled by setting strict=True in the tool decorator, though this feature may not be available with other providers.

``` python
@agent.tool(strict=True)
async def add_task(name: str, description: str | None = None) -> str:
    ...
```

### Structured output

The agent can be set up to return structured output on the messages it generates. This can be useful for more advanced use cases. Note that this feature might not be available with all providers (as of now, only OpenAI and Azure OpenAI support it).

``` python
import asyncio
from pprint import pprint
from typing import Literal

import pydantic
import os
import openai

from llmio import StructuredAgent


class OutputFormat(pydantic.BaseModel):
    answer: str
    detected_sentiment: Literal["positive", "negative", "neutral"]


agent = StructuredAgent(
    instruction="Answer the questions and detect the user sentiment.",
    client=openai.AsyncOpenAI(api_key=os.environ["OPENAI_TOKEN"]),
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

## Get Involved 🎉

Your feedback, ideas, and contributions are welcome! Feel free to open an issue, submit a pull request, or start a Discussion.