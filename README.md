# llmio
## Easily connect large language models into your application

![pylint](https://github.com/badgeir/llmio/actions/workflows/pylint.yml/badge.svg)
![mypy](https://github.com/badgeir/llmio/actions/workflows/mypy.yml/badge.svg)
![ruff](https://github.com/badgeir/llmio/actions/workflows/ruff.yml/badge.svg)
![tests](https://github.com/badgeir/llmio/actions/workflows/test.yml/badge.svg)

llmio uses type hints to enable function execution with OpenAIs API.

# Setup

```
pip install llmio
```

# Example

``` python

import asyncio

from llmio import Assistant


assistant = Assistant(
    instruction="""
        You are a calculating assistant.
        Always use commands to calculate things.
        Never try to calculate things on your own.
        """,
    key=os.environ["OPENAI_TOKEN"],
    model="gpt-4o-mini",
)


@assistant.command
async def add(num1: float, num2: float) -> float:
    """
    Add two numbers
    """
    print(f"Adding {num1} + {num2}")
    return num1 + num2


@assistant.command
async def multiply(num1: float, num2: float) -> float:
    """
    Multiply two numbers
    """
    print(f"Multiplying {num1} * {num2}")
    return num1 * num2


async def main():
    history = []

    async for reply, history in assistant.speak(input(">>"), history):
        print(reply)


if __name__ == "__main__":
    asyncio.run(main())
```
