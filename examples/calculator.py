import asyncio
import os

import openai
from llmio.assistant import Assistant


assistant = Assistant(
    instruction="""
        You are a calculating assistant.
        Always use tools to calculate things.
        Never try to calculate things on your own.
        """,
    client=openai.AsyncOpenAI(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
)


@assistant.tool()
def add(num1: float, num2: float) -> float:
    print(f"** Adding: {num1} + {num2}")
    return num1 + num2


@assistant.tool()
async def multiply(num1: float, num2: float) -> float:
    print(f"** Multiplying: {num1} * {num2}")
    return num1 * num2


@assistant.on_message
async def print_message(message: str):
    print(f"** Posting message: '{message}'")


async def main():
    history = await assistant.run("Hi! how much is 1 + 1?")
    history = await assistant.run("and how much is that times two?", history=history)


if __name__ == "__main__":
    asyncio.run(main())
