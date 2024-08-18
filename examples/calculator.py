import asyncio
import os
from pprint import pprint

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
    return num1 + num2


@assistant.tool()
async def multiply(num1: float, num2: float) -> float:
    return num1 * num2


@assistant.inspect_prompt
def print_prompt(prompt: list[dict[str, str]]):
    pprint(prompt)


@assistant.inspect_output
def print_model_output(output: dict):
    pprint(output)


async def main():
    while True:
        async for answer, _ in assistant.speak(input(">>")):
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
