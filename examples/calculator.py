import asyncio
import os
from pprint import pprint

from llmio.assistant import Assistant


assistant = Assistant(
    description="""
        You are a calculating assistant.
        Always use commands to calculate things.
        Never try to calculate things on your own.
        """,
    key=os.environ["OPENAI_TOKEN"],
    debug=False,
    engine="gpt-4o-mini",
)


@assistant.command
def add(num1: float, num2: float) -> float:
    return num1 + num2


@assistant.command
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
