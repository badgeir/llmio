import os
from pprint import pprint
from typing import Any

from llmio.assistant import Assistant

# pylint: disable
assistant = Assistant(
    description="""
        You are a calculating assistant.
        Always use commands to calculate things.
        Never try to calculate things on your own.
        """,
    key=os.environ["OPENAI_TOKEN"],
    debug=False,
    engine="gpt-3.5-turbo",
)


@assistant.command
def add(num1: float, num2: float) -> float:
    return num1 + num2


@assistant.command
def multiply(num1: float, num2: float) -> float:
    return num1 * num2


@assistant.inspect_prompt
def print_prompt(prompt: list[dict[str, str]]):
    pprint(prompt)


@assistant.inspect_output
def print_model_output(output: dict):
    pprint(output)


if __name__ == "__main__":
    history: list[dict[str, Any]] = []
    while True:
        for answer, history in assistant.speak(input(">>"), history=history):
            print(answer)
