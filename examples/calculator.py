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
)


@assistant.command
def add(num1: float, num2: float) -> float:
    """
    Add two numbers.
    """
    return num1 + num2


@assistant.command
def multiply(num1: float, num2: float) -> float:
    """
    Multiply two numbers.
    """
    return num1 * num2


@assistant.inspect_prompt
def inspect_prompt(prompt):
    print("Inspecting prompt")
    pprint(prompt)


@assistant.inspect_output
def inspect_output(output):
    print("Inspecting Output")
    pprint(output)


def main():
    history = []
    while True:
        _, history = assistant.speak(input(">>"), history=history)
        pprint(history)


if __name__ == "__main__":
    main()
