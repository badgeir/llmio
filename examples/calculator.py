import os

from llmio.assistant import Assistant


assistant = Assistant(
    description="""
        You are a calculating assistant.
        Always use commands to calculate things.
        Never try to calculate things on your own.
        """,
    key=os.environ["OPENAI_TOKEN"],
    debug=True,
)


@assistant.command
def add(num1: float, num2: float) -> float:
    return num1 + num2


@assistant.command
def multiply(num1: float, num2: float) -> float:
    return num1 * num2


def main():
    history = []
    while True:
        reply, history = assistant.speak(input(">>"), history=history)
        print(reply)


if __name__ == "__main__":
    main()
