from pprint import pprint

from llmio.assistant import Assistant


def get_token():
    with open("/Users/peterleupi/.creds/openai", "r", encoding="utf-8") as f:
        return f.read().strip()


assistant = Assistant(
    short_description="""
        You are a calculating assistant.
        Always use commands to calculate things.
        Never try to calculate things on your own.
        """,
    key=get_token(),
)


@assistant.command()
def add(num1: float, num2: float) -> float:
    """
    Add two numbers.
    """
    return num1 + num2


@assistant.command()
def multiply(num1: float, num2: float) -> float:
    """
    Multiply two numbers.
    """
    return num1 * num2


def main():
    history = []
    while True:
        _, history = assistant.speak(input(">>"), history=history)
        pprint(history)


if __name__ == "__main__":
    main()
