from pprint import pprint

from llmio.assistant import Assistant


assistant = Assistant(
    short_description="""
        You are a calculating assistant.
        Always use commands to calculate things.
        Never try to calculate things on your own.
        """,
    key=open("/Users/peterleupi/.creds/openai").read().strip(),
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


history = []
while True:
    result, history = assistant.speak(
        input(">>"), history=history
    )
    pprint(history)
