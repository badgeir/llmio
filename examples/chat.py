from pydantic import BaseModel

from llmio.assistant import Assistant


assistant = Assistant(
    short_description="""
        You are a calculator.
        Always use the provided commands to calculate things,
        do not attemt to calculate things yourself.
    """,
    key=open("/Users/peterleupi/.creds/openai").read().strip(),
)


class NumberPair(BaseModel):
    number1: float
    number2: float


class Result(BaseModel):
    result: float


@assistant.command(description="Add two numbers")
def add(params: NumberPair) -> Result:
    print(f"Add {params}")
    return Result(result=params.number1 + params.number2)


@assistant.command(description="Multiply two numbers")
def multiply(params: NumberPair) -> Result:
    print(f"Multiply {params}")
    return Result(result=params.number1 * params.number2)


reply = assistant.speak("How much is ((10 * 2) + (100 * 12345)) * 2?")
print(reply)
