import asyncio
import os
from dataclasses import dataclass
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


@dataclass
class User:
    id: int


@assistant.tool()
def add(num1: float, num2: float, _state: User) -> float:
    print(f"** {_state.id}: Adding {num1} and {num2}**")
    return num1 + num2


@assistant.tool()
async def multiply(num1: float, num2: float) -> float:
    return num1 * num2


@assistant.on_message
async def print_message(message: str, _state: User):
    print(f"{_state.id}: {message}")


async def main():
    histories = await asyncio.gather(
        *[
            assistant.run(f"Hi! how much is {i} + {i}?", _state=User(id=i))
            for i in range(10)
        ]
    )
    for i, history in enumerate(histories):
        print(f"User {i}")
        pprint(history)
        print()


if __name__ == "__main__":
    asyncio.run(main())
