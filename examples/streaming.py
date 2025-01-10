import os
import sys

from llmio import Message, Agent, OpenAIClient


agent = Agent(
    instruction="You are a taxi booking assistant.",
    client=OpenAIClient(
        api_key=os.environ["OPENAI_TOKEN"],
    ),
)


@agent.on_stream
def on_stream(delta: str) -> None:
    sys.stdout.write(delta)
    sys.stdout.flush()


@agent.tool
async def book_taxi(from_location: str, to_location: str, num_passengers: int) -> str:
    print(
        f"** Booking from {from_location} to {to_location} for {num_passengers} passengers."
    )
    return f"Booking confirmed from {from_location} to {to_location} for {num_passengers} passengers."


async def main() -> None:
    history: list[Message] = []
    while True:
        user_input = input("\n>>> ")
        response = await agent.speak(user_input, history, stream=True)
        history = response.history


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
