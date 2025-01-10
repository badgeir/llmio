import os

import pydantic

from llmio import Message, StructuredAgent, OpenAIClient


class Booking(pydantic.BaseModel):
    from_location: str = pydantic.Field(..., description="The pickup location.")
    to_location: str = pydantic.Field(..., description="The drop-off location.")
    num_passengers: int = pydantic.Field(..., description="The number of passengers.")


class Response(pydantic.BaseModel):
    message: str = pydantic.Field(..., description="The response message.")
    booking: Booking | None = pydantic.Field(
        default=None, description="The booking details."
    )


agent = StructuredAgent(
    instruction="You are a taxi booking assistant.",
    client=OpenAIClient(
        api_key=os.environ["OPENAI_TOKEN"],
    ),
    response_format=Response,
)


@agent.on_message
def on_message(message: Response) -> None:
    print(f"{message.message}")
    if message.booking is not None:
        print(f"Booking: {message.booking}")


async def main() -> None:
    history: list[Message] = []
    while True:
        user_input = input("\n>>> ")
        response = await agent.speak(user_input, history)
        history = response.history


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
