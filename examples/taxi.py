import asyncio
import os
from typing import Optional

from pydantic import BaseModel

import openai
from llmio.assistant import Assistant, Message


assistant = Assistant(
    instruction="You are a taxi booking assistant.",
    client=openai.AsyncOpenAI(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
)


class Result(BaseModel):
    success: bool
    booking_id: Optional[str] = None
    message: Optional[str] = None


@assistant.command()
def book_taxi(
    n_passengers: int,
    pickup_location: str,
    destination: str,
    additional_info: str | None,
) -> Result:
    """
    Execute a taxi booking order.
    Make sure the user confirms the details before executing this command.
    If the command returns success=false, it means the taxi was not successfully booked,
    and the message field should contain an explanation for why it failed.
    """
    print(
        f"""
        Booking a taxi for {n_passengers} passengers
        from {pickup_location} to {destination}.
        {additional_info if additional_info else ""}
    """
    )
    return Result(success=True, booking_id="abc123")


async def main() -> None:
    history: list[Message] = []
    while True:
        async for answer, history in assistant.speak(input(">>"), history=history):
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
