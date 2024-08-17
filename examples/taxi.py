import asyncio
import os
from typing import Optional

from pydantic import BaseModel

from llmio.assistant import Assistant


assistant = Assistant(
    description="You are Oslo Taxis taxi booking assistant.",
    key=os.environ["OPENAI_TOKEN"],
    debug=True,
)


class Result(BaseModel):
    success: bool
    booking_id: Optional[str] = None
    message: Optional[str] = None


@assistant.command
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


async def main():
    history = []
    while True:
        async for answer, history in assistant.speak(input(">>"), history=history):
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
