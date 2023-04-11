from datetime import datetime
from typing import Optional
from pprint import pprint

from pydantic import BaseModel, Field

from llmio.assistant import Assistant


def get_token():
    with open("/Users/peterleupi/.creds/openai", "r", encoding="utf-8") as f:
        return f.read().strip()


assistant = Assistant(
    short_description="You are Oslo Taxis taxi booking assistant.",
    key=get_token(),
)


class BookTaxi(BaseModel):
    n_passengers: int = Field(..., ge=1, le=10)
    pickup_location: str
    destination: str
    pickup_time: datetime = Field(..., description="ISO formated time.")
    additional_info: Optional[str]


class Result(BaseModel):
    success: bool
    booking_id: Optional[str] = None
    message: Optional[str] = None


@assistant.command()
def book_taxi(params: BookTaxi) -> Result:
    """
    Execute a taxi booking order.
    Make sure the user confirms the details before executing this command.
    If the command returns success=false, it means the taxi was not successfully booked,
    and the message field should contain an explanation for why it failed.
    """
    print("Booking taxi:", params)
    return Result(success=True, booking_id="abc123")


def main():
    history = []
    while True:
        _, history = assistant.speak(input(">>"), history=history)
        pprint(history)


if __name__ == "__main__":
    main()
