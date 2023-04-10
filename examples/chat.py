from datetime import datetime
from typing import Optional
from pprint import pprint

from pydantic import BaseModel, Field

from llmio.assistant import Assistant


assistant = Assistant(
    short_description="You are Oslo Taxis taxi booking assistant.",
    key=open("/Users/peterleupi/.creds/openai").read().strip(),
)


class BookTaxi(BaseModel):
    n_passengers: int = Field(..., ge=1, le=10)
    pickup_location: str
    destination: str
    pickup_time: datetime = Field(..., description="ISO formated time.")


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
    return Result(success=True, booking_id="abc123")


print(assistant.system_prompt())

while True:
    reply = assistant.speak(input(">>"))
    pprint(assistant.history)
