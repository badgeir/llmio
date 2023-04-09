from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field

from llmio.assistant import Assistant


assistant = Assistant(
    short_description="""
        You are a Oslo Taxis taxi booking assistant.
    """,
    key="123",
)


@assistant.command(name="SPEAK", description="The message here will be displayed to the user.")
def speak(message: str):
    print(message)


class BookTaxiParams(BaseModel):
    pickup_location: str = Field(..., description="Pickup location")
    destination: str = Field(..., description="Destination")
    number_of_people: int = Field(..., description="Number of people")
    pickup_time: datetime = Field(..., description="Pickup time")
    additional_info: Optional[str] = Field(..., description="Any additional info that might be useful for the operator or driver to know about.")


class BookTaxiResult(BaseModel):
    success: bool = Field(..., description="Pickup location")
    booking_id: Optional[str] = Field(None, description="Unique Booking ID upon successful booking.")
    message: str = Field("", description="Explanation of result")


@assistant.command(name="BOOK_TAXI", description="Execute a taxi booking. Make sure the user has confirmed the details before executing this.")
def book_taxi(params: BookTaxiParams) -> BookTaxiResult:
    print(params)


class EditBookingParams(BaseModel):
    booking_id: str = Field(..., description="Booking ID of existing booking.")
    pickup_location: str = Field(..., description="Change pickup location")
    destination: str = Field(..., description="Change destination")
    number_of_people: int = Field(..., description="Change number of people")
    pickup_time: datetime = Field(..., description="Change pickup time")
    additional_info: Optional[str] = Field(..., description="Append additional info that might be useful for the operator or driver to know about.")


class EditBookingResult(BaseModel):
    success: bool = Field(..., description="Pickup location")
    booking_id: Optional[str] = Field(None, description="Unique Booking ID upon successful booking.")
    message: str = Field("", description="Explanation of result")


@assistant.command(name="EDIT_BOOKING", description="Edit an existing taxi booking. Only not-null parameters will be written.")
def edit_booking(params: EditBookingParams) -> EditBookingResult:
    print(params)


reply = assistant.speak("Hei, jeg trenger en taxi.")
