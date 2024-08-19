import asyncio
import os
from dataclasses import dataclass
from datetime import datetime

import openai

from llmio.assistant import Assistant, Message


@dataclass
class User:
    name: str


assistant = Assistant(
    instruction=f"You are a personal assistant. You can help users set reminders. The current time is {datetime.now()}",
    client=openai.AsyncOpenAI(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
)


@assistant.tool()
async def set_reminder(description: str, datetime_iso: datetime, _state: User) -> str:
    print(
        f"** Creating reminder for user {_state.name}: '{description}' at {datetime_iso}"
    )
    return "Successfully created reminder"


@assistant.on_message
async def send_message(message: Message, _state: User) -> None:
    print(f"** Sending message to {_state.name}: '{message}'")


async def main() -> None:
    user = User(name="Alice")
    _ = await assistant.run(
        "Remind me that I need to pick up milk at the store in two hours",
        _state=user,
    )


if __name__ == "__main__":
    asyncio.run(main())
