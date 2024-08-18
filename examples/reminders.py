import asyncio
import os
from dataclasses import dataclass
from datetime import datetime

import openai

from llmio.assistant import Assistant, Message


assistant = Assistant(
    instruction=f"You are a personal assistant. You can help users set reminders. The current time is {datetime.now()}",
    client=openai.AsyncOpenAI(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
)


@dataclass
class User:
    id: str
    name: str


@assistant.command()
async def set_reminder(description: str, datetime_iso: datetime, _state: User) -> str:
    print(
        f"set_reminder(): Creating reminder for user {_state.id}: {description} at {datetime_iso}"
    )
    return "Successfully created reminder"


async def main() -> None:
    user = User(id="1", name="Alice")
    history: list[Message] = []

    async for message, history in assistant.speak(
        "Remind me that I need to pick up milk at the store in two hours",
        _state=user,
        history=history,
    ):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
