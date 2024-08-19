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


# Define a tool that sets a reminder for a user.
# The special argument _context will be passed in when detected in the function signature.
# The _context variable can be used to keep track of context, such as the current user.
# Note that the _context variable is invisible to the language model.
@assistant.tool()
async def set_reminder(description: str, datetime_iso: datetime, _context: User) -> str:
    print(
        f"** Creating reminder for user {_context.name}: '{description}' at {datetime_iso}"
    )
    return "Successfully created reminder"


# Define a message handler that sends a message to the user.
# The special argument _context will also be passed in to hooks such as @on_message.
@assistant.on_message
async def send_message(message: Message, _context: User) -> None:
    print(f"** Sending message to {_context.name}: '{message}'")


async def main() -> None:
    user = User(name="Alice")
    _ = await assistant.speak(
        "Remind me that I need to pick up milk at the store in two hours",
        _context=user,  # Pass in the user context to the assistant.
    )


if __name__ == "__main__":
    asyncio.run(main())
