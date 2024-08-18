import json
from dataclasses import dataclass
from datetime import datetime

import openai

from llmio import Assistant, Message

from tests.utils import mocked_async_openai_replies
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function


async def test_state() -> None:
    assistant = Assistant(
        instruction=f"You are a personal assistant. You can help users set reminders. The current time is {datetime.now()}",
        client=openai.AsyncOpenAI(api_key="abc"),
        model="gpt-4o-mini",
    )

    @dataclass
    class User:
        id: str
        name: str

    @assistant.tool()
    async def set_reminder(
        description: str, datetime_iso: datetime, _state: User
    ) -> str:
        return f"Successfully created reminder for user {_state.id}: {description} at {datetime_iso}"

    mocks = [
        ChatCompletionMessage.construct(
            content="Ok! Adding a reminder.",
            tool_calls=[
                ChatCompletionMessageToolCall.construct(
                    id="set_reminder_1",
                    type="function",
                    function=Function.construct(
                        name="set_reminder",
                        arguments=json.dumps(
                            {
                                "description": "A reminder",
                                "datetime_iso": "2022-01-01T00:00:00",
                            }
                        ),
                    ),
                ),
            ],
            role="assistant",
        ),
        ChatCompletionMessage.construct(
            role="assistant",
            content="I successfully created a reminder!",
        ),
    ]

    user = User(id="1", name="Alice")
    answers = []
    history: list[Message] = []
    with mocked_async_openai_replies(mocks):
        async for answer, history in assistant.speak(
            "Set a reminder for me", history=history, _state=user
        ):
            answers.append(answer)
    assert answers == [mocks[0].content, mocks[1].content]
    assert history == [
        {
            "role": "user",
            "content": "Set a reminder for me",
        },
        assistant._parse_completion(mocks[0]),
        {
            "role": "tool",
            "tool_call_id": "set_reminder_1",
            "content": json.dumps(
                {
                    "result": "Successfully created reminder for user 1: A reminder at 2022-01-01 00:00:00"
                }
            ),
        },
        assistant._parse_completion(mocks[1]),
    ]
