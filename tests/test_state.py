import json
from dataclasses import dataclass
from datetime import datetime

from llmio import Agent, models, OpenAIClient

from tests.utils import mocked_async_openai_replies


async def test_context() -> None:
    agent = Agent(
        instruction=f"You are a personal agent. You can help users set reminders. The current time is {datetime.now()}",
        client=OpenAIClient(api_key="abc"),
        model="gpt-4o-mini",
    )

    @dataclass
    class User:
        id: str
        name: str

    @agent.tool()
    async def set_reminder(
        description: str, datetime_iso: datetime, _context: User
    ) -> str:
        return f"Successfully created reminder for user {_context.id}: {description} at {datetime_iso}"

    mocks = [
        models.ChatCompletionMessage.construct(
            content="Ok! Adding a reminder.",
            tool_calls=[
                models.ToolCall.construct(
                    id="set_reminder_1",
                    type="function",
                    function=models.Function.construct(
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
        models.ChatCompletionMessage.construct(
            role="assistant",
            content="I successfully created a reminder!",
        ),
    ]

    user = User(id="1", name="Alice")
    with mocked_async_openai_replies(mocks):
        response = await agent.speak("Set a reminder for me", _context=user)
    assert response.messages == [mocks[0].content, mocks[1].content]
    assert response.history == [
        {
            "role": "user",
            "content": "Set a reminder for me",
        },
        agent._parse_completion(mocks[0]),
        {
            "role": "tool",
            "tool_call_id": "set_reminder_1",
            "content": "Successfully created reminder for user 1: A reminder at 2022-01-01 00:00:00",
        },
        agent._parse_completion(mocks[1]),
    ]
