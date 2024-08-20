import json

import openai

from llmio import Assistant

from tests.utils import mocked_async_openai_replies
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function


async def test_basics() -> None:
    assistant = Assistant(
        instruction="You are a calculator",
        client=openai.AsyncOpenAI(api_key="abc"),
    )

    @assistant.tool()
    async def add(num1: float, num2: float) -> float:
        return num1 + num2

    mocks = [
        ChatCompletionMessage.construct(
            tool_calls=[
                ChatCompletionMessageToolCall.construct(
                    id="add_1",
                    type="function",
                    function=Function.construct(
                        name="add", arguments=json.dumps({"num1": 10, "num3": 20})
                    ),
                ),
            ],
            role="assistant",
        ),
        ChatCompletionMessage.construct(
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCall.construct(
                    id="divide_1",
                    type="function",
                    function=Function.construct(
                        name="divide", arguments=json.dumps({"num1": 30, "num2": 2})
                    ),
                ),
            ],
        ),
        ChatCompletionMessage.construct(
            role="assistant",
            content="Something went wrong",
        ),
    ]
    with mocked_async_openai_replies(mocks):
        answers, history = await assistant.speak("What is (10 + 20) / 2?")

    assert answers == ["Something went wrong"]

    assert history == [
        {
            "role": "user",
            "content": "What is (10 + 20) / 2?",
        },
        assistant._parse_completion(mocks[0]),
        {
            "role": "tool",
            "tool_call_id": "add_1",
            "content": "The argument validation failed for the function call to add: 1 validation error for Add\nnum2\n  field required (type=value_error.missing)",
        },
        assistant._parse_completion(mocks[1]),
        {
            "role": "tool",
            "tool_call_id": "divide_1",
            "content": "No tool with the name 'divide' found.",
        },
        assistant._parse_completion(mocks[2]),
    ]
