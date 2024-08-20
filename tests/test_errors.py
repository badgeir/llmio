import json

import openai
import pytest

from llmio import Agent, errors

from tests.utils import mocked_async_openai_replies
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function


async def test_basics() -> None:
    agent = Agent(
        instruction="You are a calculator",
        client=openai.AsyncOpenAI(api_key="abc"),
    )

    @agent.tool()
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
        answers, history = await agent.speak("What is (10 + 20) / 2?")

    assert answers == ["Something went wrong"]

    assert history == [
        {
            "role": "user",
            "content": "What is (10 + 20) / 2?",
        },
        agent._parse_completion(mocks[0]),
        {
            "role": "tool",
            "tool_call_id": "add_1",
            "content": "The argument validation failed for the function call to add: 1 validation error for Add\nnum2\n  field required (type=value_error.missing)",
        },
        agent._parse_completion(mocks[1]),
        {
            "role": "tool",
            "tool_call_id": "divide_1",
            "content": "No tool with the name 'divide' found.",
        },
        agent._parse_completion(mocks[2]),
    ]


async def test_bad_tool_call_args() -> None:
    agent = Agent(
        instruction="You are a calculator",
        client=openai.AsyncOpenAI(api_key="abc"),
        graceful_errors=False,
    )

    @agent.tool()
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
    ]
    with mocked_async_openai_replies(mocks):
        with pytest.raises(errors.BadToolCall):
            await agent.speak("What is (10 + 20) / 2?")


async def test_bad_tool_call_name() -> None:
    agent = Agent(
        instruction="You are a calculator",
        client=openai.AsyncOpenAI(api_key="abc"),
        graceful_errors=False,
    )

    @agent.tool()
    async def add(num1: float, num2: float) -> float:
        return num1 + num2

    mocks = [
        ChatCompletionMessage.construct(
            tool_calls=[
                ChatCompletionMessageToolCall.construct(
                    id="add_1",
                    type="function",
                    function=Function.construct(
                        name="mul", arguments=json.dumps({"num1": 10, "num2": 20})
                    ),
                ),
            ],
            role="assistant",
        ),
    ]
    with mocked_async_openai_replies(mocks):
        with pytest.raises(errors.BadToolCall):
            await agent.speak("What is (10 + 20) / 2?")
