import json

import openai

from llmio import Agent, Message

from tests.utils import mocked_async_openai_replies
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function


async def test_basics() -> None:
    agent = Agent(
        instruction="""
            You are a calculator.

            {var1} {var2}
        """,
        client=openai.AsyncOpenAI(api_key="abc"),
    )

    @agent.tool()
    async def add(num1: float, num2: float) -> float:
        return num1 + num2

    @agent.tool(strict=True)
    async def multiply(num1: float, num2: float) -> float:
        return num1 * num2

    @agent.variable
    def var1() -> str:
        return "value1"

    @agent.variable
    async def var2() -> str:
        return "value2"

    inspect_prompt_called = False

    @agent.inspect_prompt
    async def inspect_prompt(prompt: list[Message]) -> None:
        nonlocal inspect_prompt_called
        inspect_prompt_called = True
        assert (
            prompt[0]["content"]
            == """\
You are a calculator.

value1 value2"""
        )

    mocks = [
        ChatCompletionMessage.construct(
            content="Ok! I'll calculate the answer of (10 + 20) * 2",
            tool_calls=[
                ChatCompletionMessageToolCall.construct(
                    id="add_1",
                    type="function",
                    function=Function.construct(
                        name="add", arguments=json.dumps({"num1": 10, "num2": 20})
                    ),
                ),
            ],
            role="assistant",
        ),
        ChatCompletionMessage.construct(
            content=None,
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCall.construct(
                    id="multiply_1",
                    type="function",
                    function=Function.construct(
                        name="multiply", arguments=json.dumps({"num1": 30, "num2": 2})
                    ),
                ),
            ],
        ),
        ChatCompletionMessage.construct(
            role="assistant",
            content="The answer is 60",
        ),
    ]
    with mocked_async_openai_replies(mocks):
        response = await agent.speak("What is (10 + 20) * 2?")

    assert response.messages == [mocks[0].content, mocks[2].content]
    assert response.history == [
        {
            "role": "user",
            "content": "What is (10 + 20) * 2?",
        },
        agent._parse_completion(mocks[0]),
        {
            "role": "tool",
            "tool_call_id": "add_1",
            "content": "30.0",
        },
        agent._parse_completion(mocks[1]),
        {
            "role": "tool",
            "tool_call_id": "multiply_1",
            "content": "60.0",
        },
        agent._parse_completion(mocks[2]),
    ]

    assert inspect_prompt_called
    assert (
        await agent._get_instruction(context=None)
        == """\
You are a calculator.

value1 value2"""
    )


async def test_parallel_tool_calls() -> None:
    agent = Agent(
        instruction="You are a calculator",
        client=openai.AsyncOpenAI(api_key="abc"),
    )

    @agent.tool
    async def add(num1: float, num2: float) -> float:
        return num1 + num2

    @agent.tool()
    async def multiply(num1: float, num2: float) -> float:
        return num1 * num2

    mocks = [
        ChatCompletionMessage.construct(
            content="Ok! I'll calculate the answers to (10 + 20) and (3 * 9)",
            tool_calls=[
                ChatCompletionMessageToolCall.construct(
                    id="add_1",
                    type="function",
                    function=Function.construct(
                        name="add", arguments=json.dumps({"num1": 10, "num2": 20})
                    ),
                ),
                ChatCompletionMessageToolCall.construct(
                    id="multiply_1",
                    type="function",
                    function=Function.construct(
                        name="multiply", arguments=json.dumps({"num1": 3, "num2": 9})
                    ),
                ),
            ],
            role="assistant",
        ),
        ChatCompletionMessage.construct(
            role="assistant",
            content="The answer is 30 and 27",
        ),
    ]
    with mocked_async_openai_replies(mocks):
        response = await agent.speak("What is (10 + 20) and (3 * 9)?")
    assert response.messages == [mocks[0].content, mocks[1].content]
    assert response.history == [
        {
            "role": "user",
            "content": "What is (10 + 20) and (3 * 9)?",
        },
        agent._parse_completion(mocks[0]),
        {
            "role": "tool",
            "tool_call_id": "add_1",
            "content": "30.0",
        },
        {
            "role": "tool",
            "tool_call_id": "multiply_1",
            "content": "27.0",
        },
        agent._parse_completion(mocks[1]),
    ]
