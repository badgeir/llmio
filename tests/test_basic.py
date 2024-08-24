import json
from typing import Iterable

import pytest

from openai import AsyncOpenAI, AsyncAzureOpenAI

from llmio import Agent, models, types as T
from llmio.clients import BaseClient, OpenAIClient, AzureOpenAIClient, GeminiClient

from tests.utils import mocked_async_openai_replies


def clients() -> Iterable[BaseClient | AsyncOpenAI]:
    yield OpenAIClient(api_key="abc")
    yield AzureOpenAIClient(
        api_key="abc", endpoint="http://localhost:8000", api_version="2024.01.01"
    )
    yield GeminiClient(api_key="abc", base_url="http://localhost:8000")
    yield AsyncOpenAI(api_key="abc")
    yield AsyncAzureOpenAI(
        api_key="abc", azure_endpoint="http://localhost:8000", api_version="2024.01.01"
    )


@pytest.mark.parametrize("client", clients())
async def test_basics(client: BaseClient) -> None:
    agent = Agent(
        instruction="""
            You are a calculator.

            {var1} {var2}
        """,
        client=client,
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
    async def inspect_prompt(prompt: list[T.Message]) -> None:
        nonlocal inspect_prompt_called
        inspect_prompt_called = True
        assert (
            prompt[0]["content"]
            == """\
You are a calculator.

value1 value2"""
        )

    mocks = [
        models.ChatCompletionMessage.construct(
            content="Ok! I'll calculate the answer of (10 + 20) * 2",
            tool_calls=[
                models.ToolCall.construct(
                    id="add_1",
                    type="function",
                    function=models.Function.construct(
                        name="add", arguments=json.dumps({"num1": 10, "num2": 20})
                    ),
                ),
            ],
            role="assistant",
        ),
        models.ChatCompletionMessage.construct(
            content=None,
            role="assistant",
            tool_calls=[
                models.ToolCall.construct(
                    id="multiply_1",
                    type="function",
                    function=models.Function.construct(
                        name="multiply", arguments=json.dumps({"num1": 30, "num2": 2})
                    ),
                ),
            ],
        ),
        models.ChatCompletionMessage.construct(
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
        client=OpenAIClient(api_key="abc"),
    )

    @agent.tool
    async def add(num1: float, num2: float) -> float:
        return num1 + num2

    @agent.tool()
    async def multiply(num1: float, num2: float) -> float:
        return num1 * num2

    mocks = [
        models.ChatCompletionMessage.construct(
            content="Ok! I'll calculate the answers to (10 + 20) and (3 * 9)",
            tool_calls=[
                models.ToolCall.construct(
                    id="add_1",
                    type="function",
                    function=models.Function.construct(
                        name="add", arguments=json.dumps({"num1": 10, "num2": 20})
                    ),
                ),
                models.ToolCall.construct(
                    id="multiply_1",
                    type="function",
                    function=models.Function.construct(
                        name="multiply", arguments=json.dumps({"num1": 3, "num2": 9})
                    ),
                ),
            ],
            role="assistant",
        ),
        models.ChatCompletionMessage.construct(
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
