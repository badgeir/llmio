import json

import pytest

from llmio import Agent, errors, models, OpenAIClient

from tests.utils import mocked_async_openai_replies


async def test_graceful_handling() -> None:
    agent = Agent(
        instruction="You are a calculator",
        client=OpenAIClient(api_key="abc"),
        graceful_errors=True,
    )

    @agent.tool()
    async def add(num1: float, num2: float) -> float:
        return num1 + num2

    mocks = [
        models.ChatCompletionMessage.construct(
            tool_calls=[
                models.ToolCall.construct(
                    id="add_1",
                    type="function",
                    function=models.Function.construct(
                        name="add", arguments=json.dumps({"num1": 10, "num3": 20})
                    ),
                ),
            ],
            role="assistant",
        ),
        models.ChatCompletionMessage.construct(
            role="assistant",
            tool_calls=[
                models.ToolCall.construct(
                    id="divide_1",
                    type="function",
                    function=models.Function.construct(
                        name="divide", arguments=json.dumps({"num1": 30, "num2": 2})
                    ),
                ),
            ],
        ),
        models.ChatCompletionMessage.construct(
            role="assistant",
            content="Something went wrong",
        ),
    ]
    with mocked_async_openai_replies(mocks):
        response = await agent.speak("What is (10 + 20) / 2?")

    assert response.messages == ["Something went wrong"]

    assert response.history == [
        {
            "role": "user",
            "content": "What is (10 + 20) / 2?",
        },
        agent._parse_completion(mocks[0]),
        {
            "role": "tool",
            "tool_call_id": "add_1",
            "content": "The argument validation failed for the function call to add: 1 validation error for Add\nnum2\n  Field required [type=missing, input_value={'num1': 10, 'num3': 20}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.9/v/missing",
        },
        agent._parse_completion(mocks[1]),
        {
            "role": "tool",
            "tool_call_id": "divide_1",
            "content": "No tool with the name 'divide' found.",
        },
        agent._parse_completion(mocks[2]),
    ]


async def test_bad_tool_call_args_exception() -> None:
    agent = Agent(
        instruction="You are a calculator",
        client=OpenAIClient(api_key="abc"),
        graceful_errors=False,
    )

    @agent.tool()
    async def add(num1: float, num2: float) -> float:
        return num1 + num2

    mocks = [
        models.ChatCompletionMessage.construct(
            tool_calls=[
                models.ToolCall.construct(
                    id="add_1",
                    type="function",
                    function=models.Function.construct(
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


async def test_bad_tool_call_name_exception() -> None:
    agent = Agent(
        instruction="You are a calculator",
        client=OpenAIClient(api_key="abc"),
        graceful_errors=False,
    )

    @agent.tool()
    async def add(num1: float, num2: float) -> float:
        return num1 + num2

    mocks = [
        models.ChatCompletionMessage.construct(
            tool_calls=[
                models.ToolCall.construct(
                    id="add_1",
                    type="function",
                    function=models.Function.construct(
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


async def test_missing_variable() -> None:
    agent = Agent(
        instruction="{var1} {var2} {missing_var}",
        client=OpenAIClient(api_key="abc"),
        graceful_errors=False,
    )

    @agent.variable
    async def var1() -> str:
        return "var1"

    @agent.variable
    def var2() -> str:
        return "var2"

    with pytest.raises(
        errors.MissingVariable, match="Variable 'missing_var' is not defined."
    ):
        await agent.speak("Hello")
