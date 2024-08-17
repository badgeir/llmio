import json

from llmio import Assistant

from tests.utils import mocked_async_openai_replies
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)


async def test_async_basics():
    assistant = Assistant(key="abc", instruction="You are a calculator")

    @assistant.command
    async def add(num1: float, num2: float) -> float:
        return num1 + num2

    @assistant.command
    async def multiply(num1: float, num2: float) -> float:
        return num1 * num2

    mocks = [
        ChatCompletionMessage.construct(
            content="Ok! I'll calculate the answer of (10 + 20) * 2",
            function_call=FunctionCall.construct(
                name="add", arguments=json.dumps({"num1": 10, "num2": 20})
            ),
            role="assistant",
        ),
        ChatCompletionMessage.construct(
            content=None,
            role="assistant",
            function_call=FunctionCall.construct(
                name="multiply",
                arguments=json.dumps({"num1": 30, "num2": 2}),
            ),
        ),
        ChatCompletionMessage.construct(
            role="assistant",
            content="The answer is 60",
        ),
    ]
    answers = []
    history = []
    with mocked_async_openai_replies(mocks):
        async for answer, history in assistant.speak(
            "What is (10 + 20) * 2?", history=history
        ):
            answers.append(answer)
    assert answers == [mocks[0].content, mocks[2].content]
    assert history == [
        {
            "role": "user",
            "content": "What is (10 + 20) * 2?",
        },
        assistant.parse_completion(mocks[0]),
        {"role": "function", "name": "add", "content": json.dumps({"result": 30.0})},
        assistant.parse_completion(mocks[1]),
        {
            "role": "function",
            "name": "multiply",
            "content": json.dumps({"result": 60.0}),
        },
        assistant.parse_completion(mocks[2]),
    ]
