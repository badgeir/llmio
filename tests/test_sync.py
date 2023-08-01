import json

from llmio import Assistant

from tests.utils import mocked_openai_replies, OpenAIResponse, FunctionCall


async def test_sync_basics():
    assistant = Assistant(key="abc", description="You are a calculator")

    @assistant.command
    def add(num1: float, num2: float) -> float:
        return num1 + num2

    @assistant.command
    def multiply(num1: float, num2: float) -> float:
        return num1 * num2

    mocks = [
        OpenAIResponse(
            content="Ok! I'll calculate the answer of (10 + 20) * 2",
            function_call=FunctionCall(
                name="add", arguments=json.dumps({"num1": 10, "num2": 20})
            ),
            role="assistant",
        ),
        OpenAIResponse(
            content=None,
            function_call=FunctionCall(
                name="multiply",
                arguments=json.dumps({"num1": 30, "num2": 2}),
                role="assistant",
            ),
        ),
        OpenAIResponse(
            content="The answer is 60",
        ),
    ]
    answers = []
    history = []
    with mocked_openai_replies(*mocks):
        for answer, history in assistant.speak(
            "What is (10 + 20) * 2?", history=history
        ):
            answers.append(answer)

    assert answers == [mocks[0].content, mocks[2].content]
    assert history == [
        {
            "role": "user",
            "content": "What is (10 + 20) * 2?",
        },
        mocks[0].dict(),
        {"role": "function", "name": "add", "content": json.dumps({"result": 30.0})},
        mocks[1].dict(),
        {
            "role": "function",
            "name": "multiply",
            "content": json.dumps({"result": 60.0}),
        },
        mocks[2].dict(),
    ]
