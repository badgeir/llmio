import asyncio
import json

import openai
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from llmio import Assistant, UserMessage, AssistantMessage, ToolCall, ToolMessage
from tests import utils


async def test_gather_run() -> None:
    assistant = Assistant(
        instruction="instruction",
        client=openai.AsyncOpenAI(api_key="abc"),
        model="gpt-4o-mini",
    )

    with utils.mocked_async_openai_lookup(
        replies={
            f"Q{i}": ChatCompletionMessage.construct(role="assistant", content=f"A{i}")
            for i in range(100)
        }
    ):
        histories = await asyncio.gather(*[assistant.run(f"Q{i}") for i in range(100)])

    for i, history in enumerate(histories):
        assert history == [
            UserMessage(role="user", content=f"Q{i}"),
            AssistantMessage(role="assistant", content=f"A{i}"),
        ]


async def test_gather_run_tools() -> None:
    assistant = Assistant(
        instruction="instruction",
        client=openai.AsyncOpenAI(api_key="abc"),
        model="gpt-4o-mini",
    )

    @assistant.tool()
    def add(num1: int, num2: int) -> int:
        return num1 + num2

    with utils.mocked_async_openai_lookup(
        replies={
            f"Q{i}": ChatCompletionMessage.construct(
                role="assistant",
                content=f"A{i}",
                tool_calls=[
                    ChatCompletionMessageToolCall.construct(
                        id=f"add_{i}",
                        type="function",
                        function=Function.construct(
                            name="add", arguments=json.dumps({"num1": i, "num2": i})
                        ),
                    )
                ],
            )
            for i in range(100)
        }
        | {
            f"{i + i}": ChatCompletionMessage.construct(
                role="assistant", content=f"Answer: {i + i}"
            )
            for i in range(100)
        }
    ):
        histories = await asyncio.gather(*[assistant.run(f"Q{i}") for i in range(100)])

    for i, history in enumerate(histories):
        assert history == [
            UserMessage(role="user", content=f"Q{i}"),
            AssistantMessage(
                role="assistant",
                content=f"A{i}",
                tool_calls=[
                    ToolCall(
                        id=f"add_{i}",
                        type="function",
                        function={
                            "name": "add",
                            "arguments": json.dumps({"num1": i, "num2": i}),
                        },
                    )
                ],
            ),
            ToolMessage(
                role="tool",
                tool_call_id=f"add_{i}",
                content=str(i + i),
            ),
            AssistantMessage(role="assistant", content=f"Answer: {i + i}"),
        ]
