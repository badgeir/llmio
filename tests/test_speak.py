import asyncio
from dataclasses import dataclass
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

    on_message_called_with = []

    @assistant.on_message
    async def on_message(message: str) -> None:
        on_message_called_with.append(message)
        pass

    with utils.mocked_async_openai_lookup(
        replies={
            f"Q{i}": ChatCompletionMessage.construct(role="assistant", content=f"A{i}")
            for i in range(100)
        }
    ):
        results = await asyncio.gather(*[assistant.speak(f"Q{i}") for i in range(100)])

    for i, (messages, history) in enumerate(results):
        assert history == [
            UserMessage(role="user", content=f"Q{i}"),
            AssistantMessage(role="assistant", content=f"A{i}"),
        ]
        assert messages == [f"A{i}"]

    assert sorted(on_message_called_with) == sorted([f"A{i}" for i in range(100)])


async def test_gather_run_tools() -> None:
    assistant = Assistant(
        instruction="instruction",
        client=openai.AsyncOpenAI(api_key="abc"),
        model="gpt-4o-mini",
    )

    @dataclass
    class User:
        id: int

    add_called_with = []

    @assistant.tool()
    def add(num1: int, num2: int, _state: User) -> int:
        add_called_with.append((num1, num2, User(id=_state.id)))
        return num1 + num2

    on_message_called_with = []

    @assistant.on_message
    async def on_message(message: str, _state: User) -> None:
        on_message_called_with.append((message, User(id=_state.id)))
        pass

    with utils.mocked_async_openai_lookup(
        replies={
            f"{i} + {i}?": ChatCompletionMessage.construct(
                role="assistant",
                content=f"Calculating {i} + {i}...",
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
        results = await asyncio.gather(
            *[assistant.speak(f"{i} + {i}?", _state=User(id=i)) for i in range(100)]
        )

    for i, (messages, history) in enumerate(results):
        assert history == [
            UserMessage(role="user", content=f"{i} + {i}?"),
            AssistantMessage(
                role="assistant",
                content=f"Calculating {i} + {i}...",
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
        assert messages == [f"Calculating {i} + {i}...", f"Answer: {i + i}"]

    assert sorted(add_called_with) == sorted([(i, i, User(id=i)) for i in range(100)])
    assert sorted(on_message_called_with) == sorted(
        [(f"Calculating {i} + {i}...", User(id=i)) for i in range(100)]
        + [(f"Answer: {i + i}", User(id=i)) for i in range(100)]
    )
