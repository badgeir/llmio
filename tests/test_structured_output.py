import asyncio
from dataclasses import dataclass
import json
from pydantic import BaseModel

import openai
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from llmio import (
    StructuredAgent,
    Message,
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolMessage,
)
from tests import utils


async def test_gather_structured_output_with_tools() -> None:
    batch_size = 100

    class OutputFormat(BaseModel):
        message: str
        i: int

    agent = StructuredAgent(
        instruction="instruction",
        client=openai.AsyncOpenAI(api_key="abc"),
        model="gpt-4o-mini",
        response_format=OutputFormat,
    )

    @dataclass
    class User:
        id: int

        def __hash__(self):
            return hash(self.id)

    add_called_with = set()
    mul_called_with = set()

    @agent.tool()
    def add(num1: int, num2: int, _context: User) -> str:
        add_called_with.add((num1, num2, User(id=_context.id)))
        return f"add: {num1 + num2}"

    @agent.tool()
    async def multiply(num1: int, num2: int) -> str:
        mul_called_with.add((num1, num2))
        return f"mul: {num1 * num2}"

    on_message_async_called_with = set()
    inspect_prompt_async_called_with = []
    inspect_output_async_called_with = []

    @agent.on_message
    async def on_message(message: OutputFormat, _context: User) -> None:
        on_message_async_called_with.add((message.message, User(id=_context.id)))

    @agent.inspect_prompt
    async def inspect_prompt_async(prompt: list[Message], _context: User) -> None:
        inspect_prompt_async_called_with.append((prompt, User(id=_context.id)))

    @agent.inspect_output
    async def inspect_output_async(message: Message, _context: User) -> None:
        inspect_output_async_called_with.append((message, User(id=_context.id)))

    on_message_sync_called_with = set()
    inspect_prompt_sync_called_with = []
    inspect_output_sync_called_with = []

    @agent.on_message
    def on_message_sync(message: OutputFormat, _context: User) -> None:
        on_message_sync_called_with.add((message.message, User(id=_context.id)))

    @agent.inspect_prompt
    def inspect_prompt_sync(prompt: list[Message], _context: User) -> None:
        inspect_prompt_sync_called_with.append((prompt, User(id=_context.id)))

    @agent.inspect_output
    def inspect_output_sync(message: Message, _context: User) -> None:
        inspect_output_sync_called_with.append((message, User(id=_context.id)))

    with utils.mocked_async_openai_lookup(
        replies={
            f"{i} + {i} and {i} * {i}?": ChatCompletionMessage.construct(
                role="assistant",
                content=json.dumps(
                    {
                        "message": f"Calculating {i} + {i} and {i} * {i}...",
                        "i": i,
                    }
                ),
                tool_calls=[
                    ChatCompletionMessageToolCall.construct(
                        id=f"add_{i}",
                        type="function",
                        function=Function.construct(
                            name="add", arguments=json.dumps({"num1": i, "num2": i})
                        ),
                    ),
                    ChatCompletionMessageToolCall.construct(
                        id=f"multiply_{i}",
                        type="function",
                        function=Function.construct(
                            name="multiply",
                            arguments=json.dumps({"num1": i, "num2": i}),
                        ),
                    ),
                ],
            )
            for i in range(batch_size)
        }
        | {
            f"mul: {i * i}": ChatCompletionMessage.construct(
                role="assistant",
                content=json.dumps(
                    {
                        "message": f"Answer: {i + i} and {i * i}",
                        "i": i,
                    }
                ),
            )
            for i in range(batch_size)
        }
    ):
        results = await asyncio.gather(
            *[
                agent.speak(f"{i} + {i} and {i} * {i}?", _context=User(id=i))
                for i in range(batch_size)
            ]
        )

    for i, response in enumerate(results):
        assert response.history == [
            UserMessage(role="user", content=f"{i} + {i} and {i} * {i}?"),
            AssistantMessage(
                role="assistant",
                content=json.dumps(
                    {
                        "message": f"Calculating {i} + {i} and {i} * {i}...",
                        "i": i,
                    }
                ),
                tool_calls=[
                    ToolCall(
                        id=f"add_{i}",
                        type="function",
                        function={
                            "name": "add",
                            "arguments": json.dumps({"num1": i, "num2": i}),
                        },
                    ),
                    ToolCall(
                        id=f"multiply_{i}",
                        type="function",
                        function={
                            "name": "multiply",
                            "arguments": json.dumps({"num1": i, "num2": i}),
                        },
                    ),
                ],
            ),
            ToolMessage(
                role="tool",
                tool_call_id=f"add_{i}",
                content=f"add: {i + i}",
            ),
            ToolMessage(
                role="tool",
                tool_call_id=f"multiply_{i}",
                content=f"mul: {i * i}",
            ),
            AssistantMessage(
                role="assistant",
                content=json.dumps(
                    {
                        "message": f"Answer: {i + i} and {i * i}",
                        "i": i,
                    }
                ),
            ),
        ]
        assert response.messages == [
            OutputFormat(message=f"Calculating {i} + {i} and {i} * {i}...", i=i),
            OutputFormat(message=f"Answer: {i + i} and {i * i}", i=i),
        ]

    assert add_called_with == {(i, i, User(id=i)) for i in range(batch_size)}
    assert on_message_async_called_with == {
        (f"Calculating {i} + {i} and {i} * {i}...", User(id=i))
        for i in range(batch_size)
    } | {(f"Answer: {i + i} and {i * i}", User(id=i)) for i in range(batch_size)}

    assert mul_called_with == {(i, i) for i in range(batch_size)}
    assert len(inspect_prompt_async_called_with) == batch_size * 2
    assert len(inspect_output_async_called_with) == batch_size * 2

    assert on_message_sync_called_with == {
        (f"Calculating {i} + {i} and {i} * {i}...", User(id=i))
        for i in range(batch_size)
    } | {(f"Answer: {i + i} and {i * i}", User(id=i)) for i in range(batch_size)}

    assert len(inspect_prompt_sync_called_with) == batch_size * 2
    assert len(inspect_output_sync_called_with) == batch_size * 2
