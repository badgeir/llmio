import asyncio
from dataclasses import dataclass
import json

from llmio import Agent, types as T, models, OpenAIClient

from tests import utils


async def test_gather_basic() -> None:
    agent = Agent(
        instruction="instruction",
        client=OpenAIClient(api_key="abc"),
        model="gpt-4o-mini",
    )

    on_message_called_with = set()

    @agent.on_message
    async def on_message(message: str) -> None:
        on_message_called_with.add(message)
        pass

    with utils.mocked_async_openai_lookup(
        replies={
            f"Q{i}": models.ChatCompletionMessage.construct(
                role="assistant", content=f"A{i}"
            )
            for i in range(100)
        }
    ):
        results = await asyncio.gather(*[agent.speak(f"Q{i}") for i in range(100)])

    for i, response in enumerate(results):
        assert response.history == [
            T.UserMessage(role="user", content=f"Q{i}"),
            T.AssistantMessage(role="assistant", content=f"A{i}"),
        ]
        assert response.messages == [f"A{i}"]

    assert on_message_called_with == {f"A{i}" for i in range(100)}


async def test_gather_tools() -> None:
    batch_size = 100

    agent = Agent(
        instruction="instruction for {user_name}",
        client=OpenAIClient(api_key="abc"),
        model="gpt-4o-mini",
    )

    @dataclass
    class User:
        id: int

        @property
        def name(self) -> str:
            return f"user{self.id}"

        def __hash__(self) -> int:
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

    @agent.variable
    def user_name(_context: User) -> str:
        return _context.name

    on_message_async_called_with = set()
    inspect_prompt_async_called_with = []
    inspect_output_async_called_with = []

    @agent.on_message
    async def on_message(message: str, _context: User) -> None:
        on_message_async_called_with.add((message, User(id=_context.id)))

    @agent.inspect_prompt
    async def inspect_prompt_async(prompt: list[T.Message], _context: User) -> None:
        instruction = prompt[0]["content"]
        assert prompt[0]["role"] == "system"
        assert instruction == f"instruction for {_context.name}"
        inspect_prompt_async_called_with.append(instruction)

    @agent.inspect_output
    async def inspect_output_async(message: T.Message, _context: User) -> None:
        inspect_output_async_called_with.append((message, User(id=_context.id)))

    on_message_sync_called_with = set()
    inspect_prompt_sync_called_with = []
    inspect_output_sync_called_with = []

    @agent.on_message
    def on_message_sync(message: str, _context: User) -> None:
        on_message_sync_called_with.add((message, User(id=_context.id)))

    @agent.inspect_prompt
    def inspect_prompt_sync(prompt: list[T.Message], _context: User) -> None:
        instruction = prompt[0]["content"]
        assert prompt[0]["role"] == "system"
        assert instruction == f"instruction for {_context.name}"
        inspect_prompt_sync_called_with.append(instruction)

    @agent.inspect_output
    def inspect_output_sync(message: T.Message, _context: User) -> None:
        inspect_output_sync_called_with.append((message, User(id=_context.id)))

    with utils.mocked_async_openai_lookup(
        replies={
            f"{i} + {i} and {i} * {i}?": models.ChatCompletionMessage.construct(
                role="assistant",
                content=f"Calculating {i} + {i} and {i} * {i}...",
                tool_calls=[
                    models.ToolCall.construct(
                        id=f"add_{i}",
                        type="function",
                        function=models.Function.construct(
                            name="add", arguments=json.dumps({"num1": i, "num2": i})
                        ),
                    ),
                    models.ToolCall.construct(
                        id=f"multiply_{i}",
                        type="function",
                        function=models.Function.construct(
                            name="multiply",
                            arguments=json.dumps({"num1": i, "num2": i}),
                        ),
                    ),
                ],
            )
            for i in range(batch_size)
        }
        | {
            f"mul: {i * i}": models.ChatCompletionMessage.construct(
                role="assistant", content=f"Answer: {i + i} and {i * i}"
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
            T.UserMessage(role="user", content=f"{i} + {i} and {i} * {i}?"),
            T.AssistantMessage(
                role="assistant",
                content=f"Calculating {i} + {i} and {i} * {i}...",
                tool_calls=[
                    T.ToolCall(
                        id=f"add_{i}",
                        type="function",
                        function={
                            "name": "add",
                            "arguments": json.dumps({"num1": i, "num2": i}),
                        },
                    ),
                    T.ToolCall(
                        id=f"multiply_{i}",
                        type="function",
                        function={
                            "name": "multiply",
                            "arguments": json.dumps({"num1": i, "num2": i}),
                        },
                    ),
                ],
            ),
            T.ToolMessage(
                role="tool",
                tool_call_id=f"add_{i}",
                content=f"add: {i + i}",
            ),
            T.ToolMessage(
                role="tool",
                tool_call_id=f"multiply_{i}",
                content=f"mul: {i * i}",
            ),
            T.AssistantMessage(
                role="assistant", content=f"Answer: {i + i} and {i * i}"
            ),
        ]
        assert response.messages == [
            f"Calculating {i} + {i} and {i} * {i}...",
            f"Answer: {i + i} and {i * i}",
        ]

    assert add_called_with == {(i, i, User(id=i)) for i in range(batch_size)}
    assert on_message_async_called_with == {
        (f"Calculating {i} + {i} and {i} * {i}...", User(id=i))
        for i in range(batch_size)
    } | {(f"Answer: {i + i} and {i * i}", User(id=i)) for i in range(batch_size)}

    assert mul_called_with == {(i, i) for i in range(batch_size)}
    assert set(inspect_prompt_async_called_with) == {
        f"instruction for user{i}" for i in range(batch_size)
    }
    assert len(inspect_prompt_async_called_with) == batch_size * 2
    assert len(inspect_output_async_called_with) == batch_size * 2

    assert on_message_sync_called_with == {
        (f"Calculating {i} + {i} and {i} * {i}...", User(id=i))
        for i in range(batch_size)
    } | {(f"Answer: {i + i} and {i * i}", User(id=i)) for i in range(batch_size)}

    assert set(inspect_prompt_sync_called_with) == {
        f"instruction for user{i}" for i in range(batch_size)
    }
    assert len(inspect_prompt_sync_called_with) == batch_size * 2
    assert len(inspect_output_sync_called_with) == batch_size * 2
