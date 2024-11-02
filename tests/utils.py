import contextlib
from typing import Any, Iterator
from unittest.mock import patch, MagicMock

from llmio import types as T, models


@contextlib.contextmanager
def mocked_async_openai_replies(
    replies: list[models.ChatCompletionMessage],
) -> Iterator[MagicMock]:
    with patch(
        "llmio.clients.BaseClient.get_chat_completion",
        side_effect=[
            models.ChatCompletion.construct(
                choices=[models.Choice.construct(message=reply)]
            )
            for reply in replies
        ],
    ) as patched:
        yield patched


@contextlib.contextmanager
def mocked_async_openai_lookup(
    replies: dict[str, models.ChatCompletionMessage],
) -> Iterator[MagicMock]:
    def mock_function(
        model: str,
        messages: list[T.Message],
        tools: list[models.ToolCall],
        response_format: dict[str, Any] | None,
    ) -> models.ChatCompletion:
        assert isinstance(model, str)
        content = messages[-1]["content"]
        assert isinstance(content, str)
        assert isinstance(tools, list)
        for reply in replies:
            if content == reply:
                return models.ChatCompletion.construct(
                    choices=[models.Choice.construct(message=replies[reply])]
                )

        raise ValueError(f"Unexpected prompt: {content}")

    with patch(
        "llmio.clients.BaseClient.get_chat_completion",
        side_effect=mock_function,
    ) as patched:
        yield patched
