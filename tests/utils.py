import contextlib
from typing import Any
from unittest.mock import patch

from llmio import types as T, models


@contextlib.contextmanager
def mocked_async_openai_replies(
    replies: list[models.ChatCompletionMessage],
):
    with patch(
        "llmio.client.BaseClient.get_chat_completion",
        side_effect=[
            models.ChatCompletion.construct(
                choices=[models.Choice.construct(message=reply)]
            )
            for reply in replies
        ],
    ):
        yield replies


@contextlib.contextmanager
def mocked_async_openai_lookup(
    replies: dict[str, models.ChatCompletionMessage],
):
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
        "llmio.client.BaseClient.get_chat_completion",
        side_effect=mock_function,
    ):
        yield replies
