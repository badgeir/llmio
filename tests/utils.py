import contextlib
from unittest.mock import patch

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from llmio import Message


@contextlib.contextmanager
def mocked_async_openai_replies(
    replies: list[ChatCompletionMessage],
):
    with patch(
        "llmio.Assistant._get_completion",
        side_effect=[
            ChatCompletion.construct(choices=[Choice.construct(message=reply)])
            for reply in replies
        ],
    ):
        yield replies


@contextlib.contextmanager
def mocked_async_openai_lookup(
    replies: dict[str, ChatCompletionMessage],
):
    def side_effect(messages: list[Message]) -> ChatCompletion:
        content = messages[-1]["content"]
        assert isinstance(content, str)
        for reply in replies:
            if content == reply:
                return ChatCompletion.construct(
                    choices=[Choice.construct(message=replies[reply])]
                )

        raise ValueError(f"Unexpected prompt: {content}")

    with patch(
        "llmio.Assistant._get_completion",
        side_effect=side_effect,
    ):
        yield replies
