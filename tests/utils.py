import contextlib
from unittest.mock import patch

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice


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
