import contextlib
from unittest.mock import patch
from typing import Literal

from pydantic import BaseModel

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice


class FunctionCall(BaseModel):
    name: str
    arguments: str = "{}"


class OpenAIResponse(BaseModel):
    content: str | None = None
    function_call: FunctionCall | None = None
    role: Literal["assistant"] = "assistant"


@contextlib.contextmanager
def mocked_async_openai_replies(
    replies: list[OpenAIResponse],
):
    with patch(
        "llmio.Assistant.get_completion",
        side_effect=[
            ChatCompletion.construct(choices=[Choice.construct(message=reply)])
            for reply in replies
        ],
    ):
        yield replies
