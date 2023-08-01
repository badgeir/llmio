import contextlib
from unittest.mock import patch
from typing import Literal

from pydantic import BaseModel

import openai


class FunctionCall(BaseModel):
    name: str
    arguments: str = "{}"


class OpenAIResponse(BaseModel):
    content: str | None = None
    function_call: FunctionCall | None = None
    role: Literal["assistant"] = "assistant"


@contextlib.contextmanager
def mocked_openai_replies(*replies: OpenAIResponse):
    orig = openai.ChatCompletion.acreate

    try:
        with patch(
            "openai.ChatCompletion.acreate",
            side_effect=[{"choices": [{"message": reply.dict()}]} for reply in replies],
        ):
            yield replies
    finally:
        openai.ChatCompletion.acreate = orig  # type: ignore
