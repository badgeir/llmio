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
def mocked_async_openai_replies(*replies: OpenAIResponse):
    async_orig = openai.ChatCompletion.acreate

    try:
        with patch(
            "openai.ChatCompletion.acreate",
            side_effect=[{"choices": [{"message": reply.dict()}]} for reply in replies],
        ):
            yield replies
    finally:
        openai.ChatCompletion.acreate = async_orig  # type: ignore


@contextlib.contextmanager
def mocked_openai_replies(*replies: OpenAIResponse):
    orig = openai.ChatCompletion.create

    try:
        with patch(
            "openai.ChatCompletion.create",
            side_effect=[{"choices": [{"message": reply.dict()}]} for reply in replies],
        ):
            yield replies
    finally:
        openai.ChatCompletion.create = orig  # type: ignore
