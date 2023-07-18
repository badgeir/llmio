import contextlib
from unittest.mock import MagicMock, patch

import openai


@contextlib.contextmanager
def mocked_openai_reply(reply):
    reply["role"] = "assistant"
    orig = openai.ChatCompletion.create
    try:
        openai.ChatCompletion.create = MagicMock(
            return_value={"choices": [{"message": reply}]}
        )
        yield
    finally:
        openai.ChatCompletion.create = orig


@contextlib.contextmanager
def mocked_openai_replies(replies):
    orig = openai.ChatCompletion.create

    for reply in replies:
        reply["role"] = "assistant"
    try:
        with patch(
            "openai.ChatCompletion.create",
            side_effect=[{"choices": [{"message": reply}]} for reply in replies],
        ):
            yield
    finally:
        openai.ChatCompletion.create = orig
