import contextlib
from unittest.mock import MagicMock, patch

import openai


@contextlib.contextmanager
def mocked_openai_reply(reply):
    orig = openai.ChatCompletion.create
    try:
        openai.ChatCompletion.create = MagicMock(
            return_value={"choices": [{"message": {"content": reply}}]}
        )
        yield
    finally:
        openai.ChatCompletion.create = orig


@contextlib.contextmanager
def mocked_openai_replies(replies):
    orig = openai.ChatCompletion.create
    try:
        with patch(
            "openai.ChatCompletion.create",
            side_effect=[
                {"choices": [{"message": {"content": reply}}]} for reply in replies
            ],
        ):
            yield
    finally:
        openai.ChatCompletion.create = orig
