import json

from examples import calculator

from . import utils


def test_calculator_1():
    with utils.mocked_openai_reply(
        {"content": "Hi! I am a calculator. How can I help you?"}
    ):
        reply, history = calculator.assistant.speak("hi")
        assert reply == "Hi! I am a calculator. How can I help you?"

    replies = [
        {
            "content": None,
            "function_call": {
                "name": "add",
                "arguments": json.dumps({"num1": 10, "num2": 20}),
            },
        },
        {
            "content": "The answer is 30!",
        },
    ]
    with utils.mocked_openai_replies(replies):
        reply, history = calculator.assistant.speak(
            "calculate 10 + 20", history=history
        )
        assert reply == "The answer is 30!"
        assert history == [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Hi! I am a calculator. How can I help you?",
            },
            {"role": "user", "content": "calculate 10 + 20"},
            {"role": "assistant", **replies[0]},
            {
                "role": "function",
                "name": "add",
                "content": json.dumps({"result": 30.0}),
            },
            {"role": "assistant", **replies[1]},
        ]
