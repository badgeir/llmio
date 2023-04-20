import json

from examples import calculator

from . import utils


def test_calculator_1():
    with utils.mocked_openai_reply("Hi! I am a calculator. How can I help you?"):
        reply, history = calculator.assistant.speak("hi")
        assert reply == "Hi! I am a calculator. How can I help you?"

    replies = [
        json.dumps({"command": "add", "params": {"num1": 10, "num2": 20}}),
        "The answer is 30!",
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
            {"role": "assistant", "content": replies[0]},
            {"role": "system", "content": json.dumps({"result": 30.0})},
            {"role": "assistant", "content": replies[1]},
        ]
