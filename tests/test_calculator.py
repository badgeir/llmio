import json

from examples import calculator

from . import utils


def test_calculator_1():
    with utils.mocked_openai_reply("Hello!"):
        reply, _ = calculator.assistant.speak(
            "Hi! I am a calculator. How can I help you?"
        )
        assert reply == "Hello!"

    with utils.mocked_openai_replies(
        [
            json.dumps({"command": "add", "params": {"num1": 10, "num2": 20}}),
            "The answer is 30!",
        ]
    ):
        reply, _ = calculator.assistant.speak("calculate 10 + 20")
        assert reply == "The answer is 30!"
