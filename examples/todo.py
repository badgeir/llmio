import os
from typing import List
import collections

from pprint import pprint

from llmio.assistant import Assistant


assistant = Assistant(
    description="You are a TODO-application conversation interface.",
    key=os.environ["OPENAI_TOKEN"],
)


TODOS: dict[int, list[str]] = collections.defaultdict(list)


@assistant.command
def get_todos(state: dict) -> List[str]:
    return TODOS[state["conversation_id"]]


@assistant.command
def add_todo(todo: str, state: dict) -> str:
    TODOS[state["conversation_id"]].append(todo)
    return "Added todo."


@assistant.inspect_output
def inspect_output(output, state):
    print("Inspecting Output")
    print(output, state)


def main():
    history = []
    while True:
        state = {"conversation_id": 123}
        _, history = assistant.speak(input(">>"), history=history, state=state)
        pprint(history)


if __name__ == "__main__":
    print(assistant.system_prompt())
    main()
