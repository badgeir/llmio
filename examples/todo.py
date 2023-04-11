from typing import List
import collections

from pprint import pprint

from llmio.assistant import Assistant


def get_token():
    with open("/Users/peterleupi/.creds/openai", "r", encoding="utf-8") as f:
        return f.read().strip()


assistant = Assistant(
    short_description="You are a TODO-application conversation interface.",
    key=get_token(),
)


TODOS: dict[int, list[str]] = collections.defaultdict(list)


@assistant.command()
def get_todos(state: dict) -> List[str]:
    return TODOS[state["conversation_id"]]


@assistant.command()
def add_todo(todo: str, state: dict) -> str:
    TODOS[state["conversation_id"]].append(todo)
    return "Added todo."


def main():
    history = []
    while True:
        state = {"conversation_id": 123}
        _, history = assistant.speak(input(">>"), history=history, state=state)
        pprint(history)


if __name__ == "__main__":
    print(assistant.system_prompt())
    main()
