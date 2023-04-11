from typing import List
import collections

from pprint import pprint

from llmio.assistant import Assistant


def get_token():
    return open("/Users/peterleupi/.creds/openai").read().strip()


assistant = Assistant(
    short_description="You are a TODO-application conversation interface.",
    key=get_token(),
)


TODOS = collections.defaultdict(list)


@assistant.command()
def get_todos(state: dict) -> List[str]:
    return TODOS[state["conversation_id"]]


@assistant.command()
def add_todo(todo: str, state: dict) -> str:
    TODOS[state["conversation_id"]].append(todo)
    return "Added todo."


history = []
while True:
    state = {"conversation_id": 123}
    result, history = assistant.speak(input(">>"), history=history, state=state)
    pprint(history)
