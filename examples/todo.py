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
def get_todos(system_params: dict) -> List[str]:
    return TODOS[system_params["conversation_id"]]


@assistant.command()
def add_todo(todo: str, system_params: dict) -> str:
    TODOS[system_params["conversation_id"]].append(todo)
    return "Added todo."


history = []
while True:
    system_params = {"conversation_id": 123}
    result, history = assistant.speak(
        input(">>"), history=history, system_params=system_params
    )
    pprint(history)
