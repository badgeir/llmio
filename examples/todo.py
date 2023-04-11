from typing import List

from pprint import pprint

from llmio.assistant import Assistant


def get_token():
    return open("/Users/peterleupi/.creds/openai").read().strip()


assistant = Assistant(
    short_description="You are a TODO-application conversation interface.",
    key=get_token(),
)


TODOS = []


@assistant.command()
def get_todos() -> List[str]:
    print("Get todos!")
    return TODOS


@assistant.command()
def add_todo(todo: str) -> str:
    print("Add todo!")
    TODOS.append(todo)
    return "Added todo."


while True:
    result = assistant.speak(input(">>"))
    pprint(assistant.history)
