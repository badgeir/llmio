from pprint import pprint
import requests
import bs4

from llmio.assistant import Assistant


def get_token():
    return open("/Users/peterleupi/.creds/openai").read().strip()


assistant = Assistant(
    short_description="""
        You are a wikipedia bot. When asked a question, you can query a wikipedia page and answer based on the wikipedia content.
        Never answer based on your built-in knowledge, always look for an answer on wikipedia.
        If the answer is not on the wikipedia page, politely decline to answer regardless of whether you already know it.
    """,
    key=get_token(),
)


@assistant.command()
def get_wiki(url: str) -> str:
    print(url)
    answer = input("OK to fetch this page? Y/N")
    if answer == "Y":
        result = (
            bs4.BeautifulSoup(requests.get(url).text)
            .find(attrs={"id": "mw-content-text"})
            .get_text(separator=" ")[:4000]
        )
        return result
    return "Not able to fetch page"


history = []
while True:
    result, history = assistant.speak(input(">>"), history=history)
    pprint(history)
