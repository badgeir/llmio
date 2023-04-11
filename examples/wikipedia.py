from pprint import pprint
import requests
import bs4

from llmio.assistant import Assistant


def get_token():
    with open("/Users/peterleupi/.creds/openai", "r", encoding="utf-8") as f:
        return f.read().strip()


assistant = Assistant(
    short_description="""
        You are a wikipedia bot. When asked a question, you can query a wikipedia page
        and answer based on the wikipedia content.
        Never answer based on your built-in knowledge,
        always look for an answer on wikipedia.
        If the answer is not on the wikipedia page,
        politely decline to answer regardless of whether you already know it.
    """,
    key=get_token(),
)


@assistant.command()
def get_wiki(url: str) -> str:
    """
    Fetches a wikipedia page based on its url,
    and returns the textified web page content
    """

    print(url)
    answer = input("OK to fetch this page? Y/N")
    if answer == "Y":
        result = (
            bs4.BeautifulSoup(requests.get(url, timeout=2).text)
            .find(attrs={"id": "mw-content-text"})
            .get_text(separator=" ")[:4000]  # type: ignore
        )
        return result
    return "Not able to fetch page"


def main():
    history = []
    while True:
        _, history = assistant.speak(input(">>"), history=history)
        pprint(history)


if __name__ == "__main__":
    main()
