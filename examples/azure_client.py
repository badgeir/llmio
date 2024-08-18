import asyncio

import openai
from llmio.assistant import Assistant


assistant = Assistant(
    instruction="""
        You are a calculating assistant.
        Always use tools to calculate things.
        Never try to calculate things on your own.
        """,
    client=openai.AsyncAzureOpenAI(
        api_key="<your-api-key",
        azure_endpoint="<your-azure-endpoint>",
        api_version="<your-api-version>",
    ),
    model="<your-deployment>",
)


@assistant.tool()
def add(num1: float, num2: float) -> float:
    return num1 + num2


@assistant.tool()
async def multiply(num1: float, num2: float) -> float:
    return num1 * num2


async def main():
    while True:
        async for answer, _ in assistant.speak(input(">>")):
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
