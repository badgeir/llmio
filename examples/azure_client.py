import asyncio

import openai
from llmio.assistant import Assistant


assistant = Assistant(
    instruction="""
        You are a calculating assistant running in Azure.
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
    print(f"** Adding: {num1} + {num2}")
    return num1 + num2


@assistant.tool()
async def multiply(num1: float, num2: float) -> float:
    print(f"** Multiplying: {num1} * {num2}")
    return num1 * num2


@assistant.on_message
async def print_message(message: str):
    print(f"** Posting message: '{message}'")


async def main():
    history = await assistant.speak("Hi! how much is 1 + 1?")
    history = await assistant.speak("and how much is that times two?", history=history)


if __name__ == "__main__":
    asyncio.run(main())
