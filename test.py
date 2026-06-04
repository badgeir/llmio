import asyncio
import os

import llmio
from llmio.clients import OpenAIClient

agent = llmio.Agent(
    instruction="An agent for testing purposes.",
    client=OpenAIClient(
        api_key=os.environ["OPENAI_TOKEN"],
    ),
)


@agent.tool
async def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    print(f"Multiplying {a} and {b}")
    return a * b


print(agent.summary())


async def main() -> None:
    response = await agent.speak("add 10 + 20")
    print(response.messages)


if __name__ == "__main__":
    asyncio.run(main())
