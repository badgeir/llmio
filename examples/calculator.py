import asyncio
import os

import openai
from llmio.agent import Agent


# Define an agent that can add and multiply numbers using tools.
# The agent will also print any messages it receives.
agent = Agent(
    # Define the agent's instructions.
    instruction="""
        You are a calculating agent.
        Always use tools to calculate things.
        Never try to calculate things on your own.
        """,
    # Pass in an OpenAI client that will be used to interact with the model.
    # Any API that implements the OpenAI interface can be used.
    client=openai.AsyncOpenAI(api_key=os.environ["OPENAI_TOKEN"]),
    model="gpt-4o-mini",
)


# Define tools using the `@agent.tool` decorator.
# Tools are automatically parsed by their type annotations
# and added to the agent's capabilities.
@agent.tool
def add(num1: float, num2: float) -> float:
    print(f"** Adding: {num1} + {num2}")
    return num1 + num2


@agent.tool
async def multiply(num1: float, num2: float) -> float:
    print(f"** Multiplying: {num1} * {num2}")
    return num1 * num2


# Define a message handler using the `@agent.on_message` decorator.
# The handler is optional. The messages will also be returned by the `speak` method.
@agent.on_message
async def print_message(message: str):
    print(f"** Posting message: '{message}'")


async def main():
    # pylint: disable=unused-variable

    # Run the agent with a message.
    # An empty history might also be passed in.
    # The agent will return the messages it generated and the updated history.
    messages, history = await agent.speak("Hi! how much is 1 + 1?")
    # The agent is stateless and does not remember previous messages.
    # The history must be passed in to maintain context.
    messages, history = await agent.speak(
        "and how much is that times two?", history=history
    )


if __name__ == "__main__":
    asyncio.run(main())
