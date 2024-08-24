"""

See https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library
for full instructions.

Short summary of setup:
1. install and authenticate with Google Cloud CLI
   https://cloud.google.com/sdk/docs/install-sdk
2. Set up environment variable for authenticating with Gemini API:
   export GOOGLE_GEMINI_TOKEN=$(gcloud auth application-default print-access-token)

"""

import asyncio
import os

import openai

from llmio import Agent


# Available locations:
# https://cloud.google.com/gemini/docs/locations
LOCATION = "your_gemini_location"  # e.g. "europe-west1"

PROJECT_ID = "your_project_id"

BASE_URL = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi"  # pylint: disable=line-too-long


gemini_client = openai.AsyncOpenAI(
    api_key=os.environ["GOOGLE_GEMINI_TOKEN"],
    base_url=BASE_URL,
)

agent = Agent(
    instruction="""
        You are a calculating agent.
        Always use tools to calculate things.
        Never try to calculate things on your own.
        """,
    client=gemini_client,
    model="google/gemini-1.5-pro-001",
)


@agent.tool
def add(num1: float, num2: float) -> float:
    print(f"** Executing add({num1}, {num2}) -> {num1 + num2}")
    return num1 + num2


@agent.tool
async def multiply(num1: float, num2: float) -> float:
    print(f"** Executing multiply({num1}, {num2}) -> {num1 * num2}")
    return num1 * num2


@agent.on_message
async def print_message(message: str):
    print(f"** Posting message: '{message.strip()}'")


async def main() -> None:
    response = await agent.speak("Hi! how much is 1 + 1?")
    await agent.speak("and how much is that times two?", history=response.history)


if __name__ == "__main__":
    asyncio.run(main())
