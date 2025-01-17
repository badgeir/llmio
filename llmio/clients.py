from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI, AsyncAzureOpenAI, AsyncStream
from openai.types.shared_params import ResponseFormatJSONSchema
from llmio.models import ChatCompletionChunk

from llmio import types as T, models


class BaseClient:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def get_chat_completion(
        self,
        model: str,
        messages: list[T.Message],
        tools: list[T.Tool],
        response_format: ResponseFormatJSONSchema | None,
    ) -> models.ChatCompletion:
        kwargs: dict[str, Any] = {}
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools
        return await self._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

    async def stream_chat_completion(
        self,
        model: str,
        messages: list[T.Message],
        tools: list[T.Tool],
        response_format: ResponseFormatJSONSchema | None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        kwargs: dict[str, Any] = {}
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools
        stream: AsyncStream[ChatCompletionChunk] = (
            await self._client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs,
            )
        )
        async for chunk in stream:
            assert isinstance(chunk, ChatCompletionChunk)
            yield chunk


class OpenAIClient(BaseClient):
    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        super().__init__(client=client)


class AzureOpenAIClient(BaseClient):
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
    ) -> None:
        client = AsyncAzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version=api_version
        )
        super().__init__(client=client)


class GeminiClient(BaseClient):
    def __init__(self, api_key: str, base_url: str) -> None:
        assert base_url is not None
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        super().__init__(client=client)
