import openai.types.chat.chat_completion_message_tool_call_param
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import FunctionDefinition as FunctionDefinition_

Message = ChatCompletionMessageParam
Tool = ChatCompletionToolParam
ToolMessage = ChatCompletionToolMessageParam
AssistantMessage = ChatCompletionAssistantMessageParam
UserMessage = ChatCompletionUserMessageParam
SystemMessage = ChatCompletionSystemMessageParam
ToolCall = ChatCompletionMessageToolCallParam
ToolCallFunction = openai.types.chat.chat_completion_message_tool_call_param.Function
FunctionDefinition = FunctionDefinition_


__all__ = [
    "Message",
    "Tool",
    "ToolMessage",
    "AssistantMessage",
    "UserMessage",
    "SystemMessage",
    "ToolCall",
    "ToolCallFunction",
    "FunctionDefinition",
]
