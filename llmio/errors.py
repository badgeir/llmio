class LLMIOError(Exception):
    pass


class BadToolCall(LLMIOError):
    pass
