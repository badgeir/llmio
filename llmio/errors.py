class LLMIOError(Exception):
    pass


class BadToolCall(LLMIOError):
    pass


class MissingVariable(LLMIOError):
    pass
