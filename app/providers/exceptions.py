class LLMProviderError(Exception):
    """Raised when the LLM provider fails to return a usable structured response."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause