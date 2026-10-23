from app.providers.classification_base import ClassificationProvider
from app.schemas.classification import ClassificationLLMOutput
from app.schemas.conversation import ConversationState


class FakeClassificationProvider(ClassificationProvider):
    """
    Returns a pre-set output, or raises a pre-set error. Test-only — never
    wired to the real API. `output` may be any object (not just a valid
    ClassificationLLMOutput) so tests can simulate malformed provider output.
    """

    def __init__(
        self,
        output: ClassificationLLMOutput | object | None = None,
        error: Exception | None = None,
    ):
        self._output = output
        self._error = error
        self.calls: list[ConversationState] = []

    async def classify_ticket(self, state: ConversationState) -> ClassificationLLMOutput:
        self.calls.append(state)
        if self._error is not None:
            raise self._error
        return self._output
