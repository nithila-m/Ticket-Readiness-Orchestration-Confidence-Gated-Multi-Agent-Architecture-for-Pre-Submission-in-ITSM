from abc import ABC, abstractmethod

from app.schemas.classification import ClassificationLLMOutput
from app.schemas.conversation import ConversationState


class ClassificationProvider(ABC):
    """Abstract interface for Agent 5's classify/prioritise/explain LLM call."""

    @abstractmethod
    async def classify_ticket(
        self, state: ConversationState
    ) -> ClassificationLLMOutput:
        raise NotImplementedError
