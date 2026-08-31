from abc import ABC, abstractmethod

from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState


class ClarificationProvider(ABC):
    """Abstract interface for Agent 2's decision-making LLM call."""

    @abstractmethod
    async def decide_clarification(
        self, state: ConversationState
    ) -> ClarificationDecision:
        raise NotImplementedError