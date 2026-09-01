from app.providers.clarification_base import ClarificationProvider
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState


class FakeClarificationProvider(ClarificationProvider):
    """Returns a pre-set decision. Test-only — never wired to the real API."""

    def __init__(self, decision: ClarificationDecision):
        self._decision = decision

    async def decide_clarification(self, state: ConversationState) -> ClarificationDecision:
        return self._decision