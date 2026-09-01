"""
Abstract persistence interface for ConversationState.

Kept deliberately storage-agnostic: an in-memory implementation is enough
for the Agent 2 milestone and local dev. SQLite/Postgres can implement
this same interface later without touching any code that depends on it
(clarification_service.py will depend on this ABC, not on a concrete store).
"""

from abc import ABC, abstractmethod

from app.schemas.conversation import ConversationState


class ConversationRepository(ABC):
    @abstractmethod
    async def get(self, conversation_id: str) -> ConversationState | None:
        """Return the stored state, or None if it doesn't exist yet."""
        raise NotImplementedError

    @abstractmethod
    async def save(self, state: ConversationState) -> None:
        """Persist (create or overwrite) the given state."""
        raise NotImplementedError

    async def get_or_create(self, conversation_id: str) -> ConversationState:
        """Convenience wrapper: load existing state, or start a fresh one."""
        existing = await self.get(conversation_id)
        if existing is not None:
            return existing
        fresh = ConversationState(conversation_id=conversation_id)
        await self.save(fresh)
        return fresh