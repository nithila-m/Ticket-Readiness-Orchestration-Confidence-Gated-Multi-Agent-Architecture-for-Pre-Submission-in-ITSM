"""
In-memory ConversationRepository. Process-local, non-persistent across
restarts - correct for local dev and the Review 2 demo, not for
production (digest.txt Section 21: no databases yet for this milestone).
"""

import asyncio

from app.repositories.base import ConversationRepository
from app.schemas.conversation import ConversationState


class InMemoryConversationRepository(ConversationRepository):
    def __init__(self):
        self._store: dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()

    async def get(self, conversation_id: str) -> ConversationState | None:
        async with self._lock:
            state = self._store.get(conversation_id)
            # Return a copy so callers can't mutate stored state without
            # going through save() - keeps the repository the single
            # source of truth for what's "persisted."
            return state.model_copy(deep=True) if state else None

    async def save(self, state: ConversationState) -> None:
        async with self._lock:
            self._store[state.conversation_id] = state.model_copy(deep=True)