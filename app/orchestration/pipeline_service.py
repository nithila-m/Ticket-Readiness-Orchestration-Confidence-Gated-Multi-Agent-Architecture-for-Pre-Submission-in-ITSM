"""
GraphConversationService: the conversation API backed by the LangGraph
orchestrator (app/orchestration/graph.py).

Same constructor and public methods as ClarificationService, so it is a
drop-in replacement in app/dependencies.py (settings.orchestrator) and the
same tests run against both. Each call: load state from the repository ->
`await graph.ainvoke(...)` -> save.
"""

from app.agents.adaptive_clarifier import AdaptiveClarifier
from app.agents.kb_deflection_agent import KBDeflectionAgent
from app.config.settings import settings
from app.orchestration.graph import build_graph
from app.orchestration.nodes import PipelineNodes
from app.repositories.base import ConversationRepository
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState
from app.services.clarification_service import ClarificationService
from app.services.ticket_analysis_service import TicketAnalysisService
from app.services.ticket_pipeline import TicketPipeline
from app.storage.sqlite_store import SqliteStore


class GraphConversationService:
    def __init__(
        self,
        extraction_service: TicketAnalysisService,
        clarifier: AdaptiveClarifier,
        repository: ConversationRepository,
        kb_agent: KBDeflectionAgent,
        ticket_pipeline: TicketPipeline | None = None,
        store: SqliteStore | None = None,
    ):
        # The step methods live on ClarificationService; the graph only
        # decides the order they run in.
        self._steps = ClarificationService(
            extraction_service, clarifier, repository, kb_agent,
            ticket_pipeline=ticket_pipeline, store=store,
        )
        self._ticket_pipeline = ticket_pipeline
        self._graph = build_graph(PipelineNodes(self._steps))

    @property
    def graph(self):
        return self._graph

    def _config(self) -> dict:
        return {
            "pipeline_enabled": self._ticket_pipeline is not None,
            "auto_dispatch": settings.auto_dispatch_on_ready,
        }

    async def handle_message(
        self, conversation_id: str, user_message: str
    ) -> tuple[ConversationState, ClarificationDecision]:
        state = await self._steps.begin_turn(conversation_id, user_message)
        result = await self._graph.ainvoke({
            "conversation": state, "entry": "message", "user_message": user_message,
            **self._config(),
        })
        await self._steps.end_turn(state)
        return state, result["decision"]

    async def handle_deflection_feedback(
        self, conversation_id: str, resolved: bool
    ) -> ConversationState:
        state = await self._steps.begin_feedback(conversation_id)
        await self._graph.ainvoke({
            "conversation": state, "entry": "feedback", "resolved": resolved, **self._config(),
        })
        await self._steps.save(state)
        return state

    async def handle_duplicate_feedback(
        self, conversation_id: str, is_duplicate: bool
    ) -> ConversationState:
        state = await self._steps.begin_duplicate_feedback(conversation_id)
        await self._graph.ainvoke({
            "conversation": state, "entry": "duplicate_feedback", "is_duplicate": is_duplicate,
            **self._config(),
        })
        await self._steps.save(state)
        return state

    async def dispatch_approved(self, state: ConversationState) -> ConversationState:
        """Entry for a reviewer-approved item: straight to the Dispatcher."""
        if self._ticket_pipeline is None:
            raise RuntimeError("dispatch_approved needs a TicketPipeline")
        await self._graph.ainvoke({"conversation": state, "entry": "review_approved", **self._config()})
        return state
