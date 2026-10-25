"""
LangGraph nodes. Each node is a thin wrapper around ONE existing step
method of ClarificationService or TicketPipeline, so the graph and the
hand-written services share every line of agent, gate and audit logic -
nothing is re-implemented here.
"""

from app.orchestration.state import GraphState
from app.services.clarification_service import ESCALATION_REASON, ClarificationService
from app.services.ticket_pipeline import LOW_CONFIDENCE_REASON


class PipelineNodes:
    def __init__(self, steps: ClarificationService):
        self._steps = steps

    @property
    def _pipeline(self):
        return self._steps.ticket_pipeline

    # --- conversation turn (Agents 1-3) -------------------------------------

    async def extract(self, state: GraphState) -> dict:
        await self._steps.extract(state["conversation"])
        return {"conversation": state["conversation"]}

    async def kb(self, state: GraphState) -> dict:
        kb_result = await self._steps.check_kb(state["conversation"])
        return {"kb_result": kb_result}

    async def deflect(self, state: GraphState) -> dict:
        decision = await self._steps.deflect(state["conversation"], state["kb_result"])
        self._steps.record_decision(state["conversation"], state["user_message"], decision)
        return {"decision": decision}

    async def clarify(self, state: GraphState) -> dict:
        decision = await self._steps.clarify(state["conversation"])
        self._steps.record_decision(state["conversation"], state["user_message"], decision)
        return {"decision": decision}

    # --- deflection feedback ------------------------------------------------

    async def apply_feedback(self, state: GraphState) -> dict:
        await self._steps.apply_feedback(state["conversation"], state["resolved"])
        return {"conversation": state["conversation"]}

    async def close_resolved(self, state: GraphState) -> dict:
        await self._steps.close_resolved(state["conversation"])
        return {"conversation": state["conversation"]}

    # --- ticket pipeline (Agents 4-6) ---------------------------------------

    async def correlate(self, state: GraphState) -> dict:
        await self._pipeline.correlate(state["conversation"])
        return {"conversation": state["conversation"]}

    async def confirm_duplicate(self, state: GraphState) -> dict:
        await self._pipeline.ask_duplicate_confirmation(state["conversation"])
        return {"conversation": state["conversation"]}

    async def apply_duplicate_feedback(self, state: GraphState) -> dict:
        await self._pipeline.apply_duplicate_answer(state["conversation"], state["is_duplicate"])
        return {"conversation": state["conversation"]}

    async def classify(self, state: GraphState) -> dict:
        await self._pipeline.classify(state["conversation"])
        gate = await self._pipeline.gate(state["conversation"])
        return {"gate": gate}

    async def enqueue_review(self, state: GraphState) -> dict:
        # Reached from the confidence gate (REVIEW) or from Agent 2 (ESCALATE).
        if state.get("gate") == "REVIEW":
            reason, outcome = LOW_CONFIDENCE_REASON, "PENDING_REVIEW"
        else:
            reason, outcome = ESCALATION_REASON, "ESCALATED"
        await self._pipeline.enqueue_review(state["conversation"], reason, outcome)
        return {"conversation": state["conversation"]}

    async def dispatch(self, state: GraphState) -> dict:
        await self._pipeline.dispatch(state["conversation"])
        return {"conversation": state["conversation"]}
