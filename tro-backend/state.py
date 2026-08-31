from typing import TypedDict, Optional, List, Dict, Any

# Defines the shared state object passed through the TRO workflow for each ticket/session.
class TicketState(TypedDict):
    session_id: str
    user_messages: List[str]
    extracted_fields: Dict[str, Any]
    missing_fields: List[str]
    completeness_score: float
    retrieved_kb_docs: List[Dict[str, Any]]
    similar_tickets: List[Dict[str, Any]]
    classification_confidence: float
    routing_decision: Optional[str]
    escalation_needed: bool
    audit_log: List[Dict[str, Any]]