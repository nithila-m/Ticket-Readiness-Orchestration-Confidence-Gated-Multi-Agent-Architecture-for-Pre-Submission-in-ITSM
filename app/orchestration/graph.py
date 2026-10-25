"""
The TRO orchestration graph (LangGraph StateGraph).

    START --entry--> extract .................. (user message)
                     apply_feedback ........... (answer to a KB deflection)
                     apply_duplicate_feedback . (answer to a suspected duplicate)
                     dispatch ................. (reviewer approved a queued item)

    extract --completeness_gate--> kb | clarify
    kb ------kb_gate-------------> deflect | clarify
    deflect -----------------------> END   (waits for feedback)
    clarify --clarify_router-------> correlate | enqueue_review | END
    apply_feedback --feedback_router-> close_resolved | correlate | END
    correlate --correlation_router--> confirm_duplicate | classify
    confirm_duplicate -------------> END   (waits for the duplicate answer)
    apply_duplicate_feedback ------> classify
    classify --confidence_gate--> dispatch | enqueue_review
    close_resolved / enqueue_review / dispatch -> END
"""

from langgraph.graph import END, START, StateGraph

from app.orchestration import routing
from app.orchestration.nodes import PipelineNodes
from app.orchestration.state import GraphState


def build_graph(nodes: PipelineNodes):
    g = StateGraph(GraphState)

    for name in (
        "extract", "kb", "deflect", "clarify", "apply_feedback", "close_resolved",
        "correlate", "confirm_duplicate", "apply_duplicate_feedback",
        "classify", "enqueue_review", "dispatch",
    ):
        g.add_node(name, getattr(nodes, name))

    g.add_conditional_edges(START, routing.entry_router, {
        "message": "extract",
        "feedback": "apply_feedback",
        "duplicate_feedback": "apply_duplicate_feedback",
        "review_approved": "dispatch",
    })
    g.add_conditional_edges("extract", routing.completeness_gate, {"kb": "kb", "clarify": "clarify"})
    g.add_conditional_edges("kb", routing.kb_gate, {"deflect": "deflect", "clarify": "clarify"})
    g.add_edge("deflect", END)
    g.add_conditional_edges("clarify", routing.clarify_router, {
        "correlate": "correlate",
        "enqueue_review": "enqueue_review",
        routing.END_BRANCH: END,
    })
    g.add_conditional_edges("apply_feedback", routing.feedback_router, {
        "close_resolved": "close_resolved",
        "correlate": "correlate",
        routing.END_BRANCH: END,
    })
    g.add_edge("close_resolved", END)
    g.add_conditional_edges("correlate", routing.correlation_router, {
        "confirm_duplicate": "confirm_duplicate",
        "classify": "classify",
    })
    g.add_edge("confirm_duplicate", END)
    g.add_edge("apply_duplicate_feedback", "classify")
    g.add_conditional_edges("classify", routing.confidence_gate, {
        "dispatch": "dispatch",
        "enqueue_review": "enqueue_review",
    })
    g.add_edge("enqueue_review", END)
    g.add_edge("dispatch", END)

    return g.compile()
