from langgraph.graph import StateGraph, END
from state import TicketState

from nodes.intake_clarify import intake_clarify
from nodes.deflect import deflect
from nodes.duplicate_check import duplicate_check
from nodes.classify_route import classify_route

# Build the workflow graph that passes TicketState through a sequence of processing nodes.
builder = StateGraph(TicketState)

# Register the four pipeline stages used in the current TRO workflow.
builder.add_node("intake_clarify", intake_clarify)
builder.add_node("deflect", deflect)
builder.add_node("duplicate_check", duplicate_check)
builder.add_node("classify_route", classify_route)

# Wire the stages in a simple linear flow for the initial prototype.
# This keeps the system easy to reason about before adding more complex branching.
builder.set_entry_point("intake_clarify")
builder.add_edge("intake_clarify", "deflect")
builder.add_edge("deflect", "duplicate_check")
builder.add_edge("duplicate_check", "classify_route")
builder.add_edge("classify_route", END)

# Compile the graph so it can be invoked by the API request handler.
graph = builder.compile()