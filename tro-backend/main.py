from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import graph

# Create the FastAPI application that exposes the TRO backend endpoints.
app = FastAPI(title="TRO Backend", version="0.2.0")

# Allow browser-based clients to call the API during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incoming request body from the frontend: a session identifier and the user's message.
# FastAPI validates this automatically using Pydantic.
class ChatRequest(BaseModel):
    session_id: str
    message: str


# Seed the state for a new ticket workflow. Each message starts from this base structure.
def build_initial_state(session_id: str, message: str) -> dict:
    """Build the empty TicketState that seeds every pipeline run."""
    return {
        "session_id": session_id,
        "user_messages": [message],
        "extracted_fields": {},
        "missing_fields": [],
        "completeness_score": 0.0,
        "retrieved_kb_docs": [],
        "similar_tickets": [],
        "classification_confidence": 0.0,
        "routing_decision": None,
        "escalation_needed": False,
        "audit_log": [],
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "TRO Backend", "version": "0.2.0"}


# Receive the user's input, initialize the workflow state, and run the graph pipeline.
# The graph is responsible for each stage of the TRO processing flow.
@app.post("/chat")
def chat(req: ChatRequest):
    initial_state = build_initial_state(req.session_id, req.message)
    final_state = graph.invoke(initial_state)
    return final_state

'''
v-0.1.0: That's the whole file. A few notes on why it's shaped this way:
* ChatRequest uses Pydantic, so FastAPI auto-validates the incoming JSON. 
  Bad payloads get a 422 with a clear error, so you don't have to write validation.
* The GET / health check is a 10-second addition that saves debugging time later — 
  you can open http://localhost:8000/ in a browser and instantly see whether the server is up.
*CORS is wide-open (allow_origins=["*"]) which is fine for local dev. Tighten it before any 
 real deployment, not now.
'''

'''
v-0.2.0: A few things worth noting about what changed:
* from graph import graph pulls in the compiled StateGraph you already built.
* build_initial_state() is factored out as a small helper — it's the only place the initial 
  state shape is defined, so if you add a field later you change it once.
* The return value is final_state, which is whatever LangGraph gives back after all four nodes 
  have mutated the dict. All four stubs will have touched it by the time it comes back.
'''