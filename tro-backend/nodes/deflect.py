# Looks up relevant KB knowledge and attaches it to the current state for deflection or self-service guidance.
def deflect(state: dict) -> dict:
    state["retrieved_kb_docs"] = [
        {"kb_id": "STUB_KB_001", "title": "Stub KB article", "score": 0.0}
    ]
    return state