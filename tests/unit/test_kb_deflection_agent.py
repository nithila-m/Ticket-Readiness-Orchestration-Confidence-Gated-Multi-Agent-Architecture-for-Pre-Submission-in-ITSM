from app.agents.kb_deflection_agent import KBRetrievalResult, build_deflection_decision


def test_build_deflection_decision_shapes_valid_clarification_decision():
    kb_result = KBRetrievalResult(
        outcome="STRONG_MATCH", similarity_score=0.88, articles_checked=["KB0302"],
        matched_kb_id="KB0302",
        matched_title="Wi-Fi disconnects repeatedly after device wakes from sleep",
        offered_resolution="1. Disable fast startup\n2. Forget and re-join Wi-Fi",
    )
    decision = build_deflection_decision(kb_result)

    assert decision.action == "DEFLECTED"
    assert decision.question is None
    assert decision.confidence == 0.88
    assert "Wi-Fi disconnects repeatedly after device wakes from sleep" in decision.reasoning


def test_build_deflection_decision_reasoning_includes_similarity_score():
    kb_result = KBRetrievalResult(
        outcome="STRONG_MATCH", similarity_score=0.7321, articles_checked=["KB0301"],
        matched_kb_id="KB0301", matched_title="VPN disconnects after laptop wakes from sleep",
        offered_resolution="Update VPN client to latest version.",
    )
    decision = build_deflection_decision(kb_result)

    assert "0.73" in decision.reasoning