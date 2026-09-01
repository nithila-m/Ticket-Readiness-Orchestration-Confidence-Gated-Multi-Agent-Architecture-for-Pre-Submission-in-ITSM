"""
Deterministic rule-based ClarificationProvider - an ablation baseline,
not a production alternative to Agent 2.

Exists for exactly one purpose: to make the "isn't this just a
threshold on the completeness score" challenge falsifiable rather than
rhetorical. It implements the literal naive rule a skeptical reviewer
would assume the whole system reduces to:

    if completeness_score >= READY_THRESHOLD: READY
    else: ask about the single highest-weight unresolved field

Because it implements the same ClarificationProvider interface as
GroqClarificationProvider, it drops into the existing AdaptiveClarifier
unchanged - including the turn-budget safeguard. That's worth pointing
out on its own: the safeguard is provider-agnostic, not something
bolted onto the LLM path specifically.

This provider is deliberately not "smart." It cannot combine gaps into
one question, cannot recognize an answer already given in different
wording, cannot detect contradictions, and has no real basis for its
"confidence" field beyond a fixed placeholder. The gap between this and
Agent 2's actual behavior on identical inputs is the whole point.
"""

from app.agents.clarification_context import classify_field_gaps
from app.config.category_profiles import get_profile
from app.providers.clarification_base import ClarificationProvider
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState

# A starting point, exactly as arbitrary as the other thresholds already
# flagged elsewhere in the codebase (KB deflection 0.72, field confidence
# 0.5) - not fit to any data, deliberately, since this baseline's whole
# purpose is to represent "the simplest rule a skeptic assumes we built."
READY_THRESHOLD = 0.6

# Placeholder only - a rule has no real basis for expressing confidence
# in its own decision the way an LLM's self-report at least attempts to.
_FIXED_CONFIDENCE = 0.5


class RuleBasedClarificationProvider(ClarificationProvider):
    """Ablation baseline: single-scalar threshold, no LLM call, no cost."""

    def __init__(self, ready_threshold: float = READY_THRESHOLD):
        self._ready_threshold = ready_threshold

    async def decide_clarification(self, state: ConversationState) -> ClarificationDecision:
        category = state.detected_category.value if state.detected_category else None
        profile = get_profile(category)
        never_extracted, uncertain = classify_field_gaps(category, state.extracted_fields)
        gaps = never_extracted + uncertain

        if not gaps or state.completeness_score >= self._ready_threshold:
            return ClarificationDecision(
                action="READY",
                reasoning=(
                    f"Rule-based baseline: completeness_score="
                    f"{state.completeness_score:.2f} >= threshold="
                    f"{self._ready_threshold} (or no unresolved profile fields remain)."
                ),
                information_gap=None,
                question=None,
                expected_information_gain=0.0,
                affected_fields=[],
                priority="low",
                confidence=_FIXED_CONFIDENCE,
            )

        # Pick the single highest-weight unresolved field - no attempt to
        # combine multiple gaps, recognize an answer given elsewhere in
        # the transcript, or weigh situational relevance. This is exactly
        # the fixed field-to-question mapping the Agent 2 system prompt's
        # own "hard rules" section forbids - here, deliberately, as the
        # contrast case.
        target_field = max(gaps, key=lambda f: profile.get(f, 0.0))
        weight = profile.get(target_field, 0.0)

        return ClarificationDecision(
            action="ASK_CLARIFICATION",
            reasoning=(
                f"Rule-based baseline: completeness_score={state.completeness_score:.2f} "
                f"< threshold={self._ready_threshold}. '{target_field}' has the highest "
                f"importance weight ({weight}) among unresolved fields "
                f"{sorted(gaps)}; no other signal was considered."
            ),
            information_gap=target_field,
            question=f"Could you provide more detail about {target_field.replace('_', ' ')}?",
            expected_information_gain=min(weight, 1.0),
            affected_fields=[target_field],
            priority="high" if weight >= 0.8 else "medium" if weight >= 0.5 else "low",
            confidence=_FIXED_CONFIDENCE,
        )
