"""
Agent 5: Classify, Prioritise, Explain.

Wraps a ClassificationProvider (the LLM's judgment) with deterministic
guardrails, each a standalone pure function so it can be unit-tested
with hand-built inputs (see tests/unit/test_classification_agent.py):

  - resolve_category: Agent 1's category wins unless it is missing,
    "general", or low-confidence; the LLM may only override with a key
    in VALID_CATEGORIES.
  - adjust_confidence: LLM confidence is capped by Agent 1's when they
    agree, and discounted when they disagree.
  - apply_cluster_override: an INCIDENT_CLUSTER correlation forces P1.
  - assignee_group: always routing.team_for_category, never the LLM.

classify() never raises. Any provider error, timeout, or malformed
output yields a source="rule" result instead (rule_fallback). It does
NOT apply the confidence gate - that is the orchestrator's job
(settings.classification_confidence_threshold).
"""

import asyncio
import logging

from app.config.category_profiles import VALID_CATEGORIES
from app.config.routing import team_for_category
from app.providers.classification_base import ClassificationProvider
from app.schemas.classification import ClassificationLLMOutput
from app.schemas.conversation import ConversationState
from app.schemas.pipeline import ClassificationResult, CorrelationResult, Priority

logger = logging.getLogger(__name__)

# Agent 1's category is trusted below this only if the LLM can't offer
# a valid alternative.
AGENT1_TRUST_THRESHOLD = 0.5
# Confidence multiplier when the LLM's category disagrees with Agent 1's.
DISAGREEMENT_PENALTY = 0.7

RULE_PRIORITY: Priority = "P3"
RULE_CONFIDENCE = 0.3
RULE_FALLBACK_TAG = "[rule fallback]"
CLUSTER_OVERRIDE_TAG = "[incident cluster -> P1]"

DEFAULT_TIMEOUT_SECONDS = 20.0


def _agent1_category(state: ConversationState) -> tuple[str | None, float]:
    pred = state.detected_category
    if pred is None:
        return None, 0.0
    return pred.value, pred.confidence


def resolve_category(
    agent1_category: str | None,
    agent1_confidence: float,
    llm_category: str,
) -> tuple[str, bool]:
    """
    Return (final_category, llm_overrode_agent1).

    Agent 1 is kept when it is a real category with confidence >=
    AGENT1_TRUST_THRESHOLD. Otherwise the LLM's category is used if it is
    a valid key; failing that, Agent 1's category if valid, else "general".
    """
    agent1_valid = agent1_category in VALID_CATEGORIES
    agent1_trusted = (
        agent1_valid
        and agent1_category != "general"
        and agent1_confidence >= AGENT1_TRUST_THRESHOLD
    )
    if agent1_trusted:
        return agent1_category, False

    if llm_category in VALID_CATEGORIES and llm_category != agent1_category:
        return llm_category, True

    return (agent1_category if agent1_valid else "general"), False


def adjust_confidence(
    llm_confidence: float,
    agent1_confidence: float,
    agreed: bool,
    overrode: bool,
) -> float:
    """
    agreed   -> min(llm, agent1): the LLM can't be surer than the extractor.
    overrode -> llm * DISAGREEMENT_PENALTY.
    neither  (LLM disagreed but Agent 1's category was kept) ->
                min(llm, agent1) * DISAGREEMENT_PENALTY.
    Always clamped to [0, 1].
    """
    if overrode:
        conf = llm_confidence * DISAGREEMENT_PENALTY
    elif agreed:
        conf = min(llm_confidence, agent1_confidence)
    else:
        conf = min(llm_confidence, agent1_confidence) * DISAGREEMENT_PENALTY
    return max(0.0, min(1.0, conf))


def apply_cluster_override(
    priority: Priority,
    rationale: str,
    correlation: CorrelationResult | None,
) -> tuple[Priority, str]:
    """Force P1 (and say why) when correlation found an incident cluster."""
    if correlation is None or correlation.outcome != "INCIDENT_CLUSTER":
        return priority, rationale

    detail = f"{len(correlation.similar_tickets)} similar open ticket(s)"
    if correlation.cluster_id:
        detail = f"cluster {correlation.cluster_id}, {detail}"
    reason = (
        f"{CLUSTER_OVERRIDE_TAG} Priority raised from {priority} to P1: "
        f"correlation detected an incident cluster ({detail})."
    )
    return "P1", f"{rationale} {reason}"


def _routing_sentence(category: str, team: str) -> str:
    return f"Routed to {team} via category routing table ('{category}')."


def rule_fallback(state: ConversationState, reason: str) -> ClassificationResult:
    """
    Deterministic result used whenever the LLM path fails. Low fixed
    confidence so the orchestrator's gate sends it to review.
    """
    agent1_category, _ = _agent1_category(state)
    category = agent1_category if agent1_category in VALID_CATEGORIES else "general"
    team = team_for_category(category)
    rationale = (
        f"{RULE_FALLBACK_TAG} LLM classification unavailable ({reason}); "
        f"default priority {RULE_PRIORITY} applied. {_routing_sentence(category, team)}"
    )
    priority, rationale = apply_cluster_override(RULE_PRIORITY, rationale, state.correlation)
    return ClassificationResult(
        category=category,
        priority=priority,
        assignee_group=team,
        routing_rationale=rationale,
        confidence=RULE_CONFIDENCE,
        source="rule",
    )


def was_rule_fallback(result: ClassificationResult) -> bool:
    """True if this result came from rule_fallback rather than the LLM."""
    return result.source == "rule"


def build_result(state: ConversationState, output: ClassificationLLMOutput) -> ClassificationResult:
    """Apply every deterministic guardrail to a validated LLM output."""
    agent1_category, agent1_confidence = _agent1_category(state)
    category, overrode = resolve_category(agent1_category, agent1_confidence, output.category)
    agreed = output.category == category and category == agent1_category

    confidence = adjust_confidence(output.confidence, agent1_confidence, agreed, overrode)
    team = team_for_category(category)

    rationale = output.rationale.strip()
    if overrode:
        rationale += (
            f" Category changed from Agent 1's '{agent1_category or 'undetermined'}' "
            f"to '{category}'."
        )
    elif output.category != category:
        rationale += (
            f" Model suggested '{output.category}'; kept '{category}' from extraction."
        )
    rationale = f"{rationale} {_routing_sentence(category, team)}".strip()

    priority, rationale = apply_cluster_override(output.priority, rationale, state.correlation)

    return ClassificationResult(
        category=category,
        priority=priority,
        assignee_group=team,
        routing_rationale=rationale,
        confidence=confidence,
        source="llm",
    )


class ClassificationAgent:
    """
    Agent 5. Satisfies app.agents.interfaces.ClassificationAgent.

    Same name as the Protocol by design (the Protocol is the contract,
    this is the implementation); import the Protocol under an alias where
    both are needed.
    """

    def __init__(
        self,
        provider: ClassificationProvider,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self._provider = provider
        self._timeout = timeout_seconds

    async def classify(self, state: ConversationState) -> ClassificationResult:
        try:
            output = await asyncio.wait_for(
                self._provider.classify_ticket(state), timeout=self._timeout
            )
            if not isinstance(output, ClassificationLLMOutput):
                output = ClassificationLLMOutput.model_validate(output)
            return build_result(state, output)
        except asyncio.TimeoutError:
            reason = f"provider timed out after {self._timeout:g}s"
            detail = reason
        except Exception as exc:  # noqa: BLE001 - contract: never raise
            # Only the exception type goes into the rationale (it ends up on
            # the ticket); the full message goes to the log.
            reason = type(exc).__name__
            detail = f"{reason}: {exc}"

        logger.warning("Agent 5 rule fallback for %s: %s", state.conversation_id, detail)
        return rule_fallback(state, reason)
