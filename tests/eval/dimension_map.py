"""
Maps each of Section 25's 8 evaluation dimensions to the scenario(s) that
provide evidence for it. Some dimensions have no fixture coverage yet -
these are reported as NOT YET MEASURABLE rather than given a fabricated
score (see turn_efficiency below).
"""

DIMENSION_MAP: dict[str, list[str]] = {
    "question_quality": ["one_question_multiple_gaps", "ambiguous_category"],
    "redundancy": ["one_question_multiple_gaps", "avoid_repetition"],
    "information_gain": ["one_question_multiple_gaps", "turn_budget_conservatism"],
    "turn_efficiency": [],  # not measurable from single-shot fixtures - needs a
                            # multi-turn oracle comparison (MAS2S-style); none
                            # of the 8 scenarios test this by design
    "readiness_quality": [
        "ready_despite_incomplete_fields",
        "high_completeness_critical_ambiguity",
        "low_completeness_actionable",
    ],
    "over_questioning": [
        "high_completeness_critical_ambiguity",  # correctly did NOT over-ask
        "low_completeness_actionable",           # did over-ask
    ],
    "adaptation": ["avoid_repetition", "contradiction_detection"],
    "contradiction_handling": ["contradiction_detection"],
}