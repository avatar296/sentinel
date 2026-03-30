from __future__ import annotations

import json
from dataclasses import dataclass

from sentinel.services.rules_engine import RulesVerdict


@dataclass
class EscalationDecision:
    decision: str  # "APPROVE", "FLAG", "REVIEW"
    reasons: str   # JSON string


def route(
    ml_score: float,
    rules_verdict: RulesVerdict,
    fraud_threshold: float = 0.8,
    review_threshold: float = 0.4,
) -> EscalationDecision:
    """Determine the final decision based on ML score and rules verdict.

    Decision matrix:
        ML >= fraud_threshold           -> FLAG  (any rules)
        ML < review_threshold, no rules -> APPROVE
        ML < review_threshold, rules    -> REVIEW (rules disagree with ML)
        ML in gray zone, no rules       -> REVIEW (uncertain)
        ML in gray zone, rules flagged  -> FLAG   (both agree)
    """
    triggered_names = [r.name for r in rules_verdict.triggered_rules]
    reasons = json.dumps({
        "ml_score": round(ml_score, 4),
        "rules_score": round(rules_verdict.rules_score, 4),
        "triggered_rules": triggered_names,
    })

    # High-confidence fraud
    if ml_score >= fraud_threshold:
        return EscalationDecision(decision="FLAG", reasons=reasons)

    # Gray zone
    if ml_score >= review_threshold:
        if rules_verdict.flagged:
            return EscalationDecision(decision="FLAG", reasons=reasons)
        return EscalationDecision(decision="REVIEW", reasons=reasons)

    # Low ML score
    if rules_verdict.flagged:
        return EscalationDecision(decision="REVIEW", reasons=reasons)

    return EscalationDecision(decision="APPROVE", reasons=reasons)
