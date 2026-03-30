from sentinel.services.escalation import route
from sentinel.services.rules_engine import RuleResult, RulesVerdict


def _verdict(flagged: bool, score: float = 0.0, rules: list | None = None) -> RulesVerdict:
    return RulesVerdict(
        triggered_rules=rules or [],
        rules_score=score,
        flagged=flagged,
    )


class TestEscalationRouting:
    def test_high_ml_score_flags(self):
        result = route(0.9, _verdict(False))
        assert result.decision == "FLAG"

    def test_high_ml_score_with_rules_flags(self):
        result = route(0.85, _verdict(True, 0.6))
        assert result.decision == "FLAG"

    def test_low_ml_no_rules_approves(self):
        result = route(0.1, _verdict(False))
        assert result.decision == "APPROVE"

    def test_low_ml_with_rules_reviews(self):
        rules = [RuleResult("high_amount", True, 0.5, "amount > $5k")]
        result = route(0.1, _verdict(True, 0.5, rules))
        assert result.decision == "REVIEW"

    def test_gray_zone_no_rules_reviews(self):
        result = route(0.5, _verdict(False))
        assert result.decision == "REVIEW"

    def test_gray_zone_with_rules_flags(self):
        result = route(0.5, _verdict(True, 0.6))
        assert result.decision == "FLAG"

    def test_boundary_at_fraud_threshold(self):
        result = route(0.8, _verdict(False))
        assert result.decision == "FLAG"

    def test_boundary_at_review_threshold(self):
        result = route(0.4, _verdict(False))
        assert result.decision == "REVIEW"

    def test_just_below_review_threshold(self):
        result = route(0.39, _verdict(False))
        assert result.decision == "APPROVE"
