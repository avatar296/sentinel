from datetime import datetime

from sentinel.services.rules_engine import (
    evaluate_rules,
    high_amount_rule,
    geo_anomaly_rule,
    merchant_risk_rule,
    time_anomaly_rule,
    velocity_rule,
)
from sentinel.services.velocity_tracker import VelocityTracker


class TestHighAmountRule:
    def test_low_amount(self):
        r = high_amount_rule(100.0)
        assert not r.triggered

    def test_above_5k(self):
        r = high_amount_rule(6000.0)
        assert r.triggered and r.weight == 0.4

    def test_above_10k(self):
        r = high_amount_rule(15000.0)
        assert r.triggered and r.weight == 0.7


class TestVelocityRule:
    def test_no_tracker(self):
        r = velocity_rule("1234", None)
        assert not r.triggered

    def test_below_threshold(self):
        tracker = VelocityTracker()
        tracker.record("1234", "US")
        tracker.record("1234", "US")
        r = velocity_rule("1234", tracker)
        assert not r.triggered

    def test_at_threshold(self):
        tracker = VelocityTracker()
        for _ in range(3):
            tracker.record("1234", "US")
        r = velocity_rule("1234", tracker)
        assert r.triggered and r.weight == 0.6


class TestGeoAnomalyRule:
    def test_safe_country(self):
        r = geo_anomaly_rule("US", "1234", None)
        assert not r.triggered

    def test_high_risk_country(self):
        r = geo_anomaly_rule("NG", "1234", None)
        assert r.triggered and r.weight == 0.3

    def test_country_change(self):
        tracker = VelocityTracker()
        tracker.record("1234", "US")
        r = geo_anomaly_rule("NG", "1234", tracker)
        assert r.triggered and r.weight == 0.5  # impossible travel > country risk


class TestTimeAnomalyRule:
    def test_normal_hour(self):
        r = time_anomaly_rule(datetime(2024, 1, 1, 14, 0))
        assert not r.triggered

    def test_late_night(self):
        r = time_anomaly_rule(datetime(2024, 1, 1, 3, 0))
        assert r.triggered and r.weight == 0.2


class TestMerchantRiskRule:
    def test_safe_merchant(self):
        r = merchant_risk_rule("grocery", 3000.0)
        assert not r.triggered

    def test_high_risk_low_amount(self):
        r = merchant_risk_rule("electronics", 500.0)
        assert not r.triggered

    def test_high_risk_high_amount(self):
        r = merchant_risk_rule("electronics", 3000.0)
        assert r.triggered and r.weight == 0.3


class TestEvaluateRules:
    def test_clean_transaction(self):
        txn = {
            "amount": 50.0,
            "card_last_four": "1234",
            "location_country": "US",
            "merchant_category": "grocery",
            "transaction_time": datetime(2024, 1, 1, 14, 0),
        }
        verdict = evaluate_rules(txn)
        assert verdict.rules_score == 0.0
        assert not verdict.flagged

    def test_suspicious_transaction(self):
        txn = {
            "amount": 8000.0,
            "card_last_four": "5678",
            "location_country": "NG",
            "merchant_category": "electronics",
            "transaction_time": datetime(2024, 1, 1, 3, 0),
        }
        verdict = evaluate_rules(txn)
        assert verdict.rules_score > 0.5
        assert verdict.flagged
        assert len(verdict.triggered_rules) >= 3
