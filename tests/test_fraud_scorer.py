from sentinel.services.fraud_scorer import FraudScorer
from sentinel.services.model_registry import ModelRegistry


def _scorer():
    """Create a FraudScorer with an empty registry (forces heuristic fallback)."""
    return FraudScorer(ModelRegistry())


def test_heuristic_low_risk():
    scorer = _scorer()
    score = scorer.score({
        "amount": 50,
        "is_online": False,
        "location_country": "US",
        "merchant_category": "grocery",
        "transaction_time": "2025-06-15T14:30:00",
    })
    assert 0.0 <= score <= 1.0
    assert score < 0.3


def test_heuristic_high_risk():
    scorer = _scorer()
    score = scorer.score({
        "amount": 8000,
        "is_online": True,
        "location_country": "NG",
        "merchant_category": "electronics",
        "transaction_time": "2025-06-15T02:00:00",
    })
    assert 0.0 <= score <= 1.0
    assert score >= 0.5


def test_heuristic_medium_risk():
    scorer = _scorer()
    score = scorer.score({
        "amount": 3000,
        "is_online": False,
        "location_country": "US",
        "merchant_category": "travel",
        "transaction_time": "2025-06-15T10:00:00",
    })
    assert 0.0 <= score <= 1.0
    assert 0.1 <= score <= 0.5
