from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from sentinel.services.velocity_tracker import VelocityTracker

HIGH_RISK_COUNTRIES = {"NG", "RU", "CN", "BR", "PH", "VN", "IN"}
HIGH_RISK_CATEGORIES = {"electronics", "online_retail", "travel"}


@dataclass
class RuleResult:
    name: str
    triggered: bool
    weight: float
    reason: str


@dataclass
class RulesVerdict:
    triggered_rules: list[RuleResult]
    rules_score: float
    flagged: bool


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------

def high_amount_rule(amount: float) -> RuleResult:
    if amount > 10_000:
        return RuleResult("high_amount", True, 0.7, f"Amount ${amount:,.2f} exceeds $10,000")
    if amount > 5_000:
        return RuleResult("high_amount", True, 0.4, f"Amount ${amount:,.2f} exceeds $5,000")
    return RuleResult("high_amount", False, 0.0, "")


def velocity_rule(card_last_four: str, tracker: VelocityTracker | None) -> RuleResult:
    if tracker is None:
        return RuleResult("velocity", False, 0.0, "")
    count = tracker.count_recent(card_last_four)
    if count >= 3:
        return RuleResult("velocity", True, 0.6, f"{count} transactions in last 10 min")
    return RuleResult("velocity", False, 0.0, "")


def geo_anomaly_rule(
    country: str,
    card_last_four: str,
    tracker: VelocityTracker | None,
) -> RuleResult:
    country = country.upper()
    # Check impossible travel first (higher weight)
    if tracker is not None:
        last = tracker.get_last_country(card_last_four)
        if last is not None and last.upper() != country:
            return RuleResult(
                "geo_anomaly", True, 0.5,
                f"Country changed from {last} to {country}",
            )
    if country in HIGH_RISK_COUNTRIES:
        return RuleResult("geo_anomaly", True, 0.3, f"High-risk country: {country}")
    return RuleResult("geo_anomaly", False, 0.0, "")


def time_anomaly_rule(transaction_time: datetime) -> RuleResult:
    hour = transaction_time.hour
    if 1 <= hour < 5:
        return RuleResult("time_anomaly", True, 0.2, f"Late-night transaction at {hour}:00")
    return RuleResult("time_anomaly", False, 0.0, "")


def merchant_risk_rule(category: str, amount: float) -> RuleResult:
    if category.lower() in HIGH_RISK_CATEGORIES and amount > 2_000:
        return RuleResult(
            "merchant_risk", True, 0.3,
            f"High-risk category '{category}' with amount ${amount:,.2f}",
        )
    return RuleResult("merchant_risk", False, 0.0, "")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def evaluate_rules(
    transaction: dict,
    tracker: VelocityTracker | None = None,
    rules_threshold: float = 0.5,
) -> RulesVerdict:
    """Run all rules against a transaction dict and return an aggregated verdict."""
    amount = float(transaction.get("amount", 0))
    card = str(transaction.get("card_last_four", ""))
    country = str(transaction.get("location_country", ""))
    category = str(transaction.get("merchant_category", ""))
    txn_time = transaction.get("transaction_time")
    if isinstance(txn_time, str):
        txn_time = datetime.fromisoformat(txn_time)

    results = [
        high_amount_rule(amount),
        velocity_rule(card, tracker),
        geo_anomaly_rule(country, card, tracker),
        time_anomaly_rule(txn_time),
        merchant_risk_rule(category, amount),
    ]

    triggered = [r for r in results if r.triggered]
    score = min(1.0, sum(r.weight for r in triggered))
    return RulesVerdict(
        triggered_rules=triggered,
        rules_score=score,
        flagged=score >= rules_threshold,
    )


# ---------------------------------------------------------------------------
# Batch evaluation helper (stateless rules only — for notebook use)
# ---------------------------------------------------------------------------

def evaluate_rules_on_dataframe(
    df: pd.DataFrame,
    rules_threshold: float = 0.5,
) -> pd.DataFrame:
    """Apply stateless rules to a DataFrame. Returns a DataFrame with rules_score and rules_flagged columns.

    Expected columns: amount, merchant_category, transaction_time (or TransactionDT),
    location_country (optional, defaults to 'US').
    """
    scores = []
    for _, row in df.iterrows():
        amount = float(row.get("amount", row.get("TransactionAmt", 0)))
        category = str(row.get("merchant_category", row.get("ProductCD", "unknown")))
        country = str(row.get("location_country", "US"))

        txn_time = row.get("transaction_time", None)
        if txn_time is None:
            # For Kaggle data that uses TransactionDT (seconds from reference)
            txn_time = datetime(2024, 1, 1, hour=int(row.get("TransactionDT", 0)) % 24)
        elif isinstance(txn_time, str):
            txn_time = datetime.fromisoformat(txn_time)

        results = [
            high_amount_rule(amount),
            time_anomaly_rule(txn_time),
            merchant_risk_rule(category, amount),
        ]
        # Add geo rule without tracker (stateless — country risk only)
        c = country.upper()
        if c in HIGH_RISK_COUNTRIES:
            results.append(RuleResult("geo_anomaly", True, 0.3, f"High-risk country: {c}"))

        triggered = [r for r in results if r.triggered]
        scores.append(min(1.0, sum(r.weight for r in triggered)))

    result_df = pd.DataFrame({
        "rules_score": scores,
        "rules_flagged": [s >= rules_threshold for s in scores],
    })
    return result_df
