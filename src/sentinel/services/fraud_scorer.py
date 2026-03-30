from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.services.model_registry import ModelRegistry


class FraudScorer:
    """Scores transactions using the model registry.

    Supports champion (single best model) and ensemble (average all) modes.
    Falls back to a heuristic if no models are loaded.
    """

    def __init__(self, registry: ModelRegistry, mode: str = "champion"):
        self.registry = registry
        self.mode = mode  # "champion" or "ensemble"

    def score(self, transaction: dict) -> float:
        """Primary scoring interface — backward compatible dict -> float."""
        if not self.registry.has_models():
            return self._heuristic(transaction)

        features = self._extract_features(transaction)

        if self.mode == "ensemble":
            return self.registry.score_ensemble(features)
        return self.registry.score_champion(features)

    def score_all_models(self, transaction: dict) -> dict[str, float]:
        """Score with every loaded model for comparison."""
        if not self.registry.has_models():
            return {"heuristic": self._heuristic(transaction)}

        features = self._extract_features(transaction)
        return self.registry.score_all(features)

    def get_model_name(self) -> str:
        """Return the name of the model used for primary scoring."""
        if not self.registry.has_models():
            return "heuristic"
        if self.mode == "ensemble":
            return "ensemble"
        return self.registry.get_champion_model_name() or "unknown"

    @staticmethod
    def _extract_features(transaction: dict) -> pd.DataFrame:
        """Build feature vector matching the trained model's expected input.

        The BentoML models were trained on V1-V28 + Amount_log + Time_hour.
        For API transactions (which don't have PCA features), we build a
        zero-padded feature vector and let the model do its best. In production
        you'd have the real PCA pipeline here.
        """
        amount = float(transaction.get("amount", 0))
        txn_time = transaction.get("transaction_time", "")

        hour = 12.0  # default
        if txn_time:
            try:
                hour = float(pd.to_datetime(txn_time).hour)
            except Exception:
                pass

        row = {f"V{i}": 0.0 for i in range(1, 29)}
        row["Amount_log"] = float(np.log1p(amount))
        row["Time_hour"] = hour
        return pd.DataFrame([row])

    @staticmethod
    def _heuristic(transaction: dict) -> float:
        score = 0.0
        amount = float(transaction.get("amount", 0))
        if amount > 5000:
            score += 0.4
        elif amount > 2000:
            score += 0.2
        if transaction.get("is_online"):
            score += 0.15
        country = str(transaction.get("location_country", "")).upper()
        if country in {"NG", "RU", "CN", "BR", "PH", "VN", "IN"}:
            score += 0.25
        return min(score, 1.0)
