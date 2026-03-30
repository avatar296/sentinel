from __future__ import annotations

import pandas as pd

from sentinel.ml.features import build_features
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

        df = pd.DataFrame([transaction])
        features = build_features(df)

        if self.mode == "ensemble":
            return self.registry.score_ensemble(features)
        return self.registry.score_champion(features)

    def score_all_models(self, transaction: dict) -> dict[str, float]:
        """Score with every loaded model for comparison."""
        if not self.registry.has_models():
            return {"heuristic": self._heuristic(transaction)}

        df = pd.DataFrame([transaction])
        features = build_features(df)
        return self.registry.score_all(features)

    def get_model_name(self) -> str:
        """Return the name of the model used for primary scoring."""
        if not self.registry.has_models():
            return "heuristic"
        if self.mode == "ensemble":
            return "ensemble"
        return self.registry.get_champion_model_name() or "unknown"

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
