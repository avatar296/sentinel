from pathlib import Path

import joblib
import pandas as pd

from sentinel.ml.features import build_features


class FraudScorer:
    def __init__(self, model_path: str):
        if Path(model_path).exists():
            self.pipeline = joblib.load(model_path)
        else:
            self.pipeline = None

    def score(self, transaction: dict) -> float:
        if self.pipeline is None:
            return self._heuristic(transaction)

        df = pd.DataFrame([transaction])
        features = build_features(df)
        return float(self.pipeline.predict_proba(features)[0, 1])

    def _heuristic(self, transaction: dict) -> float:
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
