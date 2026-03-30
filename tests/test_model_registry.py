import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sentinel.services.fraud_scorer import FraudScorer
from sentinel.services.model_registry import ModelRegistry


def _make_pipeline():
    """Create a tiny fitted pipeline for testing."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


class TestModelRegistry:
    def test_empty_registry(self):
        reg = ModelRegistry()
        assert not reg.has_models()
        assert reg.list_models() == []
        assert reg.champion_name is None

    def test_load_pipeline_directly(self):
        reg = ModelRegistry()
        pipe = _make_pipeline()
        reg.load_pipeline_directly("test-model", pipe, metadata={"auc_pr": 0.9}, as_champion=True)

        assert reg.has_models()
        assert reg.champion_name == "test-model"
        models = reg.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "test-model"
        assert models[0]["is_champion"] is True

    def test_champion_selection(self):
        reg = ModelRegistry()
        pipe1 = _make_pipeline()
        pipe2 = _make_pipeline()

        reg.load_pipeline_directly("model-a", pipe1, as_champion=True)
        reg.load_pipeline_directly("model-b", pipe2, as_champion=False)

        assert reg.champion_name == "model-a"

        reg.set_champion("model-b")
        assert reg.champion_name == "model-b"

        # Old champion should be demoted
        models = {m["name"]: m for m in reg.list_models()}
        assert models["model-a"]["is_champion"] is False
        assert models["model-b"]["is_champion"] is True

    def test_remove_model(self):
        reg = ModelRegistry()
        pipe = _make_pipeline()
        reg.load_pipeline_directly("to-remove", pipe, as_champion=True)

        assert reg.remove_model("to-remove")
        assert not reg.has_models()
        assert reg.champion_name is None

    def test_remove_champion_promotes_next(self):
        reg = ModelRegistry()
        reg.load_pipeline_directly("a", _make_pipeline(), as_champion=True)
        reg.load_pipeline_directly("b", _make_pipeline())

        reg.remove_model("a")
        assert reg.champion_name == "b"

    def test_remove_nonexistent(self):
        reg = ModelRegistry()
        assert not reg.remove_model("nope")

    def test_score_champion(self):
        reg = ModelRegistry()
        pipe = _make_pipeline()
        reg.load_pipeline_directly("champ", pipe, as_champion=True)

        features = pd.DataFrame([[1.0, 2.0]])
        score = reg.score_champion(features)
        assert 0.0 <= score <= 1.0

    def test_score_all(self):
        reg = ModelRegistry()
        reg.load_pipeline_directly("a", _make_pipeline(), as_champion=True)
        reg.load_pipeline_directly("b", _make_pipeline())

        features = pd.DataFrame([[1.0, 2.0]])
        scores = reg.score_all(features)
        assert "a" in scores and "b" in scores
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_ensemble_scoring(self):
        reg = ModelRegistry()
        reg.load_pipeline_directly("a", _make_pipeline(), as_champion=True)
        reg.load_pipeline_directly("b", _make_pipeline())

        features = pd.DataFrame([[1.0, 2.0]])
        ensemble = reg.score_ensemble(features)
        all_scores = reg.score_all(features)
        expected = np.mean(list(all_scores.values()))
        assert abs(ensemble - expected) < 1e-6

    def test_set_champion_nonexistent_raises(self):
        reg = ModelRegistry()
        try:
            reg.set_champion("ghost")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


class TestFraudScorerWithRegistry:
    def test_heuristic_fallback(self):
        reg = ModelRegistry()
        scorer = FraudScorer(reg)
        txn = {"amount": 50.0, "is_online": False, "location_country": "US"}
        score = scorer.score(txn)
        assert score == 0.0  # low risk heuristic

    def test_champion_mode(self):
        reg = ModelRegistry()
        reg.load_pipeline_directly("m", _make_pipeline(), as_champion=True)
        scorer = FraudScorer(reg, mode="champion")
        # Can't easily test the exact score since features won't match,
        # but verify it doesn't fall back to heuristic
        assert scorer.get_model_name() == "m"

    def test_ensemble_mode(self):
        reg = ModelRegistry()
        reg.load_pipeline_directly("a", _make_pipeline(), as_champion=True)
        reg.load_pipeline_directly("b", _make_pipeline())
        scorer = FraudScorer(reg, mode="ensemble")
        assert scorer.get_model_name() == "ensemble"

    def test_score_all_models(self):
        reg = ModelRegistry()
        scorer = FraudScorer(reg)
        txn = {"amount": 8000.0, "is_online": True, "location_country": "NG"}
        result = scorer.score_all_models(txn)
        assert "heuristic" in result  # falls back when no models loaded
