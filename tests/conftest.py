import pytest
from httpx import ASGITransport, AsyncClient

from sentinel.main import app
from sentinel.services.fraud_scorer import FraudScorer
from sentinel.services.model_registry import ModelRegistry
from sentinel.services.drift_detector import DriftDetector
from sentinel.services.velocity_tracker import VelocityTracker


class MockRegistry(ModelRegistry):
    """Registry that returns a fixed score without needing a real model."""

    def __init__(self, score: float = 0.15):
        super().__init__()
        self._mock_score = score

    def has_models(self) -> bool:
        return False  # Force heuristic fallback in FraudScorer


@pytest.fixture
def mock_registry():
    return MockRegistry()


@pytest.fixture
def mock_scorer(mock_registry):
    return FraudScorer(mock_registry)


@pytest.fixture
async def client(mock_scorer, mock_registry):
    app.state.registry = mock_registry
    app.state.scorer = mock_scorer
    app.state.tracker = VelocityTracker()
    app.state.drift_detector = DriftDetector()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
