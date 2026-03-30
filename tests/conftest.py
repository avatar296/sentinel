import pytest
from httpx import ASGITransport, AsyncClient

from sentinel.main import app
from sentinel.services.fraud_scorer import FraudScorer


class MockScorer(FraudScorer):
    def __init__(self):
        self.pipeline = None

    def score(self, transaction: dict) -> float:
        return 0.15


@pytest.fixture
def mock_scorer():
    return MockScorer()


@pytest.fixture
async def client(mock_scorer):
    app.state.scorer = mock_scorer
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
