import pytest

SAMPLE_TRANSACTION = {
    "amount": 125.50,
    "currency": "USD",
    "merchant_category": "grocery",
    "merchant_name": "Whole Foods",
    "card_last_four": "4242",
    "card_type": "visa",
    "transaction_time": "2025-06-15T14:30:00Z",
    "location_country": "US",
    "is_online": False,
}


@pytest.mark.asyncio
async def test_submit_transaction(client):
    response = await client.post("/api/v1/transactions", json=SAMPLE_TRANSACTION)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert "fraud_score" in data
    assert isinstance(data["fraud_score"], float)
    assert data["is_flagged"] is not None


@pytest.mark.asyncio
async def test_submit_invalid_transaction(client):
    bad_data = {**SAMPLE_TRANSACTION, "amount": -50}
    response = await client.post("/api/v1/transactions", json=bad_data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_invalid_card(client):
    bad_data = {**SAMPLE_TRANSACTION, "card_last_four": "abc"}
    response = await client.post("/api/v1/transactions", json=bad_data)
    assert response.status_code == 422
