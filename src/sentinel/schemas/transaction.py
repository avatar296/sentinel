import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class TransactionCreate(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field("USD", min_length=3, max_length=3)
    merchant_category: str = Field(..., max_length=64)
    merchant_name: str = Field(..., max_length=255)
    card_last_four: str = Field(..., min_length=4, max_length=4, pattern=r"^\d{4}$")
    card_type: str = Field(..., max_length=20)
    transaction_time: datetime
    location_country: str = Field(..., min_length=2, max_length=3)
    is_online: bool


class TransactionResponse(BaseModel):
    id: uuid.UUID
    amount: float
    currency: str
    merchant_category: str
    merchant_name: str
    card_last_four: str
    card_type: str
    transaction_time: datetime
    location_country: str
    is_online: bool
    fraud_score: float | None
    is_flagged: bool | None
    created_at: datetime

    model_config = {"from_attributes": True}
