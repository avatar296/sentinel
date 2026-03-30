from __future__ import annotations

from pydantic import BaseModel

from sentinel.schemas.transaction import TransactionCreate


class ModelInfo(BaseModel):
    name: str
    tag: str
    is_champion: bool
    metadata: dict
    loaded_at: str


class StoreModelInfo(BaseModel):
    tag: str
    labels: dict
    metadata: dict


class LoadModelRequest(BaseModel):
    tag: str
    as_champion: bool = False


class SetModeRequest(BaseModel):
    mode: str  # "champion" or "ensemble"


class CompareRequest(BaseModel):
    transaction: TransactionCreate


class CompareResponse(BaseModel):
    scores: dict[str, float]
    ensemble_score: float | None = None
    champion_score: float | None = None
    champion_model: str | None = None
