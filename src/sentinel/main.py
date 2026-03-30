from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sentinel.api.router import api_router
from sentinel.config import settings
from sentinel.services.fraud_scorer import FraudScorer
from sentinel.services.model_registry import ModelRegistry
from sentinel.services.velocity_tracker import VelocityTracker


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()

    # Try loading models from BentoML store
    loaded = registry.load_from_store()

    # Fallback: load legacy joblib if no BentoML models found
    if not registry.has_models() and Path(settings.MODEL_PATH).exists():
        pipeline = joblib.load(settings.MODEL_PATH)
        registry.load_pipeline_directly(
            "legacy-xgboost", pipeline, metadata={}, as_champion=True,
        )

    app.state.registry = registry
    app.state.scorer = FraudScorer(registry, mode=settings.SCORING_MODE)
    app.state.tracker = VelocityTracker()
    yield


app = FastAPI(
    title="Sentinel",
    description="Transaction fraud detection API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_PREFIX)
