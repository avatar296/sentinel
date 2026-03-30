from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sentinel.api.router import api_router
from sentinel.config import settings
from sentinel.services.fraud_scorer import FraudScorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = FraudScorer(settings.MODEL_PATH)
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
