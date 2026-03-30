from fastapi import APIRouter, HTTPException, Request

from sentinel.schemas.model import (
    CompareRequest,
    CompareResponse,
    LoadModelRequest,
    ModelInfo,
    SetModeRequest,
    StoreModelInfo,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
async def list_models(request: Request):
    """List all currently loaded models."""
    registry = request.app.state.registry
    return registry.list_models()


@router.get("/store", response_model=list[StoreModelInfo])
async def list_store(request: Request):
    """List all models in the BentoML store (not just loaded ones)."""
    registry = request.app.state.registry
    return registry.list_store()


@router.post("/load", response_model=ModelInfo)
async def load_model(body: LoadModelRequest, request: Request):
    """Load a model from the BentoML store into memory."""
    registry = request.app.state.registry
    try:
        name = registry.load_model(body.tag, as_champion=body.as_champion)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    models = registry.list_models()
    return next(m for m in models if m["name"] == name)


@router.delete("/{name}")
async def remove_model(name: str, request: Request):
    """Unload a model from memory."""
    registry = request.app.state.registry
    if not registry.remove_model(name):
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return {"status": "removed", "name": name}


@router.put("/{name}/champion")
async def set_champion(name: str, request: Request):
    """Promote a loaded model to champion."""
    registry = request.app.state.registry
    try:
        registry.set_champion(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not loaded")
    return {"status": "promoted", "champion": name}


@router.put("/mode")
async def set_mode(body: SetModeRequest, request: Request):
    """Switch scoring mode between 'champion' and 'ensemble'."""
    if body.mode not in ("champion", "ensemble"):
        raise HTTPException(status_code=400, detail="Mode must be 'champion' or 'ensemble'")
    scorer = request.app.state.scorer
    scorer.mode = body.mode
    return {"status": "updated", "mode": body.mode}


@router.post("/compare", response_model=CompareResponse)
async def compare_models(body: CompareRequest, request: Request):
    """Score a transaction with all loaded models for side-by-side comparison."""
    scorer = request.app.state.scorer
    registry = request.app.state.registry
    tx_dict = body.transaction.model_dump()

    all_scores = scorer.score_all_models(tx_dict)

    champion_name = registry.get_champion_model_name()
    champion_score = all_scores.get(champion_name) if champion_name else None

    ensemble_score = None
    if len(all_scores) > 1:
        import numpy as np
        ensemble_score = float(np.mean(list(all_scores.values())))

    return CompareResponse(
        scores=all_scores,
        ensemble_score=ensemble_score,
        champion_score=champion_score,
        champion_model=champion_name,
    )
