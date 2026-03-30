import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.config import settings
from sentinel.database import get_db
from sentinel.schemas.transaction import TransactionCreate, TransactionResponse
from sentinel.services import transaction_service

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post("", response_model=TransactionResponse, status_code=201)
async def submit_transaction(
    data: TransactionCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    scorer = request.app.state.scorer
    tracker = request.app.state.tracker
    drift_detector = request.app.state.drift_detector
    txn = await transaction_service.create_transaction(
        db,
        data,
        scorer,
        tracker=tracker,
        drift_detector=drift_detector,
        fraud_threshold=settings.FRAUD_THRESHOLD,
        review_threshold=settings.REVIEW_THRESHOLD,
        rules_threshold=settings.RULES_THRESHOLD,
    )
    return txn


@router.get("/review-queue", response_model=list[TransactionResponse])
async def review_queue(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return await transaction_service.list_review_queue(db, limit, offset)


@router.get("/{txn_id}", response_model=TransactionResponse)
async def get_transaction(txn_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    txn = await transaction_service.get_transaction(db, txn_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return txn


@router.get("", response_model=list[TransactionResponse])
async def list_transactions(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    flagged_only: bool = Query(False),
):
    return await transaction_service.list_transactions(db, limit, offset, flagged_only)
