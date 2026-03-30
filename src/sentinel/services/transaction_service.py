import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.models.transaction import Transaction
from sentinel.schemas.transaction import TransactionCreate
from sentinel.services.drift_detector import DriftDetector
from sentinel.services.escalation import route
from sentinel.services.fraud_scorer import FraudScorer
from sentinel.services.rules_engine import evaluate_rules
from sentinel.services.velocity_tracker import VelocityTracker


async def create_transaction(
    db: AsyncSession,
    data: TransactionCreate,
    scorer: FraudScorer,
    tracker: VelocityTracker | None = None,
    drift_detector: DriftDetector | None = None,
    fraud_threshold: float = 0.8,
    review_threshold: float = 0.4,
    rules_threshold: float = 0.5,
) -> Transaction:
    tx_dict = data.model_dump()

    # ML layer
    fraud_score = scorer.score(tx_dict)

    # Rules layer
    rules_verdict = evaluate_rules(tx_dict, tracker=tracker, rules_threshold=rules_threshold)

    # Escalation routing
    escalation = route(
        ml_score=fraud_score,
        rules_verdict=rules_verdict,
        fraud_threshold=fraud_threshold,
        review_threshold=review_threshold,
    )

    txn = Transaction(
        **tx_dict,
        fraud_score=fraud_score,
        is_flagged=escalation.decision == "FLAG",
        rules_score=rules_verdict.rules_score,
        decision=escalation.decision,
        decision_reasons=escalation.reasons,
        model_used=scorer.get_model_name(),
    )
    db.add(txn)
    await db.commit()
    await db.refresh(txn)

    # Record in velocity tracker after successful commit
    if tracker is not None:
        tracker.record(data.card_last_four, data.location_country)

    # Record score for drift detection
    if drift_detector is not None:
        drift_detector.record(fraud_score)

    return txn


async def get_transaction(db: AsyncSession, txn_id: uuid.UUID) -> Transaction | None:
    return await db.get(Transaction, txn_id)


async def list_transactions(
    db: AsyncSession,
    limit: int = 20,
    offset: int = 0,
    flagged_only: bool = False,
) -> list[Transaction]:
    stmt = select(Transaction).order_by(Transaction.created_at.desc())
    if flagged_only:
        stmt = stmt.where(Transaction.is_flagged.is_(True))
    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def list_review_queue(
    db: AsyncSession,
    limit: int = 20,
    offset: int = 0,
) -> list[Transaction]:
    stmt = (
        select(Transaction)
        .where(Transaction.decision == "REVIEW")
        .order_by(Transaction.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())
