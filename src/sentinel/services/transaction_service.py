import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.models.transaction import Transaction
from sentinel.schemas.transaction import TransactionCreate
from sentinel.services.fraud_scorer import FraudScorer


async def create_transaction(
    db: AsyncSession,
    data: TransactionCreate,
    scorer: FraudScorer,
    threshold: float,
) -> Transaction:
    tx_dict = data.model_dump()
    fraud_score = scorer.score(tx_dict)

    txn = Transaction(
        **tx_dict,
        fraud_score=fraud_score,
        is_flagged=fraud_score >= threshold,
    )
    db.add(txn)
    await db.commit()
    await db.refresh(txn)
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
