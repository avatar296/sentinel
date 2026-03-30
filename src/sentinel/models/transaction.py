import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from sentinel.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    merchant_category: Mapped[str] = mapped_column(String(64), nullable=False)
    merchant_name: Mapped[str] = mapped_column(String(255), nullable=False)
    card_last_four: Mapped[str] = mapped_column(String(4), nullable=False)
    card_type: Mapped[str] = mapped_column(String(20), nullable=False)
    transaction_time: Mapped[datetime] = mapped_column(nullable=False)
    location_country: Mapped[str] = mapped_column(String(3), nullable=False)
    is_online: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fraud_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_flagged: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
