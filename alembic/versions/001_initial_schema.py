"""Initial schema - transactions table

Revision ID: 001
Revises:
Create Date: 2026-03-30
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "transactions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("currency", sa.String(3), server_default="USD"),
        sa.Column("merchant_category", sa.String(64), nullable=False),
        sa.Column("merchant_name", sa.String(255), nullable=False),
        sa.Column("card_last_four", sa.String(4), nullable=False),
        sa.Column("card_type", sa.String(20), nullable=False),
        sa.Column("transaction_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("location_country", sa.String(3), nullable=False),
        sa.Column("is_online", sa.Boolean, nullable=False),
        sa.Column("fraud_score", sa.Float, nullable=True),
        sa.Column("is_flagged", sa.Boolean, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("transactions")
