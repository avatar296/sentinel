"""Add model_used column to transactions

Revision ID: 003
Revises: 002
Create Date: 2026-03-30
"""

import sqlalchemy as sa

from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("transactions", sa.Column("model_used", sa.String(64), nullable=True))


def downgrade():
    op.drop_column("transactions", "model_used")
