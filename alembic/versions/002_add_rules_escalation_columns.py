"""Add rules_score, decision, decision_reasons columns

Revision ID: 002
Revises: 001
Create Date: 2026-03-30
"""

import sqlalchemy as sa

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("transactions", sa.Column("rules_score", sa.Float, nullable=True))
    op.add_column("transactions", sa.Column("decision", sa.String(10), nullable=True))
    op.add_column("transactions", sa.Column("decision_reasons", sa.String, nullable=True))


def downgrade():
    op.drop_column("transactions", "decision_reasons")
    op.drop_column("transactions", "decision")
    op.drop_column("transactions", "rules_score")
