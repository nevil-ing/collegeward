"""Add user_id column to xp_transactions

Revision ID: 20251209_add_user_id_to_xp_transactions
Revises: 
Create Date: 2025-12-09 21:34:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20251209_add_user_id_to_xp_transactions"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "xp_transactions",
        sa.Column(
            "user_id",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("xp_transactions", "user_id")
