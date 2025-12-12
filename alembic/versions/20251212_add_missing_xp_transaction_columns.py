"""Add missing columns to xp_transactions table

Revision ID: b20251212xpcols
Revises: a20251209xpuser
Create Date: 2025-12-12 13:20:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision = "b20251212xpcols"
down_revision = "a20251209xpuser"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add missing columns to xp_transactions table
    # Using batch_alter_table for safety with existing data
    
    # Check if columns exist before adding (prevents errors if partially migrated)
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_columns = [col['name'] for col in inspector.get_columns('xp_transactions')]
    
    if 'activity_type' not in existing_columns:
        op.add_column(
            'xp_transactions',
            sa.Column('activity_type', sa.String(30), nullable=True)
        )
        # Set default value for existing rows
        op.execute("UPDATE xp_transactions SET activity_type = 'unknown' WHERE activity_type IS NULL")
        # Make it non-nullable after setting defaults
        op.alter_column('xp_transactions', 'activity_type', nullable=False)
    
    if 'activity_id' not in existing_columns:
        op.add_column(
            'xp_transactions',
            sa.Column('activity_id', UUID(as_uuid=True), nullable=True)
        )
    
    if 'xp_earned' not in existing_columns:
        op.add_column(
            'xp_transactions',
            sa.Column('xp_earned', sa.Integer(), nullable=True)
        )
        # Set default value for existing rows
        op.execute("UPDATE xp_transactions SET xp_earned = 0 WHERE xp_earned IS NULL")
        # Make it non-nullable after setting defaults
        op.alter_column('xp_transactions', 'xp_earned', nullable=False)
    
    if 'multiplier' not in existing_columns:
        op.add_column(
            'xp_transactions',
            sa.Column('multiplier', sa.DECIMAL(3, 2), server_default='1.0', nullable=True)
        )
    
    if 'reason' not in existing_columns:
        op.add_column(
            'xp_transactions',
            sa.Column('reason', sa.String(200), nullable=True)
        )
        # Set default value for existing rows
        op.execute("UPDATE xp_transactions SET reason = 'Legacy transaction' WHERE reason IS NULL")
        # Make it non-nullable after setting defaults
        op.alter_column('xp_transactions', 'reason', nullable=False)
    
    if 'transaction_metadata' not in existing_columns:
        op.add_column(
            'xp_transactions',
            sa.Column('transaction_metadata', JSONB, nullable=True)
        )
    
    # Make user_id non-nullable if it was nullable before
    op.alter_column('xp_transactions', 'user_id', nullable=False)


def downgrade() -> None:
    op.drop_column('xp_transactions', 'transaction_metadata')
    op.drop_column('xp_transactions', 'reason')
    op.drop_column('xp_transactions', 'multiplier')
    op.drop_column('xp_transactions', 'xp_earned')
    op.drop_column('xp_transactions', 'activity_id')
    op.drop_column('xp_transactions', 'activity_type')
