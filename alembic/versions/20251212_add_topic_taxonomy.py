"""Add topic taxonomy tables

Revision ID: c20251212topics
Revises: b20251212xpcols
Create Date: 2025-12-12 14:15:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY


# revision identifiers, used by Alembic.
revision = "c20251212topics"
down_revision = "b20251212xpcols"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create topic_categories table
    op.create_table(
        'topic_categories',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('code', sa.String(50), unique=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('display_order', sa.Integer, default=0),
        sa.Column('source', sa.String(50), nullable=True),
        sa.Column('external_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Create topics table
    op.create_table(
        'topics',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('category_id', UUID(as_uuid=True), sa.ForeignKey('topic_categories.id', ondelete='CASCADE'), nullable=False),
        sa.Column('code', sa.String(50), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('keywords', ARRAY(sa.String(50)), nullable=True),
        sa.Column('parent_topic_id', UUID(as_uuid=True), sa.ForeignKey('topics.id', ondelete='SET NULL'), nullable=True),
        sa.Column('source', sa.String(50), nullable=True),
        sa.Column('external_id', sa.String(100), nullable=True),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('display_order', sa.Integer, default=0),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Create indexes
    op.create_index('ix_topics_category_code', 'topics', ['category_id', 'code'])
    op.create_index('ix_topics_keywords', 'topics', ['keywords'], postgresql_using='gin')
    
    # Create topic_sync_logs table
    op.create_table(
        'topic_sync_logs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('source', sa.String(50), nullable=False),
        sa.Column('sync_type', sa.String(20), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('topics_added', sa.Integer, default=0),
        sa.Column('topics_updated', sa.Integer, default=0),
        sa.Column('topics_removed', sa.Integer, default=0),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('sync_metadata', JSONB, nullable=True),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('completed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Add chat_interactions column to subject_mastery
    op.add_column(
        'subject_mastery',
        sa.Column('chat_interactions', sa.Integer, default=0, nullable=True)
    )
    op.add_column(
        'subject_mastery',
        sa.Column('chat_correct_answers', sa.Integer, default=0, nullable=True)
    )


def downgrade() -> None:
    op.drop_column('subject_mastery', 'chat_correct_answers')
    op.drop_column('subject_mastery', 'chat_interactions')
    op.drop_table('topic_sync_logs')
    op.drop_index('ix_topics_keywords')
    op.drop_index('ix_topics_category_code')
    op.drop_table('topics')
    op.drop_table('topic_categories')
