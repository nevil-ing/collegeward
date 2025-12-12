"""
Topic Taxonomy Models for Hierarchical Topic Classification

Supports expandable topics from external sources (MeSH, UMLS) 
synced to database for configuration.
"""

from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy import String, Integer, Boolean, Text, TIMESTAMP, ForeignKey, Index
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
import uuid
from app.db.base import BaseModel


class TopicCategory(BaseModel):
    """
    Top-level category for topic taxonomy
    
    Examples: basic_sciences, clinical_sciences, nursing, pharmacy
    """
    __tablename__ = "topic_categories"
    
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    display_order: Mapped[int] = mapped_column(Integer, default=0)
    
    # External source reference
    source: Mapped[Optional[str]] = mapped_column(String(50))  # "mesh", "umls", "custom"
    external_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Relationships
    topics: Mapped[List["Topic"]] = relationship("Topic", back_populates="category", cascade="all, delete-orphan")


class Topic(BaseModel):
    """
    Individual topic within a category
    
    Examples: anatomy, cardiology, pharmacology
    """
    __tablename__ = "topics"
    
    category_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("topic_categories.id", ondelete="CASCADE"),
        nullable=False
    )
    
    code: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Keywords for matching
    keywords: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String(50)))
    
    # Hierarchical structure
    parent_topic_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("topics.id", ondelete="SET NULL")
    )
    
    # External source reference
    source: Mapped[Optional[str]] = mapped_column(String(50))  # "mesh", "umls", "custom"
    external_id: Mapped[Optional[str]] = mapped_column(String(100))  # MeSH ID, UMLS CUI
    
    # Metadata
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    display_order: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    category: Mapped["TopicCategory"] = relationship("TopicCategory", back_populates="topics")
    parent_topic: Mapped[Optional["Topic"]] = relationship("Topic", remote_side="Topic.id", backref="subtopics")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_topics_category_code', 'category_id', 'code'),
        Index('ix_topics_keywords', 'keywords', postgresql_using='gin'),
    )


class TopicSyncLog(BaseModel):
    """Log of taxonomy sync operations from external sources"""
    __tablename__ = "topic_sync_logs"
    
    source: Mapped[str] = mapped_column(String(50), nullable=False)  # "mesh", "umls"
    sync_type: Mapped[str] = mapped_column(String(20), nullable=False)  # "full", "incremental"
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # "success", "failed", "partial"
    
    topics_added: Mapped[int] = mapped_column(Integer, default=0)
    topics_updated: Mapped[int] = mapped_column(Integer, default=0)
    topics_removed: Mapped[int] = mapped_column(Integer, default=0)
    
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    sync_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
