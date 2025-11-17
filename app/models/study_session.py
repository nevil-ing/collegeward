from typing import Optional, List
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from decimal import Decimal
from sqlalchemy import String, Integer, Boolean, Text, TIMESTAMP, ForeignKey, DECIMAL, ARRAY
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
import uuid
from app.db.base import BaseModel
from app.models.user import User

class StudySession(BaseModel):
    __tablename__ = "study_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),nullable=False)
    activity_type: Mapped[str] = mapped_column(String(20), nullable=False)
    activity_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    subject_tags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    ended_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="study_sessions")
