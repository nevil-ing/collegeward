from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from decimal import Decimal
from sqlalchemy import String, Integer, Boolean, Text, TIMESTAMP, ForeignKey, DECIMAL, ARRAY
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
import uuid
from app.db.base import BaseModel
if TYPE_CHECKING:
    from app.models.user import User


class SubjectMastery(BaseModel):
    """tracking fpr progress analytics"""
    __tablename__ = "subject_mastery"
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),nullable=False)
    subject_tag: Mapped[str] = mapped_column(String(100), nullable=False)
    mastery_percentage: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), default=0.0)
    total_questions_answered: Mapped[int] = mapped_column(Integer, default=0)
    correct_answers: Mapped[int] = mapped_column(Integer, default=0)
    flashcards_mastered: Mapped[int] = mapped_column(Integer, default=0)
    total_flashcards: Mapped[int] = mapped_column(Integer, default=0)
    last_activity_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    # Chat-based learning tracking
    chat_interactions: Mapped[int] = mapped_column(Integer, default=0)
    chat_correct_answers: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="subject_masteries")
