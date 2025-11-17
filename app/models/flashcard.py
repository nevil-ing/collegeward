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

class Flashcard(BaseModel):
    __tablename__ = "flashcards"
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                                               nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    subject_tags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    difficulty_level: Mapped[int] = mapped_column(Integer, default=1)
    leitner_box: Mapped[int] = mapped_column(Integer, default=1)
    next_review_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    times_reviewed: Mapped[int] = mapped_column(Integer, default=0)
    times_correct: Mapped[int] = mapped_column(Integer, default=0)
    created_from: Mapped[Optional[str]] = mapped_column(String(20))
    source_reference: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="flashcards")

