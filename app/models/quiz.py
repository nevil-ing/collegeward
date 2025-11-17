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

class Quiz(BaseModel):
    __tablename__ = "quizzes"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(200))
    subject_tags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[Optional[int]] = mapped_column(Integer)
    percentage: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(5, 2))
    time_taken: Mapped[Optional[int]] = mapped_column(Integer)
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="quizzes")
    questions: Mapped[List["QuizQuestion"]] = relationship("QuizQuestion", back_populates="quiz",cascade="all, delete-orphan")
