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
    from app.models.quiz import Quiz

class QuizQuestion(BaseModel):
    __tablename__ = "quiz_questions"

    quiz_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("quizzes.id", ondelete="CASCADE"),nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[dict] = mapped_column(JSONB, nullable=False)
    correct_answer: Mapped[int] = mapped_column(Integer, nullable=False)
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    user_answer: Mapped[Optional[int]] = mapped_column(Integer)
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    question_order: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    quiz: Mapped["Quiz"] = relationship("Quiz", back_populates="questions")
