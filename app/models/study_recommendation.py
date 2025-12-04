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


class StudyRecommendation(BaseModel):
    """recommendation model for personalized suggestions"""
    __tablename__ = "study_recommendations"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),nullable=False)
    recommendation_type: Mapped[str] = mapped_column(String(30),nullable=False)
    subject_tag: Mapped[Optional[str]] = mapped_column(String(100))
    priority_score: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    action_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="study_recommendations")
