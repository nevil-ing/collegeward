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

class StudyStreak(BaseModel):
    __tablename__ = "study_streaks"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),nullable=False)
    streak_date: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    activities_completed: Mapped[int] = mapped_column(Integer, default=1)
    total_study_time: Mapped[int] = mapped_column(Integer, default=0)
    streak_maintained: Mapped[bool] = mapped_column(Boolean, default=True)
    freeze_used: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="study_streaks")
