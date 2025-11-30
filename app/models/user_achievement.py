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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.achievement import Achievement

class UserAchievement(BaseModel):
    __tablename__ = "user_achievements"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),nullable=False)
    achievement_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True),ForeignKey("achievements.id", ondelete="CASCADE"), nullable=False)
    earned_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    progress_data: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Relationships
    user: Mapped["User"] = relationship("User")
    achievement: Mapped["Achievement"] = relationship("Achievement", back_populates="user_achievements")
