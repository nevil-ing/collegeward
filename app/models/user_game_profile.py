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


class UserGameProfile(BaseModel):
    __tablename__ = "user_game_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),nullable=False, unique=True)
    total_xp: Mapped[int] = mapped_column(Integer, default=0)
    current_level: Mapped[int] = mapped_column(Integer, default=1)
    current_streak: Mapped[int] = mapped_column(Integer, default=0)
    longest_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_activity_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    streak_freeze_count: Mapped[int] = mapped_column(Integer, default=0)
    leaderboard_visible: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationship
    user: Mapped["User"] = relationship("User", back_populates="game_profile")
