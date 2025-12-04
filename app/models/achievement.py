from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from decimal import Decimal
from sqlalchemy import String, Integer, Boolean, Text, TIMESTAMP, ForeignKey, DECIMAL, ARRAY
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.db.base import BaseModel
if TYPE_CHECKING:
   from app.models.user_achievement import UserAchievement

class Achievement(BaseModel):
    __tablename__ = "achievements"

    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(30),nullable=False)
    badge_icon: Mapped[str] = mapped_column(String(50), nullable=False)
    badge_color: Mapped[str] = mapped_column(String(20), nullable=False)
    xp_reward: Mapped[int] = mapped_column(Integer, default=0)
    criteria: Mapped[dict] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    user_achievements: Mapped[List["UserAchievement"]] = relationship("UserAchievement", back_populates="achievement",
                                                                      cascade="all, delete-orphan")

