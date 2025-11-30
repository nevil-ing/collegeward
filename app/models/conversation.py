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

class Conversation(BaseModel):
    __tablename__ = "conversation"


    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(200))
    mode: Mapped[str] = mapped_column(String(20), nullable=False)

        # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversation")
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="conversation",cascade="all, delete-orphan")

