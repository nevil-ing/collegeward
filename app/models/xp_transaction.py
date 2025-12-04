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

class XPTransaction(BaseModel):
    __tablename__ = "xp_transactions"


user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),nullable=False)
activity_type: Mapped[str] = mapped_column(String(30),nullable=False)
activity_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
xp_earned: Mapped[int] = mapped_column(Integer, nullable=False)
multiplier: Mapped[Decimal] = mapped_column(DECIMAL(3, 2), default=1.0)
reason: Mapped[str] = mapped_column(String(200), nullable=False)
transaction_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)

# Relationships
user: Mapped["User"] = relationship("User")
