from typing import Optional
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import BaseModel
import enum


class SubscriptionTier(str, enum.Enum):
    FREE = "free"
    PRO_MONTHLY = "pro_monthly"
    PRO_YEARLY = "pro_yearly"


class Subscription(BaseModel):
    __tablename__ = "subscriptions"

    firebase_uid: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    
    # RevenueCat integration
    revenuecat_customer_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    
    # Subscription status
    tier: Mapped[SubscriptionTier] = mapped_column(
        SQLEnum(SubscriptionTier), 
        default=SubscriptionTier.FREE, 
        nullable=False
    )
    is_premium: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Billing info from RevenueCat
    product_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    original_purchase_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_in_trial: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    will_renew: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Usage tracking for rate limiting
    notes_uploaded_this_month: Mapped[int] = mapped_column(default=0, nullable=False)
    ai_questions_today: Mapped[int] = mapped_column(default=0, nullable=False)
    flashcards_generated_this_month: Mapped[int] = mapped_column(default=0, nullable=False)
    quizzes_generated_this_month: Mapped[int] = mapped_column(default=0, nullable=False)
    
    # Reset trackers
    usage_reset_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    daily_reset_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
