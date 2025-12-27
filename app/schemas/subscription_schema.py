from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class SubscriptionTierEnum(str, Enum):
    FREE = "free"
    PRO_MONTHLY = "pro_monthly"
    PRO_YEARLY = "pro_yearly"


class SubscriptionStatus(BaseModel):
    """Response model for subscription status"""
    tier: SubscriptionTierEnum = SubscriptionTierEnum.FREE
    is_premium: bool = False
    expires_date: Optional[datetime] = None
    will_renew: bool = True
    is_in_trial: bool = False


class UsageLimits(BaseModel):
    """Current usage vs limits"""
    notes_uploaded: int = 0
    notes_limit: int = 5  # Free limit
    ai_questions_used: int = 0
    ai_questions_limit: int = 20  # Free limit per day
    flashcards_generated: int = 0
    flashcards_limit: int = 50  # Free limit per month
    quizzes_generated: int = 0
    quizzes_limit: int = 5  # Free limit per month


class SubscriptionResponse(BaseModel):
    """Full subscription info for the app"""
    subscription: SubscriptionStatus
    usage: UsageLimits
    can_upload_notes: bool = True
    can_ask_ai: bool = True
    can_generate_flashcards: bool = True
    can_generate_quiz: bool = True


class RevenueCatWebhookEvent(BaseModel):
    """RevenueCat webhook payload"""
    event: dict
    api_version: str = Field(default="1.0")


class RevenueCatEventType(str, Enum):
    INITIAL_PURCHASE = "INITIAL_PURCHASE"
    RENEWAL = "RENEWAL"
    CANCELLATION = "CANCELLATION"
    EXPIRATION = "EXPIRATION"
    PRODUCT_CHANGE = "PRODUCT_CHANGE"
    BILLING_ISSUE = "BILLING_ISSUE"
    SUBSCRIBER_ALIAS = "SUBSCRIBER_ALIAS"
