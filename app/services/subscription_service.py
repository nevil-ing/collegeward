from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models.subscription import Subscription, SubscriptionTier
from app.models.user import User
from app.schemas.subscription_schema import (
    SubscriptionResponse, SubscriptionStatus, UsageLimits, SubscriptionTierEnum
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# Rate limits by tier
LIMITS = {
    SubscriptionTier.FREE: {
        "notes_per_month": 5,
        "ai_questions_per_day": 20,
        "flashcards_per_month": 50,
        "quizzes_per_month": 5,
    },
    SubscriptionTier.PRO_MONTHLY: {
        "notes_per_month": float("inf"),
        "ai_questions_per_day": float("inf"),
        "flashcards_per_month": float("inf"),
        "quizzes_per_month": float("inf"),
    },
    SubscriptionTier.PRO_YEARLY: {
        "notes_per_month": float("inf"),
        "ai_questions_per_day": float("inf"),
        "flashcards_per_month": float("inf"),
        "quizzes_per_month": float("inf"),
    },
}


class SubscriptionService:
    """Service for managing user subscriptions and rate limiting"""

    async def get_or_create_subscription(
        self, db: AsyncSession, firebase_uid: str
    ) -> Subscription:
        """Get or create subscription record for a user"""
        result = await db.execute(
            select(Subscription).where(Subscription.firebase_uid == firebase_uid)
        )
        subscription = result.scalar_one_or_none()

        if not subscription:
            subscription = Subscription(
                firebase_uid=firebase_uid,
                tier=SubscriptionTier.FREE,
                is_premium=False,
                usage_reset_date=datetime.utcnow(),
                daily_reset_date=datetime.utcnow(),
            )
            db.add(subscription)
            await db.commit()
            await db.refresh(subscription)
            logger.info(f"Created subscription for user {firebase_uid}")

        return subscription

    async def get_subscription_status(
        self, db: AsyncSession, firebase_uid: str
    ) -> SubscriptionResponse:
        """Get full subscription status with usage info"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)
        
        # Reset usage if needed
        await self._reset_usage_if_needed(db, subscription)

        limits = LIMITS[subscription.tier]
        is_premium = subscription.is_premium

        usage = UsageLimits(
            notes_uploaded=subscription.notes_uploaded_this_month,
            notes_limit=int(limits["notes_per_month"]) if limits["notes_per_month"] != float("inf") else 999999,
            ai_questions_used=subscription.ai_questions_today,
            ai_questions_limit=int(limits["ai_questions_per_day"]) if limits["ai_questions_per_day"] != float("inf") else 999999,
            flashcards_generated=subscription.flashcards_generated_this_month,
            flashcards_limit=int(limits["flashcards_per_month"]) if limits["flashcards_per_month"] != float("inf") else 999999,
            quizzes_generated=subscription.quizzes_generated_this_month,
            quizzes_limit=int(limits["quizzes_per_month"]) if limits["quizzes_per_month"] != float("inf") else 999999,
        )

        status = SubscriptionStatus(
            tier=SubscriptionTierEnum(subscription.tier.value),
            is_premium=is_premium,
            expires_date=subscription.expires_date,
            will_renew=subscription.will_renew,
            is_in_trial=subscription.is_in_trial,
        )

        return SubscriptionResponse(
            subscription=status,
            usage=usage,
            can_upload_notes=is_premium or subscription.notes_uploaded_this_month < limits["notes_per_month"],
            can_ask_ai=is_premium or subscription.ai_questions_today < limits["ai_questions_per_day"],
            can_generate_flashcards=is_premium or subscription.flashcards_generated_this_month < limits["flashcards_per_month"],
            can_generate_quiz=is_premium or subscription.quizzes_generated_this_month < limits["quizzes_per_month"],
        )

    async def check_can_upload_note(self, db: AsyncSession, firebase_uid: str) -> tuple[bool, str]:
        """Check if user can upload a note. Returns (allowed, message)"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)
        await self._reset_usage_if_needed(db, subscription)

        if subscription.is_premium:
            return True, "Premium user - unlimited uploads"

        limits = LIMITS[subscription.tier]
        if subscription.notes_uploaded_this_month >= limits["notes_per_month"]:
            return False, f"Monthly upload limit reached ({int(limits['notes_per_month'])} notes). Upgrade to Pro for unlimited uploads."

        return True, f"Uploads remaining: {int(limits['notes_per_month']) - subscription.notes_uploaded_this_month}"

    async def check_can_ask_ai(self, db: AsyncSession, firebase_uid: str) -> tuple[bool, str]:
        """Check if user can ask AI tutor"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)
        await self._reset_usage_if_needed(db, subscription)

        if subscription.is_premium:
            return True, "Premium user - unlimited questions"

        limits = LIMITS[subscription.tier]
        if subscription.ai_questions_today >= limits["ai_questions_per_day"]:
            return False, f"Daily AI question limit reached ({int(limits['ai_questions_per_day'])} questions). Upgrade to Pro for unlimited access."

        return True, f"Questions remaining today: {int(limits['ai_questions_per_day']) - subscription.ai_questions_today}"

    async def check_can_generate_flashcards(self, db: AsyncSession, firebase_uid: str) -> tuple[bool, str]:
        """Check if user can generate flashcards"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)
        await self._reset_usage_if_needed(db, subscription)

        if subscription.is_premium:
            return True, "Premium user - unlimited flashcards"

        limits = LIMITS[subscription.tier]
        if subscription.flashcards_generated_this_month >= limits["flashcards_per_month"]:
            return False, f"Monthly flashcard limit reached ({int(limits['flashcards_per_month'])} flashcards). Upgrade to Pro for unlimited."

        return True, f"Flashcards remaining: {int(limits['flashcards_per_month']) - subscription.flashcards_generated_this_month}"

    async def check_can_generate_quiz(self, db: AsyncSession, firebase_uid: str) -> tuple[bool, str]:
        """Check if user can generate a quiz"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)
        await self._reset_usage_if_needed(db, subscription)

        if subscription.is_premium:
            return True, "Premium user - unlimited quizzes"

        limits = LIMITS[subscription.tier]
        if subscription.quizzes_generated_this_month >= limits["quizzes_per_month"]:
            return False, f"Monthly quiz limit reached ({int(limits['quizzes_per_month'])} quizzes). Upgrade to Pro for unlimited."

        return True, f"Quizzes remaining: {int(limits['quizzes_per_month']) - subscription.quizzes_generated_this_month}"

    async def increment_note_upload(self, db: AsyncSession, firebase_uid: str) -> None:
        """Increment note upload counter"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)
        subscription.notes_uploaded_this_month += 1
        await db.commit()

    async def increment_ai_question(self, db: AsyncSession, firebase_uid: str) -> None:
        """Increment AI question counter"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)
        subscription.ai_questions_today += 1
        await db.commit()

    async def increment_flashcard_generation(self, db: AsyncSession, firebase_uid: str, count: int = 1) -> None:
        """Increment flashcard generation counter"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)
        subscription.flashcards_generated_this_month += count
        await db.commit()

    async def increment_quiz_generation(self, db: AsyncSession, firebase_uid: str) -> None:
        """Increment quiz generation counter"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)
        subscription.quizzes_generated_this_month += 1
        await db.commit()

    async def update_subscription_from_revenuecat(
        self,
        db: AsyncSession,
        firebase_uid: str,
        product_id: str,
        expires_date: Optional[datetime],
        is_active: bool,
        is_trial: bool = False,
        will_renew: bool = True,
    ) -> Subscription:
        """Update subscription from RevenueCat webhook"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)

        # Determine tier from product_id
        if "yearly" in product_id.lower():
            tier = SubscriptionTier.PRO_YEARLY
        elif "monthly" in product_id.lower():
            tier = SubscriptionTier.PRO_MONTHLY
        else:
            tier = SubscriptionTier.FREE

        subscription.tier = tier if is_active else SubscriptionTier.FREE
        subscription.is_premium = is_active
        subscription.product_id = product_id
        subscription.expires_date = expires_date
        subscription.is_in_trial = is_trial
        subscription.will_renew = will_renew

        if is_active and not subscription.original_purchase_date:
            subscription.original_purchase_date = datetime.utcnow()

        await db.commit()
        await db.refresh(subscription)

        logger.info(f"Updated subscription for {firebase_uid}: tier={tier}, is_premium={is_active}")
        return subscription

    async def cancel_subscription(self, db: AsyncSession, firebase_uid: str) -> Subscription:
        """Handle subscription cancellation"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)
        subscription.will_renew = False
        # Note: is_premium stays true until expires_date
        await db.commit()
        await db.refresh(subscription)
        logger.info(f"Subscription cancelled for {firebase_uid}, will expire at {subscription.expires_date}")
        return subscription

    async def expire_subscription(self, db: AsyncSession, firebase_uid: str) -> Subscription:
        """Handle subscription expiration"""
        subscription = await self.get_or_create_subscription(db, firebase_uid)
        subscription.tier = SubscriptionTier.FREE
        subscription.is_premium = False
        subscription.will_renew = False
        await db.commit()
        await db.refresh(subscription)
        logger.info(f"Subscription expired for {firebase_uid}")
        return subscription

    async def _reset_usage_if_needed(self, db: AsyncSession, subscription: Subscription) -> None:
        """Reset usage counters if period has elapsed"""
        now = datetime.utcnow()

        # Reset monthly counters
        if subscription.usage_reset_date:
            if (now - subscription.usage_reset_date).days >= 30:
                subscription.notes_uploaded_this_month = 0
                subscription.flashcards_generated_this_month = 0
                subscription.quizzes_generated_this_month = 0
                subscription.usage_reset_date = now
                await db.commit()
                logger.info(f"Reset monthly usage for {subscription.firebase_uid}")

        # Reset daily counters
        if subscription.daily_reset_date:
            if subscription.daily_reset_date.date() < now.date():
                subscription.ai_questions_today = 0
                subscription.daily_reset_date = now
                await db.commit()
                logger.info(f"Reset daily usage for {subscription.firebase_uid}")


# Singleton instance
subscription_service = SubscriptionService()
