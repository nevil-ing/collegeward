'''
import logging
import asyncio
from datetime import datetime, timedelta, time
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, or_

#from app.db.session import SessionLocal
from app.models.user_game_profile import UserGameProfile
from app.models.user import User
from app.services.notification_service import NotificationService
from app.services.gamification_service import GamificationService

logger = logging.getLogger(__name__)


class NotificationScheduler:
    """Service for scheduling and sending gamification notifications"""

    def __init__(self):
        self.notification_service = NotificationService()
        self.running = False

    async def start_scheduler(self):
        """Start the notification scheduler"""
        self.running = True
        logger.info("Notification scheduler started")

        while self.running:
            try:
                await self._check_and_send_notifications()
                # Check every hour
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Error in notification scheduler: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    def stop_scheduler(self):
        """Stop the notification scheduler"""
        self.running = False
        logger.info("Notification scheduler stopped")

    async def _check_and_send_notifications(self):
        """Check for users who need notifications and send them"""
        db = SessionLocal()
        try:
            # Check for streak reminders
            await self._send_streak_reminders(db)

            # Check for daily reminders
            await self._send_daily_reminders(db)

        finally:
            db.close()

    async def _send_streak_reminders(self, db: Session):
        """Send streak reminder notifications to users at risk"""
        try:
            # Find users whose streaks are at risk (no activity in last 20 hours)
            cutoff_time = datetime.now() - timedelta(hours=20)

            at_risk_users = db.query(UserGameProfile, User).join(User).filter(
                and_(
                    UserGameProfile.current_streak > 0,
                    UserGameProfile.last_activity_date < cutoff_time,
                    UserGameProfile.last_activity_date > datetime.now() - timedelta(hours=24)
                )
            ).all()

            notifications_sent = 0

            for profile, user in at_risk_users:
                # Calculate hours remaining
                hours_since_activity = (datetime.now() - profile.last_activity_date).total_seconds() / 3600
                hours_remaining = max(0, int(24 - hours_since_activity))

                if hours_remaining > 0:
                    # Check notification preferences (mock implementation)
                    preferences = self.notification_service.get_notification_preferences(user.id)

                    if preferences.get("streak_reminders", True):
                        success = await self.notification_service.send_streak_reminder(
                            user_id=user.id,
                            current_streak=profile.current_streak,
                            hours_remaining=hours_remaining
                        )

                        if success:
                            notifications_sent += 1

            if notifications_sent > 0:
                logger.info(f"Sent {notifications_sent} streak reminder notifications")

        except Exception as e:
            logger.error(f"Error sending streak reminders: {e}")

    async def _send_daily_reminders(self, db: Session):
        """Send daily study reminders"""
        try:
            # Only send reminders at specific times (e.g., 9 AM, 6 PM)
            current_hour = datetime.now().hour

            if current_hour not in [9, 18]:  # 9 AM and 6 PM
                return

            # Find users who haven't studied today
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            inactive_users = db.query(UserGameProfile, User).join(User).filter(
                or_(
                    UserGameProfile.last_activity_date < today_start,
                    UserGameProfile.last_activity_date.is_(None)
                )
            ).limit(100).all()  # Limit to avoid overwhelming the service

            notifications_sent = 0

            for profile, user in inactive_users:
                # Check notification preferences
                preferences = self.notification_service.get_notification_preferences(user.id)

                if preferences.get("daily_reminders", True):
                    reminder_message = self._get_personalized_reminder(profile)

                    success = await self.notification_service.send_daily_reminder(
                        user_id=user.id,
                        reminder_message=reminder_message
                    )

                    if success:
                        notifications_sent += 1

            if notifications_sent > 0:
                logger.info(f"Sent {notifications_sent} daily reminder notifications")

        except Exception as e:
            logger.error(f"Error sending daily reminders: {e}")

    def _get_personalized_reminder(self, profile: UserGameProfile) -> str:
        """Generate personalized reminder message based on user profile"""
        if profile.current_streak > 0:
            return f"Don't break your {profile.current_streak}-day streak! Time to study! 🔥"
        elif profile.total_xp > 1000:
            return f"You're at level {profile.current_level}! Keep up the great work! 📚"
        else:
            return "Ready to learn something new today? Let's study! 💪"

    async def send_achievement_notification(self, user_id, achievement_name: str, achievement_description: str,
                                            xp_reward: int):
        """Send achievement notification immediately"""
        try:
            success = await self.notification_service.send_achievement_notification(
                user_id=user_id,
                achievement_name=achievement_name,
                achievement_description=achievement_description,
                xp_reward=xp_reward
            )

            if success:
                logger.info(f"Sent achievement notification to user {user_id}: {achievement_name}")

            return success

        except Exception as e:
            logger.error(f"Error sending achievement notification: {e}")
            return False

    async def send_level_up_notification(self, user_id, new_level: int):
        """Send level up notification immediately"""
        try:
            success = await self.notification_service.send_level_up_notification(
                user_id=user_id,
                new_level=new_level
            )

            if success:
                logger.info(f"Sent level up notification to user {user_id}: Level {new_level}")

            return success

        except Exception as e:
            logger.error(f"Error sending level up notification: {e}")
            return False

    async def send_streak_milestone_notification(self, user_id, streak_days: int, xp_bonus: int):
        """Send streak milestone notification immediately"""
        try:
            success = await self.notification_service.send_streak_milestone_notification(
                user_id=user_id,
                streak_days=streak_days,
                xp_bonus=xp_bonus
            )

            if success:
                logger.info(f"Sent streak milestone notification to user {user_id}: {streak_days} days")

            return success

        except Exception as e:
            logger.error(f"Error sending streak milestone notification: {e}")
            return False


# Global scheduler instance
notification_scheduler = NotificationScheduler()


async def start_notification_scheduler():
    """Start the global notification scheduler"""
    await notification_scheduler.start_scheduler()


def stop_notification_scheduler():
    """Stop the global notification scheduler"""
    notification_scheduler.stop_scheduler()
'''