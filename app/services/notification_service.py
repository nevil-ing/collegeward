import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from uuid import UUID
import asyncio
import json


logger = logging.getLogger(__name__)


class NotificationService:
    """Service for sending push notifications via Firebase"""

    def __init__(self):
        # In a real implementation, initialize Firebase Admin SDK here
        # import firebase_admin
        # from firebase_admin import credentials, messaging
        # cred = credentials.Certificate("path/to/serviceAccountKey.json")
        # firebase_admin.initialize_app(cred)
        self.initialized = True
        logger.info("Notification service initialized (mock mode)")

    async def send_notification(
            self,
            user_id: UUID,
            title: str,
            body: str,
            data: Optional[Dict[str, Any]] = None,
            fcm_token: Optional[str] = None
    ) -> bool:
        """Send a push notification to a user"""
        try:
            # In a real implementation, you would:
            # 1. Get user's FCM token from database if not provided
            # 2. Create Firebase message
            # 3. Send via Firebase Admin SDK

            notification_data = {
                "user_id": str(user_id),
                "title": title,
                "body": body,
                "data": data or {},
                "timestamp": datetime.now().isoformat(),
                "fcm_token": fcm_token or "mock_token"
            }

            # Mock implementation - log the notification
            logger.info(f"NOTIFICATION SENT: {json.dumps(notification_data, indent=2)}")

            # In real implementation:
            # from firebase_admin import messaging
            # message = messaging.Message(
            #     notification=messaging.Notification(title=title, body=body),
            #     data=data or {},
            #     token=fcm_token
            # )
            # response = messaging.send(message)
            # return bool(response)

            return True

        except Exception as e:
            logger.error(f"Failed to send notification to user {user_id}: {str(e)}")
            return False

    async def send_streak_reminder(
            self,
            user_id: UUID,
            current_streak: int,
            hours_remaining: int,
            fcm_token: Optional[str] = None
    ) -> bool:
        """Send streak reminder notification"""
        title = "Don't break your streak!"

        if hours_remaining <= 2:
            body = f"Only {hours_remaining} hours left to maintain your {current_streak}-day streak!"
        elif hours_remaining <= 6:
            body = f"You have {hours_remaining} hours to keep your {current_streak}-day streak alive!"
        else:
            body = f"Don't forget to study today! Keep your {current_streak}-day streak going!"

        data = {
            "type": "streak_reminder",
            "current_streak": str(current_streak),
            "hours_remaining": str(hours_remaining),
            "action": "open_app"
        }

        return await self.send_notification(
            user_id=user_id,
            title=title,
            body=body,
            data=data,
            fcm_token=fcm_token
        )

    async def send_achievement_notification(
            self,
            user_id: UUID,
            achievement_name: str,
            achievement_description: str,
            xp_reward: int,
            fcm_token: Optional[str] = None
    ) -> bool:
        """Send achievement earned notification"""
        title = f"Achievement Unlocked!"
        body = f"You earned '{achievement_name}' (+{xp_reward} XP)"

        data = {
            "type": "achievement_earned",
            "achievement_name": achievement_name,
            "achievement_description": achievement_description,
            "xp_reward": str(xp_reward),
            "action": "view_achievements"
        }

        return await self.send_notification(
            user_id=user_id,
            title=title,
            body=body,
            data=data,
            fcm_token=fcm_token
        )

    async def send_level_up_notification(
            self,
            user_id: UUID,
            new_level: int,
            fcm_token: Optional[str] = None
    ) -> bool:
        """Send level up notification"""
        title = f"Level Up!"
        body = f"Congratulations! You've reached level {new_level}!"

        data = {
            "type": "level_up",
            "new_level": str(new_level),
            "action": "view_profile"
        }

        return await self.send_notification(
            user_id=user_id,
            title=title,
            body=body,
            data=data,
            fcm_token=fcm_token
        )

    async def send_streak_milestone_notification(
            self,
            user_id: UUID,
            streak_days: int,
            xp_bonus: int,
            fcm_token: Optional[str] = None
    ) -> bool:
        """Send streak milestone notification"""
        title = f"Streak Milestone!"
        body = f"Amazing! You've maintained a {streak_days}-day streak! (+{xp_bonus} XP bonus)"

        data = {
            "type": "streak_milestone",
            "streak_days": str(streak_days),
            "xp_bonus": str(xp_bonus),
            "action": "view_streak"
        }

        return await self.send_notification(
            user_id=user_id,
            title=title,
            body=body,
            data=data,
            fcm_token=fcm_token
        )

    async def send_daily_reminder(
            self,
            user_id: UUID,
            reminder_message: str = "Time to study!",
            fcm_token: Optional[str] = None
    ) -> bool:
        """Send daily study reminder"""
        title = "StudyBlitzAI Reminder"
        body = reminder_message

        data = {
            "type": "daily_reminder",
            "action": "open_app"
        }

        return await self.send_notification(
            user_id=user_id,
            title=title,
            body=body,
            data=data,
            fcm_token=fcm_token
        )

    async def send_leaderboard_update(
            self,
            user_id: UUID,
            new_rank: int,
            leaderboard_type: str = "xp",
            fcm_token: Optional[str] = None
    ) -> bool:
        """Send leaderboard position update notification"""
        title = f"Leaderboard Update!"
        body = f"You're now #{new_rank} on the {leaderboard_type} leaderboard!"

        data = {
            "type": "leaderboard_update",
            "new_rank": str(new_rank),
            "leaderboard_type": leaderboard_type,
            "action": "view_leaderboard"
        }

        return await self.send_notification(
            user_id=user_id,
            title=title,
            body=body,
            data=data,
            fcm_token=fcm_token
        )

    async def send_bulk_notifications(
            self,
            notifications: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Send multiple notifications in bulk"""
        results = {"sent": 0, "failed": 0}

        # Process notifications in batches to avoid overwhelming the service
        batch_size = 100
        for i in range(0, len(notifications), batch_size):
            batch = notifications[i:i + batch_size]

            # Send batch concurrently
            tasks = []
            for notification in batch:
                task = self.send_notification(**notification)
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    results["failed"] += 1
                    logger.error(f"Bulk notification failed: {str(result)}")
                elif result:
                    results["sent"] += 1
                else:
                    results["failed"] += 1

            # Small delay between batches
            if i + batch_size < len(notifications):
                await asyncio.sleep(0.1)

        logger.info(f"Bulk notifications completed: {results['sent']} sent, {results['failed']} failed")
        return results

    def schedule_streak_reminders(self, db_session) -> int:
        """Schedule streak reminder notifications for users at risk"""
        # This would typically be called by a background job/cron task
        # In a real implementation, you would:
        # 1. Query users whose streaks are at risk
        # 2. Check their notification preferences
        # 3. Schedule notifications via a task queue (Celery, etc.)

        logger.info("Streak reminder scheduling called (mock implementation)")
        return 0

    def get_notification_preferences(self, user_id: UUID) -> Dict[str, bool]:
        """Get user's notification preferences"""
        # In a real implementation, fetch from database
        return {
            "streak_reminders": True,
            "achievement_notifications": True,
            "xp_milestones": True,
            "leaderboard_updates": False,
            "daily_reminders": True
        }

    def update_notification_preferences(
            self,
            user_id: UUID,
            preferences: Dict[str, bool]
    ) -> bool:
        """Update user's notification preferences"""
        # In a real implementation, save to database
        logger.info(f"Updated notification preferences for user {user_id}: {preferences}")
        return True