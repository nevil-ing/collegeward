import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta, date
from decimal import Decimal
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_, select

from app.models.user import User
from app.models.user_game_profile import UserGameProfile
from app.models.achievement import Achievement
from app.models.user_achievement import UserAchievement
from app.models.xp_transaction import XPTransaction
from app.models.study_streak import  StudyStreak
from app.models.study_session import StudySession

from app.schemas.gamification_schema import (
    UserGameProfileCreate, UserGameProfileUpdate, AchievementCreate,
    XPTransactionCreate, StudyStreakCreate, LeaderboardResponse,
    GamificationStats, ActivityXPReward, StreakStatus, GamificationSummary
)
from app.utils.exceptions import NotFoundError, ValidationError

logger = logging.getLogger(__name__)


class GamificationService:
    """Service for managing gamification features"""

    # XP rewards for different activities
    XP_REWARDS = {
        "chat_session": 10,
        "quiz_completed": 25,
        "flashcard_review": 5,
        "daily_login": 5,
        "note_upload": 15,
        "streak_milestone": 50,
        "achievement_earned": 100,
        "perfect_quiz": 50,
        "study_session": 1  # per minute
    }

    # Level thresholds (XP required for each level)
    LEVEL_THRESHOLDS = [
        0, 100, 250, 500, 1000, 1750, 2750, 4000, 5500, 7250, 9250,
        11500, 14000, 16750, 19750, 23000, 26500, 30250, 34250, 38500,
        43000, 47750, 52750, 58000, 63500, 69250, 75250, 81500, 88000,
        94750, 101750, 109000, 116500, 124250, 132250, 140500, 149000
    ]

    def __init__(self, db: Session | AsyncSession):
        self.db = db
        self.is_async = isinstance(db, AsyncSession)

    async def get_or_create_user_profile(self, user_id: UUID) -> UserGameProfile:
        """Get or create user game profile"""
        if self.is_async:
            # Use async operations
            result = await self.db.execute(
                select(UserGameProfile).where(UserGameProfile.user_id == user_id)
            )
            profile = result.scalar_one_or_none()

            if not profile:
                profile = UserGameProfile(
                    user_id=user_id,
                    total_xp=0,
                    current_level=1,
                    current_streak=0,
                    longest_streak=0,
                    streak_freeze_count=3,  # Start with 3 streak freezes
                    leaderboard_visible=True
                )
                self.db.add(profile)
                await self.db.commit()
                await self.db.refresh(profile)
                logger.info(f"Created new game profile for user {user_id}")
        else:
            # Use sync operations
            profile = self.db.query(UserGameProfile).filter(
                UserGameProfile.user_id == user_id
            ).first()

            if not profile:
                profile = UserGameProfile(
                    user_id=user_id,
                    total_xp=0,
                    current_level=1,
                    current_streak=0,
                    longest_streak=0,
                    streak_freeze_count=3,  # Start with 3 streak freezes
                    leaderboard_visible=True
                )
                self.db.add(profile)
                self.db.commit()
                self.db.refresh(profile)
                logger.info(f"Created new game profile for user {user_id}")

        return profile

    async def award_xp(
            self,
            user_id: UUID,
            activity_type: str,
            activity_id: Optional[UUID] = None,
            base_xp: Optional[int] = None,
            multiplier: Decimal = Decimal("1.0"),
            reason: Optional[str] = None
    ) -> ActivityXPReward:
        """Award XP points to user for activity"""
        profile = await self.get_or_create_user_profile(user_id)

        # Calculate XP
        if base_xp is None:
            base_xp = self.XP_REWARDS.get(activity_type, 0)

        total_xp = int(base_xp * multiplier)

        if total_xp <= 0:
            raise ValidationError(f"Invalid XP amount: {total_xp}")

        # Create XP transaction
        transaction = XPTransaction(
            user_id=user_id,
            activity_type=activity_type,
            activity_id=activity_id,
            xp_earned=total_xp,
            multiplier=multiplier,
            reason=reason or f"Earned from {activity_type}",
            transaction_metadata={"base_xp": base_xp}
        )
        self.db.add(transaction)

        # Update user profile
        old_level = profile.current_level
        profile.total_xp += total_xp
        profile.current_level = self._calculate_level(profile.total_xp)

        if self.is_async:
            await self.db.commit()
        else:
            self.db.commit()

        # Check for level up achievements
        if profile.current_level > old_level:
            await self._check_level_achievements(user_id, profile.current_level)

        logger.info(f"Awarded {total_xp} XP to user {user_id} for {activity_type}")

        return ActivityXPReward(
            activity_type=activity_type,
            base_xp=base_xp,
            multiplier=multiplier,
            total_xp=total_xp,
            bonus_reason=reason if multiplier > 1 else None
        )

    async def update_study_streak(
            self,
            user_id: UUID,
            study_date: Optional[date] = None,
            study_time_seconds: int = 0
    ) -> StreakStatus:
        """Update user's study streak"""
        if study_date is None:
            study_date = date.today()

        profile = await self.get_or_create_user_profile(user_id)

        # Check if streak already exists for this date
        if self.is_async:
            result = await self.db.execute(
                select(StudyStreak).where(
                    and_(
                        StudyStreak.user_id == user_id,
                        func.date(StudyStreak.streak_date) == study_date
                    )
                )
            )
            existing_streak = result.scalar_one_or_none()
        else:
            existing_streak = self.db.query(StudyStreak).filter(
                and_(
                    StudyStreak.user_id == user_id,
                    func.date(StudyStreak.streak_date) == study_date
                )
            ).first()

        if existing_streak:
            # Update existing streak
            existing_streak.activities_completed += 1
            existing_streak.total_study_time += study_time_seconds
        else:
            # Create new streak entry
            streak_entry = StudyStreak(
                user_id=user_id,
                streak_date=datetime.combine(study_date, datetime.min.time()),
                activities_completed=1,
                total_study_time=study_time_seconds,
                streak_maintained=True,
                freeze_used=False
            )
            self.db.add(streak_entry)

            # Update streak count
            yesterday = study_date - timedelta(days=1)
            if self.is_async:
                result = await self.db.execute(
                    select(StudyStreak).where(
                        and_(
                            StudyStreak.user_id == user_id,
                            func.date(StudyStreak.streak_date) == yesterday
                        )
                    )
                )
                yesterday_streak = result.scalar_one_or_none()
            else:
                yesterday_streak = self.db.query(StudyStreak).filter(
                    and_(
                        StudyStreak.user_id == user_id,
                        func.date(StudyStreak.streak_date) == yesterday
                    )
                ).first()

            if yesterday_streak or profile.current_streak == 0:
                profile.current_streak += 1
            else:
                # Streak broken, reset to 1
                profile.current_streak = 1

            # Update longest streak
            if profile.current_streak > profile.longest_streak:
                profile.longest_streak = profile.current_streak

        profile.last_activity_date = datetime.now()
        if self.is_async:
            await self.db.commit()
        else:
            self.db.commit()

        # Check for streak achievements
        await self._check_streak_achievements(user_id, profile.current_streak)

        # Award streak XP for milestones
        if profile.current_streak > 0 and profile.current_streak % 7 == 0:
            await self.award_xp(
                user_id=user_id,
                activity_type="streak_milestone",
                reason=f"{profile.current_streak} day streak milestone"
            )

        return await self.get_streak_status(user_id)

    async def get_streak_status(self, user_id: UUID) -> StreakStatus:
        """Get current streak status for user"""
        from datetime import timezone
        
        profile = await self.get_or_create_user_profile(user_id)

        # Check if streak is at risk
        streak_at_risk = False
        hours_until_break = None

        if profile.last_activity_date:
            # Use timezone-aware datetime for comparison
            now = datetime.now(timezone.utc)
            # Ensure last_activity_date is timezone-aware
            last_activity = profile.last_activity_date
            if last_activity.tzinfo is None:
                # If naive, assume UTC
                from datetime import timezone as tz
                last_activity = last_activity.replace(tzinfo=tz.utc)
            
            hours_since_activity = (now - last_activity).total_seconds() / 3600
            if hours_since_activity > 20:  # 20 hours without activity
                streak_at_risk = True
                hours_until_break = max(0, int(24 - hours_since_activity))

        return StreakStatus(
            current_streak=profile.current_streak,
            longest_streak=profile.longest_streak,
            last_activity_date=profile.last_activity_date,
            streak_at_risk=streak_at_risk,
            hours_until_break=hours_until_break,
            freeze_available=profile.streak_freeze_count > 0,
            freeze_count=profile.streak_freeze_count
        )

    async def use_streak_freeze(self, user_id: UUID) -> bool:
        """Use a streak freeze to maintain streak"""
        profile = await self.get_or_create_user_profile(user_id)

        if profile.streak_freeze_count <= 0:
            return False

        # Create streak entry for yesterday with freeze used
        yesterday = date.today() - timedelta(days=1)
        freeze_streak = StudyStreak(
            user_id=user_id,
            streak_date=datetime.combine(yesterday, datetime.min.time()),
            activities_completed=0,
            total_study_time=0,
            streak_maintained=True,
            freeze_used=True
        )
        self.db.add(freeze_streak)

        profile.streak_freeze_count -= 1
        if self.is_async:
            await self.db.commit()
        else:
            self.db.commit()

        logger.info(f"User {user_id} used streak freeze. Remaining: {profile.streak_freeze_count}")
        return True

    async def check_and_award_achievements(self, user_id: UUID) -> List[UserAchievement]:
        """Check and award any pending achievements"""
        new_achievements = []

        # Get all active achievements
        if self.is_async:
            result = await self.db.execute(
                select(Achievement).where(Achievement.is_active == True)
            )
            achievements = result.scalars().all()

            result = await self.db.execute(
                select(UserAchievement).where(UserAchievement.user_id == user_id)
            )
            user_achievements = result.scalars().all()
        else:
            achievements = self.db.query(Achievement).filter(
                Achievement.is_active == True
            ).all()

            user_achievements = self.db.query(UserAchievement).filter(
                UserAchievement.user_id == user_id
            ).all()
        earned_achievement_ids = {ua.achievement_id for ua in user_achievements}

        for achievement in achievements:
            if achievement.id not in earned_achievement_ids:
                if await self._check_achievement_criteria(user_id, achievement):
                    # Award achievement
                    user_achievement = UserAchievement(
                        user_id=user_id,
                        achievement_id=achievement.id,
                        earned_at=datetime.now()
                    )
                    self.db.add(user_achievement)
                    new_achievements.append(user_achievement)

                    # Award XP for achievement
                    if achievement.xp_reward > 0:
                        await self.award_xp(
                            user_id=user_id,
                            activity_type="achievement_earned",
                            base_xp=achievement.xp_reward,
                            reason=f"Achievement: {achievement.name}"
                        )

        if new_achievements:
            if self.is_async:
                await self.db.commit()
            else:
                self.db.commit()
            logger.info(f"Awarded {len(new_achievements)} new achievements to user {user_id}")

        return new_achievements

    async def get_leaderboard(
            self,
            user_id: Optional[UUID] = None,
            limit: int = 50,
            leaderboard_type: str = "xp"
    ) -> LeaderboardResponse:
        """Get leaderboard rankings"""
        if self.is_async:
            # Use async operations
            # Base query for users with visible profiles
            base_stmt = (
                select(
                    UserGameProfile.user_id,
                    User.display_name,
                    UserGameProfile.total_xp,
                    UserGameProfile.current_level,
                    UserGameProfile.current_streak
                )
                .join(User)
                .where(UserGameProfile.leaderboard_visible == True)
            )

            # Order by leaderboard type
            if leaderboard_type == "xp":
                base_stmt = base_stmt.order_by(desc(UserGameProfile.total_xp))
            elif leaderboard_type == "level":
                base_stmt = base_stmt.order_by(desc(UserGameProfile.current_level), desc(UserGameProfile.total_xp))
            elif leaderboard_type == "streak":
                base_stmt = base_stmt.order_by(desc(UserGameProfile.current_streak))

            # Get top entries
            stmt = base_stmt.limit(limit)
            result = await self.db.execute(stmt)
            top_entries = result.all()

            # Calculate ranks and create entries
            entries = []
            user_rank = None

            for rank, entry in enumerate(top_entries, 1):
                if user_id and entry.user_id == user_id:
                    user_rank = rank

                entries.append({
                    "user_id": entry.user_id,
                    "display_name": entry.display_name,
                    "total_xp": entry.total_xp,
                    "current_level": entry.current_level,
                    "current_streak": entry.current_streak,
                    "rank": rank
                })

            # If user not in top entries, find their rank
            if user_id and user_rank is None:
                user_profile = await self.get_or_create_user_profile(user_id)
                if user_profile.leaderboard_visible:
                    if leaderboard_type == "xp":
                        rank_stmt = select(func.count(UserGameProfile.id)).where(
                            and_(
                                UserGameProfile.total_xp > user_profile.total_xp,
                                UserGameProfile.leaderboard_visible == True
                            )
                        )
                    elif leaderboard_type == "level":
                        rank_stmt = select(func.count(UserGameProfile.id)).where(
                            and_(
                                or_(
                                    UserGameProfile.current_level > user_profile.current_level,
                                    and_(
                                        UserGameProfile.current_level == user_profile.current_level,
                                        UserGameProfile.total_xp > user_profile.total_xp
                                    )
                                ),
                                UserGameProfile.leaderboard_visible == True
                            )
                        )
                    elif leaderboard_type == "streak":
                        rank_stmt = select(func.count(UserGameProfile.id)).where(
                            and_(
                                UserGameProfile.current_streak > user_profile.current_streak,
                                UserGameProfile.leaderboard_visible == True
                            )
                        )

                    rank_result = await self.db.execute(rank_stmt)
                    user_rank = (rank_result.scalar() or 0) + 1

            # Get total participants
            total_stmt = select(func.count(UserGameProfile.id)).where(
                UserGameProfile.leaderboard_visible == True
            )
            total_result = await self.db.execute(total_stmt)
            total_participants = total_result.scalar() or 0
        else:
            # Use sync operations
            query = self.db.query(
                UserGameProfile.user_id,
                User.display_name,
                UserGameProfile.total_xp,
                UserGameProfile.current_level,
                UserGameProfile.current_streak
            ).join(User).filter(
                UserGameProfile.leaderboard_visible == True
            )

            # Order by leaderboard type
            if leaderboard_type == "xp":
                query = query.order_by(desc(UserGameProfile.total_xp))
            elif leaderboard_type == "level":
                query = query.order_by(desc(UserGameProfile.current_level), desc(UserGameProfile.total_xp))
            elif leaderboard_type == "streak":
                query = query.order_by(desc(UserGameProfile.current_streak))

            # Get top entries
            top_entries = query.limit(limit).all()

            # Calculate ranks and create entries
            entries = []
            user_rank = None

            for rank, entry in enumerate(top_entries, 1):
                if user_id and entry.user_id == user_id:
                    user_rank = rank

                entries.append({
                    "user_id": entry.user_id,
                    "display_name": entry.display_name,
                    "total_xp": entry.total_xp,
                    "current_level": entry.current_level,
                    "current_streak": entry.current_streak,
                    "rank": rank
                })

            # If user not in top entries, find their rank
            if user_id and user_rank is None:
                user_profile = await self.get_or_create_user_profile(user_id)
                if user_profile.leaderboard_visible:
                    if leaderboard_type == "xp":
                        rank_query = self.db.query(func.count(UserGameProfile.id)).filter(
                            and_(
                                UserGameProfile.total_xp > user_profile.total_xp,
                                UserGameProfile.leaderboard_visible == True
                            )
                        )
                    elif leaderboard_type == "level":
                        rank_query = self.db.query(func.count(UserGameProfile.id)).filter(
                            and_(
                                or_(
                                    UserGameProfile.current_level > user_profile.current_level,
                                    and_(
                                        UserGameProfile.current_level == user_profile.current_level,
                                        UserGameProfile.total_xp > user_profile.total_xp
                                    )
                                ),
                                UserGameProfile.leaderboard_visible == True
                            )
                        )
                    elif leaderboard_type == "streak":
                        rank_query = self.db.query(func.count(UserGameProfile.id)).filter(
                            and_(
                                UserGameProfile.current_streak > user_profile.current_streak,
                                UserGameProfile.leaderboard_visible == True
                            )
                        )

                    user_rank = rank_query.scalar() + 1

            total_participants = self.db.query(func.count(UserGameProfile.id)).filter(
                UserGameProfile.leaderboard_visible == True
            ).scalar()

        return LeaderboardResponse(
            entries=entries,
            user_rank=user_rank,
            total_participants=total_participants
        )

    async def get_gamification_stats(self, user_id: UUID) -> GamificationStats:
        """Get comprehensive gamification statistics for user"""
        profile = await self.get_or_create_user_profile(user_id)

        if self.is_async:
            # Use async operations
            # Get recent achievements (last 10)
            result = await self.db.execute(
                select(UserAchievement)
                .where(UserAchievement.user_id == user_id)
                .order_by(desc(UserAchievement.earned_at))
                .limit(10)
            )
            recent_achievements = result.scalars().all()

            # Get recent XP transactions (last 20)
            result = await self.db.execute(
                select(XPTransaction)
                .where(XPTransaction.user_id == user_id)
                .order_by(desc(XPTransaction.created_at))
                .limit(20)
            )
            recent_xp = result.scalars().all()

            # Count total achievements
            result = await self.db.execute(
                select(func.count(Achievement.id))
                .where(Achievement.is_active == True)
            )
            total_achievements = result.scalar() or 0

            # Count achievements earned
            result = await self.db.execute(
                select(func.count(UserAchievement.id))
                .where(UserAchievement.user_id == user_id)
            )
            achievements_earned = result.scalar() or 0
        else:
            # Use sync operations
            recent_achievements = self.db.query(UserAchievement).filter(
                UserAchievement.user_id == user_id
            ).order_by(desc(UserAchievement.earned_at)).limit(10).all()

            recent_xp = self.db.query(XPTransaction).filter(
                XPTransaction.user_id == user_id
            ).order_by(desc(XPTransaction.created_at)).limit(20).all()

            total_achievements = self.db.query(func.count(Achievement.id)).filter(
                Achievement.is_active == True
            ).scalar()

            achievements_earned = self.db.query(func.count(UserAchievement.id)).filter(
                UserAchievement.user_id == user_id
            ).scalar()

        # Calculate XP to next level
        current_level_xp = self.LEVEL_THRESHOLDS[profile.current_level - 1] if profile.current_level > 1 else 0
        next_level_xp = self.LEVEL_THRESHOLDS[profile.current_level] if profile.current_level < len(
            self.LEVEL_THRESHOLDS) else profile.total_xp
        xp_to_next_level = max(0, next_level_xp - profile.total_xp)

        return GamificationStats(
            total_xp=profile.total_xp,
            current_level=profile.current_level,
            xp_to_next_level=xp_to_next_level,
            current_streak=profile.current_streak,
            longest_streak=profile.longest_streak,
            streak_freeze_count=profile.streak_freeze_count,
            achievements_earned=achievements_earned,
            total_achievements=total_achievements,
            recent_achievements=recent_achievements,
            recent_xp_transactions=recent_xp
        )

    def _calculate_level(self, total_xp: int) -> int:
        """Calculate user level based on total XP"""
        for level, threshold in enumerate(self.LEVEL_THRESHOLDS, 1):
            if total_xp < threshold:
                return level - 1
        return len(self.LEVEL_THRESHOLDS)

    async def _check_achievement_criteria(self, user_id: UUID, achievement: Achievement) -> bool:
        """Check if user meets achievement criteria"""
        criteria = achievement.criteria

        if achievement.category == "study_time":
            # Check total study time
            if self.is_async:
                result = await self.db.execute(
                    select(func.sum(StudySession.duration_seconds)).where(
                        StudySession.user_id == user_id
                    )
                )
                total_time = result.scalar() or 0
            else:
                total_time = self.db.query(func.sum(StudySession.duration_seconds)).filter(
                    StudySession.user_id == user_id
                ).scalar() or 0
            return total_time >= criteria.get("minutes", 0) * 60

        elif achievement.category == "streak":
            # Check streak milestones
            profile = await self.get_or_create_user_profile(user_id)
            return profile.current_streak >= criteria.get("days", 0)

        elif achievement.category == "xp":
            # Check XP milestones
            profile = await self.get_or_create_user_profile(user_id)
            return profile.total_xp >= criteria.get("points", 0)

        elif achievement.category == "quiz":
            # Check quiz performance
            from app.models.quiz import Quiz
            if self.is_async:
                result = await self.db.execute(
                    select(func.count(Quiz.id)).where(
                        and_(
                            Quiz.user_id == user_id,
                            Quiz.percentage >= criteria.get("min_score", 0)
                        )
                    )
                )
                quiz_count = result.scalar() or 0
            else:
                quiz_count = self.db.query(func.count(Quiz.id)).filter(
                    and_(
                        Quiz.user_id == user_id,
                        Quiz.percentage >= criteria.get("min_score", 0)
                    )
                ).scalar() or 0
            return quiz_count >= criteria.get("count", 0)

        elif achievement.category == "flashcard":
            # Check flashcard mastery
            from app.models.flashcard import Flashcard
            if self.is_async:
                result = await self.db.execute(
                    select(func.count(Flashcard.id)).where(
                        and_(
                            Flashcard.user_id == user_id,
                            Flashcard.leitner_box >= criteria.get("min_box", 5)
                        )
                    )
                )
                mastered_count = result.scalar() or 0
            else:
                mastered_count = self.db.query(func.count(Flashcard.id)).filter(
                    and_(
                        Flashcard.user_id == user_id,
                        Flashcard.leitner_box >= criteria.get("min_box", 5)
                    )
                ).scalar() or 0
            return mastered_count >= criteria.get("count", 0)

        return False

    async def _check_level_achievements(self, user_id: UUID, level: int):
        """Check for level-based achievements"""
        if self.is_async:
            result = await self.db.execute(
                select(Achievement).where(
                    and_(
                        Achievement.category == "level",
                        Achievement.is_active == True
                    )
                )
            )
            level_achievements = result.scalars().all()
        else:
            level_achievements = self.db.query(Achievement).filter(
                and_(
                    Achievement.category == "level",
                    Achievement.is_active == True
                )
            ).all()

        for achievement in level_achievements:
            if level >= achievement.criteria.get("level", 0):
                # Check if user already has this achievement
                if self.is_async:
                    result = await self.db.execute(
                        select(UserAchievement).where(
                            and_(
                                UserAchievement.user_id == user_id,
                                UserAchievement.achievement_id == achievement.id
                            )
                        )
                    )
                    existing = result.scalar_one_or_none()
                else:
                    existing = self.db.query(UserAchievement).filter(
                        and_(
                            UserAchievement.user_id == user_id,
                            UserAchievement.achievement_id == achievement.id
                        )
                    ).first()

                if not existing:
                    user_achievement = UserAchievement(
                        user_id=user_id,
                        achievement_id=achievement.id,
                        earned_at=datetime.now()
                    )
                    self.db.add(user_achievement)
                    if self.is_async:
                        await self.db.commit()
                    else:
                        self.db.commit()

    async def _check_streak_achievements(self, user_id: UUID, streak: int):
        """Check for streak-based achievements"""
        if self.is_async:
            result = await self.db.execute(
                select(Achievement).where(
                    and_(
                        Achievement.category == "streak",
                        Achievement.is_active == True
                    )
                )
            )
            streak_achievements = result.scalars().all()
        else:
            streak_achievements = self.db.query(Achievement).filter(
                and_(
                    Achievement.category == "streak",
                    Achievement.is_active == True
                )
            ).all()

        for achievement in streak_achievements:
            if streak >= achievement.criteria.get("days", 0):
                # Check if user already has this achievement
                if self.is_async:
                    result = await self.db.execute(
                        select(UserAchievement).where(
                            and_(
                                UserAchievement.user_id == user_id,
                                UserAchievement.achievement_id == achievement.id
                            )
                        )
                    )
                    existing = result.scalar_one_or_none()
                else:
                    existing = self.db.query(UserAchievement).filter(
                        and_(
                            UserAchievement.user_id == user_id,
                            UserAchievement.achievement_id == achievement.id
                        )
                    ).first()

                if not existing:
                    user_achievement = UserAchievement(
                        user_id=user_id,
                        achievement_id=achievement.id,
                        earned_at=datetime.now()
                    )
                    self.db.add(user_achievement)
                    if self.is_async:
                        await self.db.commit()
                    else:
                        self.db.commit()