import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.db.session import get_db
from app.dependencies import get_current_db_user
from app.models.user import User
from app.services.gamification_service import GamificationService
from app.schemas.gamification_schema import (
    UserGameProfileResponse, UserGameProfileUpdate, AchievementResponse,
    UserAchievementResponse, XPTransactionResponse, LeaderboardResponse,
    GamificationStats, StreakStatus, GamificationSummary, ActivityXPReward
)
from app.utils.exceptions import NotFoundError, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gamification", tags=["gamification"])


@router.get("/profile", response_model=UserGameProfileResponse)
async def get_user_game_profile(
        current_user: User = Depends(get_current_db_user),
        db: AsyncSession = Depends(get_db)
):
    """Get current user's gamification profile"""
    try:
        service = GamificationService(db)
        profile = await service.get_or_create_user_profile(current_user.id)
        return profile
    except Exception as e:
        logger.error(f"Error getting game profile for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get game profile")


@router.patch("/profile", response_model=UserGameProfileResponse)
async def update_user_game_profile(
        profile_update: UserGameProfileUpdate,
        current_user: User = Depends(get_current_db_user),
        db: AsyncSession = Depends(get_db)
):
    """Update user's gamification profile settings"""
    try:
        service = GamificationService(db)
        profile = await service.get_or_create_user_profile(current_user.id)

        # Update allowed fields
        if profile_update.leaderboard_visible is not None:
            profile.leaderboard_visible = profile_update.leaderboard_visible

        if profile_update.streak_freeze_count is not None:
            profile.streak_freeze_count = profile_update.streak_freeze_count

        await db.commit()
        await db.refresh(profile)

        logger.info(f"Updated game profile for user {current_user.id}")
        return profile
    except Exception as e:
        logger.error(f"Error updating game profile for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update game profile")


@router.get("/stats", response_model=GamificationStats)
async def get_gamification_stats(
        current_user: User = Depends(get_current_db_user),
        db: AsyncSession = Depends(get_db)
):
    """Get comprehensive gamification statistics"""
    try:
        service = GamificationService(db)
        stats = await service.get_gamification_stats(current_user.id)
        return stats
    except Exception as e:
        logger.error(f"Error getting gamification stats for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get gamification stats")


@router.get("/summary", response_model=GamificationSummary)
async def get_gamification_summary(
        current_user: User = Depends(get_current_db_user),
        db: AsyncSession = Depends(get_db)
):
    """Get gamification summary for dashboard"""
    try:
        service = GamificationService(db)

        # Get all components
        profile = await service.get_or_create_user_profile(current_user.id)
        stats = await service.get_gamification_stats(current_user.id)
        streak_status = await service.get_streak_status(current_user.id)

        # Get leaderboard position
        leaderboard = await service.get_leaderboard(user_id=current_user.id, limit=10)
        leaderboard_position = leaderboard.user_rank

        return GamificationSummary(
            profile=profile,
            stats=stats,
            streak_status=streak_status,
            leaderboard_position=leaderboard_position,
            pending_achievements=[]  # Could add logic to show almost-earned achievements
        )
    except Exception as e:
        logger.error(f"Error getting gamification summary for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get gamification summary")


@router.get("/achievements", response_model=List[UserAchievementResponse])
async def get_user_achievements(
        current_user: User = Depends(get_current_db_user),
        db: AsyncSession = Depends(get_db)
):
    """Get user's earned achievements"""
    try:
        from app.db.models import UserAchievement

        result = await db.execute(
            select(UserAchievement)
            .where(UserAchievement.user_id == current_user.id)
            .order_by(desc(UserAchievement.earned_at))
        )
        achievements = result.scalars().all()

        return achievements
    except Exception as e:
        logger.error(f"Error getting achievements for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get achievements")


@router.post("/achievements/check", response_model=List[UserAchievementResponse])
async def check_achievements(
        current_user: User = Depends(get_current_db_user),
        db: AsyncSession = Depends(get_db)
):
    """Check and award any pending achievements"""
    try:
        service = GamificationService(db)
        new_achievements = await service.check_and_award_achievements(current_user.id)
        return new_achievements
    except Exception as e:
        logger.error(f"Error checking achievements for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check achievements")


@router.get("/xp/transactions", response_model=List[XPTransactionResponse])
async def get_xp_transactions(
        limit: int = Query(50, ge=1, le=100),
        current_user: User = Depends(get_current_db_user),
        db: AsyncSession = Depends(get_db)
):
    """Get user's XP transaction history"""
    try:
        from app.db.models import XPTransaction

        result = await db.execute(
            select(XPTransaction)
            .where(XPTransaction.user_id == current_user.id)
            .order_by(desc(XPTransaction.created_at))
            .limit(limit)
        )
        transactions = result.scalars().all()

        return transactions
    except Exception as e:
        logger.error(f"Error getting XP transactions for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get XP transactions")


@router.get("/streak", response_model=StreakStatus)
async def get_streak_status(
        current_user: User = Depends(get_current_db_user),
        db: AsyncSession = Depends(get_db)
):
    """Get current streak status"""
    try:
        service = GamificationService(db)
        streak_status = await service.get_streak_status(current_user.id)
        return streak_status
    except Exception as e:
        logger.error(f"Error getting streak status for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get streak status")


@router.post("/streak/freeze", response_model=dict)
async def use_streak_freeze(
        current_user: User = Depends(get_current_db_user),
        db: AsyncSession = Depends(get_db)
):
    """Use a streak freeze to maintain streak"""
    try:
        service = GamificationService(db)
        success = await service.use_streak_freeze(current_user.id)

        if not success:
            raise HTTPException(status_code=400, detail="No streak freezes available")

        return {"message": "Streak freeze used successfully", "success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error using streak freeze for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to use streak freeze")


@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
        leaderboard_type: str = Query("xp", regex="^(xp|level|streak)$"),
        limit: int = Query(50, ge=1, le=100),
        current_user: User = Depends(get_current_db_user),
        db: AsyncSession = Depends(get_db)
):
    """Get leaderboard rankings"""
    try:
        service = GamificationService(db)
        leaderboard = await service.get_leaderboard(
            user_id=current_user.id,
            limit=limit,
            leaderboard_type=leaderboard_type
        )
        return leaderboard
    except Exception as e:
        logger.error(f"Error getting leaderboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get leaderboard")


@router.get("/achievements/available", response_model=List[AchievementResponse])
async def get_available_achievements(
        category: Optional[str] = Query(None),
        db: AsyncSession = Depends(get_db)
):
    """Get all available achievements"""
    try:
        from app.db.models import Achievement

        stmt = select(Achievement).where(Achievement.is_active == True)

        if category:
            stmt = stmt.where(Achievement.category == category)

        stmt = stmt.order_by(Achievement.category, Achievement.name)
        result = await db.execute(stmt)
        achievements = result.scalars().all()
        return achievements
    except Exception as e:
        logger.error(f"Error getting available achievements: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get available achievements")


# Internal endpoint for awarding XP (used by other services)
@router.post("/internal/award-xp", response_model=ActivityXPReward)
async def award_xp_internal(
        user_id: UUID,
        activity_type: str,
        activity_id: Optional[UUID] = None,
        base_xp: Optional[int] = None,
        multiplier: float = 1.0,
        reason: Optional[str] = None,
        db: AsyncSession = Depends(get_db)
):
    """Internal endpoint for awarding XP (used by other services)"""
    try:
        service = GamificationService(db)
        reward = await service.award_xp(
            user_id=user_id,
            activity_type=activity_type,
            activity_id=activity_id,
            base_xp=base_xp,
            multiplier=multiplier,
            reason=reason
        )
        return reward
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error awarding XP: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to award XP")


# Internal endpoint for updating streaks (used by other services)
@router.post("/internal/update-streak", response_model=StreakStatus)
async def update_streak_internal(
        user_id: UUID,
        study_time_seconds: int = 0,
        db: AsyncSession = Depends(get_db)
):
    """Internal endpoint for updating study streaks (used by other services)"""
    try:
        service = GamificationService(db)
        streak_status = await service.update_study_streak(
            user_id=user_id,
            study_time_seconds=study_time_seconds
        )
        return streak_status
    except Exception as e:
        logger.error(f"Error updating streak: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update streak")