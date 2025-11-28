from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional

from app.core.security import get_current_user
from app.db.session import get_db
from app.core.logging import get_logger
from app.services.progress_service import ProgressService
from app.schemas.progress_schema import (
    StudySessionCreate, StudySessionResponse, ProgressAnalyticsResponse,
    StudyTimeAnalytics, PerformanceAnalytics, StudyRecommendationResponse,
    RecommendationRequest, ReviewReminderResponse
)

logger = get_logger(__name__)
router = APIRouter()


@router.post("/sessions", response_model=StudySessionResponse)
async def create_study_session(
        session_data: StudySessionCreate,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Create a new study session record"""
    try:
        service = ProgressService(db)
        user_id = current_user.get("uid")

        session = await service.create_study_session(user_id, session_data)
        return session

    except Exception as e:
        logger.error(f"Error creating study session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create study session")


@router.get("/analytics", response_model=ProgressAnalyticsResponse)
async def get_progress_analytics(
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get comprehensive progress analytics for the current user"""
    try:
        service = ProgressService(db)
        user_id = current_user.get("uid")

        analytics = await service.get_progress_analytics(user_id)
        return analytics

    except Exception as e:
        logger.error(f"Error getting progress analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get progress analytics")


@router.get("/analytics/study-time", response_model=StudyTimeAnalytics)
async def get_study_time_analytics(
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get detailed study time analytics"""
    try:
        service = ProgressService(db)
        user_id = current_user.get("uid")

        analytics = await service.get_study_time_analytics(user_id)
        return analytics

    except Exception as e:
        logger.error(f"Error getting study time analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get study time analytics")


@router.get("/analytics/performance", response_model=PerformanceAnalytics)
async def get_performance_analytics(
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get performance analytics including accuracy and trends"""
    try:
        service = ProgressService(db)
        user_id = current_user.get("uid")

        analytics = await service.get_performance_analytics(user_id)
        return analytics

    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance analytics")


@router.post("/recommendations", response_model=List[StudyRecommendationResponse])
async def generate_study_recommendations(
        request: RecommendationRequest,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Generate personalized study recommendations"""
    try:
        service = ProgressService(db)
        user_id = current_user.get("uid")

        recommendations = await service.generate_study_recommendations(
            user_id=user_id,
            max_recommendations=request.max_recommendations,
            focus_areas=request.focus_areas
        )
        return recommendations

    except Exception as e:
        logger.error(f"Error generating study recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")


@router.get("/recommendations", response_model=List[StudyRecommendationResponse])
async def get_study_recommendations(
        max_recommendations: int = Query(5, ge=1, le=20),
        focus_areas: Optional[List[str]] = Query(None),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get existing study recommendations or generate new ones"""
    try:
        service = ProgressService(db)
        user_id = current_user.get("uid")

        recommendations = await service.generate_study_recommendations(
            user_id=user_id,
            max_recommendations=max_recommendations,
            focus_areas=focus_areas
        )
        return recommendations

    except Exception as e:
        logger.error(f"Error getting study recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")


@router.get("/reminders", response_model=ReviewReminderResponse)
async def get_review_reminders(
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get review reminders for flashcards and weak areas"""
    try:
        service = ProgressService(db)
        user_id = current_user.get("uid")

        reminders = await service.get_review_reminders(user_id)
        return reminders

    except Exception as e:
        logger.error(f"Error getting review reminders: {e}")
        raise HTTPException(status_code=500, detail="Failed to get review reminders")