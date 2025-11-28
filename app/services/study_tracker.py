from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.progress_service import ProgressService
from app.schemas.progress_schema import StudySessionCreate
from app.core.logging import get_logger

logger = get_logger(__name__)


class StudySessionTracker:
    """Context manager for tracking study sessions automatically"""

    def __init__(
            self,
            db: AsyncSession,
            user_id: str,
            activity_type: str,
            activity_id: Optional[str] = None,
            subject_tags: Optional[List[str]] = None
    ):
        self.db = db
        self.user_id = user_id
        self.activity_type = activity_type
        self.activity_id = activity_id
        self.subject_tags = subject_tags
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    async def __aenter__(self):
        """Start tracking the study session"""
        self.start_time = datetime.utcnow()
        logger.info(f"Started tracking {self.activity_type} session for user {self.user_id}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End tracking and save the study session"""
        self.end_time = datetime.utcnow()

        if self.start_time and self.end_time:
            duration_seconds = int((self.end_time - self.start_time).total_seconds())

            # Only track sessions longer than 10 seconds
            if duration_seconds >= 10:
                try:
                    progress_service = ProgressService(self.db)
                    session_data = StudySessionCreate(
                        activity_type=self.activity_type,
                        activity_id=self.activity_id,
                        subject_tags=self.subject_tags,
                        duration_seconds=duration_seconds,
                        started_at=self.start_time,
                        ended_at=self.end_time
                    )

                    await progress_service.create_study_session(self.user_id, session_data)
                    logger.info(f"Tracked {self.activity_type} session: {duration_seconds}s for user {self.user_id}")

                except Exception as e:
                    logger.error(f"Failed to track study session: {e}")
                    # Don't raise the exception to avoid disrupting the main flow
            else:
                logger.debug(f"Session too short ({duration_seconds}s), not tracking")


@asynccontextmanager
async def track_study_session(
        db: AsyncSession,
        user_id: str,
        activity_type: str,
        activity_id: Optional[str] = None,
        subject_tags: Optional[List[str]] = None
):
    """Async context manager for tracking study sessions"""
    tracker = StudySessionTracker(db, user_id, activity_type, activity_id, subject_tags)
    async with tracker:
        yield tracker


class StudyActivityTracker:
    """Helper class for tracking different types of study activities"""

    @staticmethod
    async def track_chat_session(
            db: AsyncSession,
            user_id: str,
            conversation_id: str,
            subject_tags: Optional[List[str]] = None
    ):
        """Track a chat/tutoring session"""
        async with track_study_session(
                db=db,
                user_id=user_id,
                activity_type="chat",
                activity_id=conversation_id,
                subject_tags=subject_tags
        ) as tracker:
            yield tracker

    @staticmethod
    async def track_flashcard_session(
            db: AsyncSession,
            user_id: str,
            subject_tags: Optional[List[str]] = None
    ):
        """Track a flashcard review session"""
        async with track_study_session(
                db=db,
                user_id=user_id,
                activity_type="flashcard",
                subject_tags=subject_tags
        ) as tracker:
            yield tracker

    @staticmethod
    async def track_quiz_session(
            db: AsyncSession,
            user_id: str,
            quiz_id: str,
            subject_tags: Optional[List[str]] = None
    ):
        """Track a quiz taking session"""
        async with track_study_session(
                db=db,
                user_id=user_id,
                activity_type="quiz",
                activity_id=quiz_id,
                subject_tags=subject_tags
        ) as tracker:
            yield tracker

    @staticmethod
    async def track_note_review_session(
            db: AsyncSession,
            user_id: str,
            note_id: str,
            subject_tags: Optional[List[str]] = None
    ):
        """Track a note review session"""
        async with track_study_session(
                db=db,
                user_id=user_id,
                activity_type="note_review",
                activity_id=note_id,
                subject_tags=subject_tags
        ) as tracker:
            yield tracker