from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from uuid import UUID
from pydantic import BaseModel

from app.core.security import get_current_user
from app.db.session import get_db
from app.core.logging import get_logger
from app.services.quiz_service import QuizService
from app.services.user_service import user_service
from app.schemas.quiz_schema import (
    QuizResponse, QuizSubmission, QuizResult, QuizStats
)
from app.utils.exceptions import NotFoundError, ValidationError

logger = get_logger(__name__)
router = APIRouter()
quiz_service = QuizService()


class QuizGenerationRequest(BaseModel):
    """Request model for quiz generation"""
    note_ids: Optional[List[str]] = None  # Accept as strings, convert to UUID in endpoint
    question_count: Optional[int] = None
    num_questions: Optional[int] = None
    difficulty_level: Optional[int] = None
    subject_tags: Optional[List[str]] = None
    focus_subjects: Optional[List[str]] = None
    title: Optional[str] = None
    quiz_title: Optional[str] = None


@router.get("/stats", response_model=QuizStats)
async def get_quiz_statistics(
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        days: int = Query(30, ge=1, le=365)
):
    """
    Get user's quiz performance statistics and analytics

    Args:
        days: Number of days to analyze (1-365)
    """
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        stats = await quiz_service.get_quiz_statistics(
            db=db,
            user_id=user.id,
            days=days
        )

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quiz statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quiz statistics")


@router.get("/", response_model=List[QuizResponse])
async def get_quizzes(
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
        completed_only: bool = Query(False)
):
    """
    Get quizzes for the current user with pagination

    Args:
        limit: Maximum number of quizzes to return (1-100)
        offset: Number of quizzes to skip for pagination
        completed_only: Whether to return only completed quizzes
    """
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        quizzes = await quiz_service.get_user_quizzes(
            db=db,
            user_id=user.id,
            limit=limit,
            offset=offset,
            completed_only=completed_only
        )

        logger.info(f"Retrieved {len(quizzes)} quizzes for user {user.id}")
        return quizzes

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quizzes: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quizzes")


@router.get("/{quiz_id}", response_model=QuizResponse)
async def get_quiz(
        quiz_id: UUID,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        include_answers: bool = Query(False)
):
    """
    Get a specific quiz by ID

    Args:
        quiz_id: Quiz ID
        include_answers: Whether to include correct answers (for completed quizzes)
    """
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        quiz = await quiz_service.get_quiz(
            db=db,
            quiz_id=quiz_id,
            user_id=user.id,
            include_answers=include_answers
        )

        return quiz

    except HTTPException:
        raise
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get quiz {quiz_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quiz")


@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(
        request: QuizGenerationRequest = Body(...),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Generate a new quiz from user's study materials

    Args:
        note_ids: Specific note IDs to use (if None, uses all user notes)
        num_questions: Number of questions to generate (5-50)
        difficulty_level: Target difficulty level (1-5)
        focus_subjects: Specific medical subjects to focus on
        quiz_title: Custom quiz title
    """
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Normalize parameters (accept both names from frontend)
        final_num_questions = request.num_questions or request.question_count or 10
        # Normalize subject tags to lowercase for consistency
        subjects_list = request.focus_subjects or request.subject_tags
        final_focus_subjects = [s.lower() for s in subjects_list] if subjects_list else None
        final_quiz_title = request.quiz_title or request.title

        # Convert note_ids from strings to UUIDs if provided
        final_note_ids = None
        if request.note_ids:
            try:
                final_note_ids = [UUID(note_id) for note_id in request.note_ids]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid note ID format: {e}")

        quiz = await quiz_service.generate_quiz_from_notes(
            db=db,
            user_id=user.id,
            note_ids=final_note_ids,
            num_questions=final_num_questions,
            difficulty_level=request.difficulty_level,
            focus_subjects=final_focus_subjects,
            quiz_title=final_quiz_title
        )

        logger.info(f"Generated quiz {quiz.id} for user {user.id}")
        return quiz

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Quiz generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate quiz")


@router.post("/generate/targeted", response_model=QuizResponse)
async def generate_targeted_quiz(
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        weak_areas: List[str] = Query(..., min_items=1),
        num_questions: int = Query(15, ge=5, le=50)
):
    """
    Generate a targeted quiz focusing on user's weak areas

    Args:
        weak_areas: Subject areas where user needs improvement
        num_questions: Number of questions to generate (5-50)
    """
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        quiz = await quiz_service.generate_targeted_quiz(
            db=db,
            user_id=user.id,
            weak_areas=weak_areas,
            num_questions=num_questions
        )

        logger.info(f"Generated targeted quiz {quiz.id} for weak areas: {weak_areas}")
        return quiz

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Targeted quiz generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate targeted quiz")


@router.patch("/{quiz_id}/save-answer", status_code=200)
async def save_quiz_answer(
        quiz_id: UUID,
        question_order: int = Body(..., ge=1),
        answer: int = Body(..., ge=0),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Save a partial answer for a quiz question (allows resuming quizzes)

    Args:
        quiz_id: Quiz ID
        question_order: Question order number (1-based)
        answer: Selected answer index (0-based)
    """
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Save the answer
        await quiz_service.save_quiz_answer(
            db=db,
            quiz_id=quiz_id,
            user_id=user.id,
            question_order=question_order,
            answer=answer
        )

        return {"message": "Answer saved successfully"}

    except HTTPException:
        raise
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to save quiz answer: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save answer")


@router.post("/{quiz_id}/submit", response_model=QuizResult)
async def submit_quiz(
        quiz_id: UUID,
        submission: QuizSubmission,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Submit quiz answers and get results with detailed explanations

    Args:
        quiz_id: Quiz ID
        submission: Quiz submission with answers and time taken
    """
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Validate submission quiz_id matches URL parameter
        if submission.quiz_id != quiz_id:
            raise HTTPException(status_code=400, detail="Quiz ID mismatch")

        result = await quiz_service.submit_quiz(
            db=db,
            quiz_id=quiz_id,
            user_id=user.id,
            submission=submission
        )

        # Award XP and update gamification for quiz completion
        try:
            from app.services.gamification_service import GamificationService
            gamification_service = GamificationService(db)

            # Calculate XP based on performance
            base_xp = 25  # Base XP for completing quiz
            performance_bonus = 0

            if result.percentage >= 100:
                # Perfect score bonus
                performance_bonus = 25
                await gamification_service.award_xp(
                    user_id=user.id,
                    activity_type="perfect_quiz",
                    activity_id=quiz_id,
                    base_xp=performance_bonus,
                    reason=f"Perfect score on quiz ({result.percentage}%)"
                )
            elif result.percentage >= 90:
                performance_bonus = 15
            elif result.percentage >= 80:
                performance_bonus = 10
            elif result.percentage >= 70:
                performance_bonus = 5

            # Award base XP
            await gamification_service.award_xp(
                user_id=user.id,
                activity_type="quiz_completed",
                activity_id=quiz_id,
                base_xp=base_xp + performance_bonus,
                reason=f"Completed quiz with {result.percentage}% score"
            )

            # Update study streak (estimate time based on questions)
            estimated_time = len(result.question_results) * 30  # 30 seconds per question
            await gamification_service.update_study_streak(
                user_id=user.id,
                study_time_seconds=estimated_time
            )

            # Check for new achievements
            await gamification_service.check_and_award_achievements(user.id)

        except Exception as e:
            logger.warning(f"Failed to update gamification for quiz {quiz_id}: {e}")

        logger.info(f"Quiz {quiz_id} submitted with score {result.score}/{result.total_questions}")
        return result

    except HTTPException:
        raise
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Quiz submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit quiz")


@router.get("/analysis/weak-areas", response_model=List[str])
async def get_weak_areas(
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        min_questions: int = Query(5, ge=3, le=20)
):
    """
    Identify subject areas where user needs improvement based on quiz performance

    Args:
        min_questions: Minimum questions needed to consider a subject area
    """
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        weak_areas = await quiz_service.identify_weak_areas(
            db=db,
            user_id=user.id,
            min_questions=min_questions
        )

        logger.info(f"Identified weak areas for user {user.id}: {weak_areas}")
        return weak_areas

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to identify weak areas: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze weak areas")