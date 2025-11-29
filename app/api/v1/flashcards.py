from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from uuid import UUID
from pydantic import BaseModel

from app.core.security import get_current_user
from app.db.session import get_db
from app.core.logging import get_logger
from app.utils.exceptions import ValidationError, NotFoundError
from app.services.flashcard_services import
from app.services.ai_service import ai_service_manager
from app.schemas.flashcard_schema import (
    FlashcardCreate, FlashcardResponse, FlashcardReview, FlashcardStats, FlashcardUpdate
)

logger = get_logger(__name__)
router = APIRouter()

# Initialize flashcard service
flashcard_service = FlashcardService(ai_service_manager)


async def get_user_from_token(
        current_user: Dict[str, Any],
        db: AsyncSession
):
    """Helper function to get user from Firebase UID token"""
    firebase_uid = current_user.get("uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    # Get user from database by Firebase UID
    from app.services.user_service import user_service
    user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@router.get("/", response_model=List[FlashcardResponse])
async def get_flashcards(
        subject_tags: Optional[List[str]] = Query(None, description="Filter by subject tags"),
        difficulty_level: Optional[int] = Query(None, ge=1, le=5, description="Filter by difficulty level"),
        note_id: Optional[UUID] = Query(None, description="Filter by source note ID"),
        limit: int = Query(100, ge=1, le=500, description="Maximum number of flashcards to return"),
        offset: int = Query(0, ge=0, description="Number of flashcards to skip"),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get flashcards for the current user with optional filtering"""
    try:
        user = await get_user_from_token(current_user, db)

        flashcards = await flashcard_service.get_user_flashcards(
            db=db,
            user_id=user.id,
            subject_tags=subject_tags,
            difficulty_level=difficulty_level,
            note_id=note_id,
            limit=limit,
            offset=offset
        )

        return flashcards

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get flashcards: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/review", response_model=List[FlashcardResponse])
async def get_review_flashcards(
        limit: int = Query(20, ge=1, le=100, description="Maximum number of review cards to return"),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get flashcards due for review"""
    try:
        user = await get_user_from_token(current_user, db)

        review_cards = await flashcard_service.get_review_flashcards(
            db=db,
            user_id=user.id,
            limit=limit
        )

        return review_cards

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get review flashcards: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats", response_model=FlashcardStats)
async def get_flashcard_stats(
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get user's flashcard statistics"""
    try:
        user = await get_user_from_token(current_user, db)

        stats = await flashcard_service.get_flashcard_stats(
            db=db,
            user_id=user.id
        )

        return stats

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get flashcard stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/by-note/{note_id}", response_model=List[FlashcardResponse])
async def get_flashcards_by_note(
        note_id: UUID,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get all flashcards generated from a specific note"""
    try:
        user = await get_user_from_token(current_user, db)

        flashcards = await flashcard_service.get_flashcards_by_note(
            db=db,
            user_id=user.id,
            note_id=note_id
        )

        return flashcards

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get flashcards by note: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/grouped-by-note", response_model=Dict[str, List[FlashcardResponse]])
async def get_flashcards_grouped_by_note(
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get flashcards grouped by their source note"""
    try:
        user = await get_user_from_token(current_user, db)

        grouped = await flashcard_service.get_flashcards_grouped_by_note(
            db=db,
            user_id=user.id
        )

        # Convert UUID keys to strings for JSON serialization
        return {str(note_id): flashcards for note_id, flashcards in grouped.items()}

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get flashcards grouped by note: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/", response_model=FlashcardResponse)
async def create_flashcard(
        flashcard_data: FlashcardCreate,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Create a new flashcard manually"""
    try:
        user = await get_user_from_token(current_user, db)

        flashcard = await flashcard_service.create_flashcard(
            db=db,
            user_id=user.id,
            flashcard_data=flashcard_data
        )

        return flashcard

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create flashcard: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


class FlashcardGenerationRequest(BaseModel):
    """Request model for flashcard generation"""
    note_ids: List[str]
    max_cards: int = 10


@router.post("/generate/notes", response_model=List[FlashcardResponse])
async def generate_flashcards_from_notes(
        request: FlashcardGenerationRequest = Body(...),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Generate flashcards from multiple notes"""
    try:
        user = await get_user_from_token(current_user, db)

        # Convert note IDs from strings to UUIDs
        try:
            note_uuids = [UUID(note_id) for note_id in request.note_ids]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid note ID format: {e}")

        flashcards = await flashcard_service.generate_flashcards_from_notes(
            db=db,
            user_id=user.id,
            note_ids=note_uuids,
            max_cards=request.max_cards
        )

        return flashcards

    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate flashcards from notes: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/generate/note/{note_id}", response_model=List[FlashcardResponse])
async def generate_flashcards_from_note(
        note_id: UUID,
        max_cards: int = Query(10, ge=5, le=50),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Generate flashcards from uploaded note"""
    try:
        user = await get_user_from_token(current_user, db)

        flashcards = await flashcard_service.generate_flashcards_from_note(
            db=db,
            user_id=user.id,
            note_id=note_id,
            max_cards=max_cards
        )

        return flashcards

    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate flashcards from note: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/generate/conversation/{conversation_id}", response_model=List[FlashcardResponse])
async def generate_flashcards_from_conversation(
        conversation_id: UUID,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Generate flashcards from conversation"""
    try:
        user = await get_user_from_token(current_user, db)

        flashcards = await flashcard_service.generate_flashcards_from_conversation(
            db=db,
            user_id=user.id,
            conversation_id=conversation_id
        )

        return flashcards

    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate flashcards from conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{flashcard_id}/review", response_model=FlashcardResponse)
async def review_flashcard(
        flashcard_id: UUID,
        review_data: FlashcardReview,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Submit flashcard review result and update spaced repetition schedule"""
    try:
        user = await get_user_from_token(current_user, db)

        # Validate that the flashcard_id matches the review data
        if review_data.flashcard_id != flashcard_id:
            raise HTTPException(status_code=400, detail="Flashcard ID mismatch")

        flashcard = await flashcard_service.review_flashcard(
            db=db,
            user_id=user.id,
            flashcard_id=flashcard_id,
            is_correct=review_data.is_correct
        )

        # Award XP and update gamification for flashcard review
        try:
            from app.services.gamification_service import GamificationService
            gamification_service = GamificationService(db)

            # Award XP for flashcard review
            base_xp = 5
            if review_data.is_correct:
                # Bonus XP for correct answers
                if flashcard.leitner_box >= 4:  # Advanced level
                    base_xp = 8
                elif flashcard.leitner_box >= 2:  # Intermediate level
                    base_xp = 6

            await gamification_service.award_xp(
                user_id=user.id,
                activity_type="flashcard_review",
                activity_id=flashcard_id,
                base_xp=base_xp,
                reason=f"Reviewed flashcard ({'correct' if review_data.is_correct else 'incorrect'})"
            )

            # Update study streak
            await gamification_service.update_study_streak(
                user_id=user.id,
                study_time_seconds=30  # Estimate 30 seconds per flashcard
            )

            # Check for new achievements
            await gamification_service.check_and_award_achievements(user.id)

        except Exception as e:
            logger.warning(f"Failed to update gamification for flashcard {flashcard_id}: {e}")

        return flashcard

    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to review flashcard: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{flashcard_id}", response_model=FlashcardResponse)
async def update_flashcard(
        flashcard_id: UUID,
        update_data: FlashcardUpdate,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Update an existing flashcard"""
    try:
        user = await get_user_from_token(current_user, db)

        # Get existing flashcard
        from sqlalchemy import select, and_
        from app.db.models import Flashcard

        query = select(Flashcard).where(
            and_(Flashcard.id == flashcard_id, Flashcard.user_id == user.id)
        )
        result = await db.execute(query)
        flashcard = result.scalar_one_or_none()

        if not flashcard:
            raise HTTPException(status_code=404, detail="Flashcard not found")

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(flashcard, field, value)

        await db.commit()
        await db.refresh(flashcard)

        return FlashcardResponse.model_validate(flashcard)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update flashcard: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{flashcard_id}")
async def delete_flashcard(
        flashcard_id: UUID,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Delete a flashcard"""
    try:
        user = await get_user_from_token(current_user, db)

        success = await flashcard_service.delete_flashcard(
            db=db,
            user_id=user.id,
            flashcard_id=flashcard_id
        )

        if success:
            return {"message": "Flashcard deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Flashcard not found")

    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete flashcard: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")