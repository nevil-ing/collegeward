import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update
from sqlalchemy.orm import selectinload

from app.models.flashcard import Flashcard
from app.models.user import User
from app.models.note import Note
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.flashcard_schema import (
    FlashcardCreate, FlashcardResponse, FlashcardReview, FlashcardStats
)
from app.services.ai_service import AIServiceManager, ConversationMessage
from app.ai.flashcard_generator import MedicalFlashcardGenerator
from app.utils.exceptions import ValidationError, NotFoundError
from app.core.config import settings

logger = logging.getLogger(__name__)


class LeitnerSystem:
    """
    Implements the Leitner system for spaced repetition scheduling
    """

    # Box intervals in days
    BOX_INTERVALS = {
        1: 1,  # Review daily
        2: 3,  # Review every 3 days
        3: 7,  # Review weekly
        4: 14,  # Review bi-weekly
        5: 30  # Review monthly (mastered)
    }

    @classmethod
    def calculate_next_review(cls, current_box: int, is_correct: bool) -> Tuple[int, datetime]:
        """
        Calculate next review date and box based on performance

        Args:
            current_box: Current Leitner box (1-5)
            is_correct: Whether the answer was correct

        Returns:
            Tuple of (new_box, next_review_date)
        """
        if is_correct:
            # Move to next box (max 5)
            new_box = min(current_box + 1, 5)
        else:
            # Move back to box 1
            new_box = 1

        # Calculate next review date
        interval_days = cls.BOX_INTERVALS[new_box]
        next_review = datetime.utcnow() + timedelta(days=interval_days)

        return new_box, next_review

    @classmethod
    def get_mastery_level(cls, box: int, times_reviewed: int, accuracy_rate: float) -> str:
        """
        Determine mastery level based on box and performance

        Args:
            box: Current Leitner box
            times_reviewed: Number of times reviewed
            accuracy_rate: Accuracy rate (0.0-1.0)

        Returns:
            Mastery level string
        """
        if box >= 5 and times_reviewed >= 3 and accuracy_rate >= 0.8:
            return "mastered"
        elif box >= 3 and accuracy_rate >= 0.7:
            return "proficient"
        elif box >= 2:
            return "learning"
        else:
            return "new"


class FlashcardGenerator:
    """
    Generates flashcards from various content sources using AI
    """

    def __init__(self, ai_service: AIServiceManager):
        self.ai_service = ai_service
        self.medical_generator = MedicalFlashcardGenerator(ai_service)

    async def generate_from_text(
            self,
            text: str,
            source_type: str,
            source_id: Optional[UUID] = None,
            subject_tags: Optional[List[str]] = None,
            max_cards: int = 10
    ) -> List[FlashcardCreate]:
        """
        Generate byte-sized summary flashcards from text content.
        Each flashcard represents a digestible chunk of the content, breaking
        down notes into concise, reviewable summaries.

        Args:
            text: Source text content
            source_type: Type of source ('notes', 'chat', 'manual')
            source_id: ID of source document/conversation
            subject_tags: Subject classifications
            max_cards: Maximum number of summary cards to generate

        Returns:
            List of flashcard creation objects with byte-sized summaries
        """
        try:
            # Use medical-specialized generator for better results
            flashcards = await self.medical_generator.generate_from_medical_text(
                text=text,
                source_type=source_type,
                source_id=source_id,
                existing_tags=subject_tags,
                max_cards=max_cards
            )

            logger.info(f"Generated {len(flashcards)} medical flashcards from {source_type}")
            return flashcards

        except Exception as e:
            logger.error(f"Medical flashcard generation failed: {str(e)}")
            return []

    async def generate_from_conversation(
            self,
            conversation_messages: List[Message],
            conversation_id: UUID,
            subject_tags: Optional[List[str]] = None
    ) -> List[FlashcardCreate]:
        """
        Generate flashcards from conversation content

        Args:
            conversation_messages: List of conversation messages
            conversation_id: ID of the conversation
            subject_tags: Subject classifications

        Returns:
            List of flashcard creation objects
        """
        # Extract meaningful content from conversation
        content_text = self._extract_conversation_content(conversation_messages)

        if len(content_text.strip()) < 100:  # Skip very short conversations
            return []

        return await self.generate_from_text(
            text=content_text,
            source_type="chat",
            source_id=conversation_id,
            subject_tags=subject_tags,
            max_cards=5  # Fewer cards from conversations
        )

    def _extract_conversation_content(self, messages: List[Message]) -> str:
        """Extract meaningful content from conversation messages"""
        content_parts = []

        for message in messages:
            if message.role == "assistant":
                # Extract key information from AI responses
                content = message.content

                # Remove citations and references
                content = re.sub(r'\[.*?\]', '', content)
                content = re.sub(r'Source:.*?\n', '', content)

                # Focus on educational content
                if any(keyword in content.lower() for keyword in [
                    'definition', 'mechanism', 'pathophysiology', 'treatment',
                    'diagnosis', 'symptoms', 'causes', 'function'
                ]):
                    content_parts.append(content.strip())

        return '\n\n'.join(content_parts)


class FlashcardService:
    """
    Main service for flashcard management and spaced repetition
    """

    def __init__(self, ai_service: AIServiceManager):
        self.ai_service = ai_service
        self.generator = FlashcardGenerator(ai_service)
        self.leitner = LeitnerSystem()

    async def create_flashcard(
            self,
            db: AsyncSession,
            user_id: UUID,
            flashcard_data: FlashcardCreate
    ) -> FlashcardResponse:
        """Create a new flashcard"""
        try:
            # Set initial review date (tomorrow)
            next_review = datetime.utcnow() + timedelta(days=1)

            flashcard = Flashcard(
                user_id=user_id,
                question=flashcard_data.question,
                answer=flashcard_data.answer,
                subject_tags=flashcard_data.subject_tags,
                difficulty_level=flashcard_data.difficulty_level,
                leitner_box=1,
                next_review_date=next_review,
                times_reviewed=0,
                times_correct=0,
                created_from=flashcard_data.created_from,
                source_reference=flashcard_data.source_reference
            )

            db.add(flashcard)
            await db.commit()
            await db.refresh(flashcard)

            return FlashcardResponse.model_validate(flashcard)

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create flashcard: {str(e)}")
            raise ValidationError(f"Failed to create flashcard: {str(e)}")

    async def get_user_flashcards(
            self,
            db: AsyncSession,
            user_id: UUID,
            subject_tags: Optional[List[str]] = None,
            difficulty_level: Optional[int] = None,
            note_id: Optional[UUID] = None,
            limit: int = 100,
            offset: int = 0
    ) -> List[FlashcardResponse]:
        """Get user's flashcards with optional filtering"""
        try:
            query = select(Flashcard).where(Flashcard.user_id == user_id)

            # Apply filters
            if subject_tags:
                query = query.where(Flashcard.subject_tags.overlap(subject_tags))

            if difficulty_level:
                query = query.where(Flashcard.difficulty_level == difficulty_level)

            if note_id:
                query = query.where(
                    and_(
                        Flashcard.source_reference == note_id,
                        Flashcard.created_from == "notes"
                    )
                )

            query = query.order_by(Flashcard.created_at.desc()).limit(limit).offset(offset)

            result = await db.execute(query)
            flashcards = result.scalars().all()

            return [FlashcardResponse.model_validate(fc) for fc in flashcards]

        except Exception as e:
            logger.error(f"Failed to get user flashcards: {str(e)}")
            raise ValidationError(f"Failed to retrieve flashcards: {str(e)}")

    async def get_flashcards_by_note(
            self,
            db: AsyncSession,
            user_id: UUID,
            note_id: UUID
    ) -> List[FlashcardResponse]:
        """Get all flashcards generated from a specific note"""
        try:
            query = select(Flashcard).where(
                and_(
                    Flashcard.user_id == user_id,
                    Flashcard.source_reference == note_id,
                    Flashcard.created_from == "notes"
                )
            ).order_by(Flashcard.created_at.desc())

            result = await db.execute(query)
            flashcards = result.scalars().all()

            return [FlashcardResponse.model_validate(fc) for fc in flashcards]

        except Exception as e:
            logger.error(f"Failed to get flashcards by note: {str(e)}")
            raise ValidationError(f"Failed to retrieve flashcards by note: {str(e)}")

    async def get_flashcards_grouped_by_note(
            self,
            db: AsyncSession,
            user_id: UUID
    ) -> Dict[UUID, List[FlashcardResponse]]:
        """Get flashcards grouped by their source note"""
        try:
            # Get all flashcards created from notes
            query = select(Flashcard).where(
                and_(
                    Flashcard.user_id == user_id,
                    Flashcard.created_from == "notes",
                    Flashcard.source_reference.isnot(None)
                )
            ).order_by(Flashcard.created_at.desc())

            result = await db.execute(query)
            flashcards = result.scalars().all()

            # Group by source_reference (note_id)
            grouped: Dict[UUID, List[FlashcardResponse]] = {}
            for flashcard in flashcards:
                if flashcard.source_reference:
                    note_id = flashcard.source_reference
                    if note_id not in grouped:
                        grouped[note_id] = []
                    grouped[note_id].append(FlashcardResponse.model_validate(flashcard))

            return grouped

        except Exception as e:
            logger.error(f"Failed to get flashcards grouped by note: {str(e)}")
            raise ValidationError(f"Failed to retrieve flashcards grouped by note: {str(e)}")

    async def get_review_flashcards(
            self,
            db: AsyncSession,
            user_id: UUID,
            limit: int = 20
    ) -> List[FlashcardResponse]:
        """Get flashcards due for review"""
        try:
            now = datetime.utcnow()

            query = select(Flashcard).where(
                and_(
                    Flashcard.user_id == user_id,
                    or_(
                        Flashcard.next_review_date <= now,
                        Flashcard.next_review_date.is_(None)
                    )
                )
            ).order_by(
                Flashcard.next_review_date.asc().nulls_first(),
                Flashcard.leitner_box.asc()
            ).limit(limit)

            result = await db.execute(query)
            flashcards = result.scalars().all()

            return [FlashcardResponse.model_validate(fc) for fc in flashcards]

        except Exception as e:
            logger.error(f"Failed to get review flashcards: {str(e)}")
            raise ValidationError(f"Failed to retrieve review flashcards: {str(e)}")

    async def review_flashcard(
            self,
            db: AsyncSession,
            user_id: UUID,
            flashcard_id: UUID,
            is_correct: bool
    ) -> FlashcardResponse:
        """Submit flashcard review and update spaced repetition schedule"""
        try:
            # Get flashcard
            query = select(Flashcard).where(
                and_(
                    Flashcard.id == flashcard_id,
                    Flashcard.user_id == user_id
                )
            )
            result = await db.execute(query)
            flashcard = result.scalar_one_or_none()

            if not flashcard:
                raise NotFoundError("Flashcard not found")

            # Update review statistics
            flashcard.times_reviewed += 1
            if is_correct:
                flashcard.times_correct += 1

            # Calculate new Leitner box and review date
            new_box, next_review = self.leitner.calculate_next_review(
                flashcard.leitner_box, is_correct
            )

            flashcard.leitner_box = new_box
            flashcard.next_review_date = next_review

            await db.commit()
            await db.refresh(flashcard)

            logger.info(f"Flashcard {flashcard_id} reviewed: correct={is_correct}, new_box={new_box}")

            return FlashcardResponse.model_validate(flashcard)

        except NotFoundError:
            raise
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to review flashcard: {str(e)}")
            raise ValidationError(f"Failed to review flashcard: {str(e)}")

    async def get_flashcard_stats(
            self,
            db: AsyncSession,
            user_id: UUID
    ) -> FlashcardStats:
        """Get user's flashcard statistics"""
        try:
            # Total flashcards
            total_query = select(func.count(Flashcard.id)).where(Flashcard.user_id == user_id)
            total_result = await db.execute(total_query)
            total_flashcards = total_result.scalar() or 0

            # Due for review
            now = datetime.utcnow()
            due_query = select(func.count(Flashcard.id)).where(
                and_(
                    Flashcard.user_id == user_id,
                    or_(
                        Flashcard.next_review_date <= now,
                        Flashcard.next_review_date.is_(None)
                    )
                )
            )
            due_result = await db.execute(due_query)
            due_for_review = due_result.scalar() or 0

            # Mastered (box 5 with good accuracy)
            # Use COALESCE to handle None values, defaulting to 0
            mastered_query = select(func.count(Flashcard.id)).where(
                and_(
                    Flashcard.user_id == user_id,
                    Flashcard.leitner_box == 5,
                    func.coalesce(Flashcard.times_reviewed, 0) >= 3,
                    (func.coalesce(Flashcard.times_correct, 0) * 1.0 /
                     func.coalesce(Flashcard.times_reviewed, 1)) >= 0.8
                )
            )
            mastered_result = await db.execute(mastered_query)
            mastered = mastered_result.scalar() or 0

            # Overall accuracy rate
            # Use COALESCE to handle None values
            accuracy_query = select(
                func.sum(func.coalesce(Flashcard.times_correct, 0)),
                func.sum(func.coalesce(Flashcard.times_reviewed, 0))
            ).where(
                and_(
                    Flashcard.user_id == user_id,
                    func.coalesce(Flashcard.times_reviewed, 0) > 0
                )
            )
            accuracy_result = await db.execute(accuracy_query)
            result_row = accuracy_result.first()
            correct_sum = result_row[0] if result_row and result_row[0] is not None else 0
            reviewed_sum = result_row[1] if result_row and result_row[1] is not None else 0

            accuracy_rate = (correct_sum / reviewed_sum) if reviewed_sum > 0 else 0.0

            # Average difficulty
            difficulty_query = select(func.avg(Flashcard.difficulty_level)).where(
                Flashcard.user_id == user_id
            )
            difficulty_result = await db.execute(difficulty_query)
            average_difficulty = float(difficulty_result.scalar() or 1.0)

            return FlashcardStats(
                total_flashcards=total_flashcards,
                due_for_review=due_for_review,
                mastered=mastered,
                accuracy_rate=accuracy_rate,
                average_difficulty=average_difficulty
            )

        except Exception as e:
            logger.error(f"Failed to get flashcard stats: {str(e)}")
            raise ValidationError(f"Failed to retrieve flashcard statistics: {str(e)}")

    async def generate_flashcards_from_note(
            self,
            db: AsyncSession,
            user_id: UUID,
            note_id: UUID,
            max_cards: int = 10
    ) -> List[FlashcardResponse]:
        """Generate flashcards from uploaded note"""
        try:
            # Get note
            query = select(Note).where(
                and_(Note.id == note_id, Note.user_id == user_id)
            )
            result = await db.execute(query)
            note = result.scalar_one_or_none()

            if not note:
                raise NotFoundError("Note not found")

            if not note.extracted_text:
                raise ValidationError("Note has no extracted text")

            # Generate flashcards
            flashcard_creates = await self.generator.generate_from_text(
                text=note.extracted_text,
                source_type="notes",
                source_id=note_id,
                subject_tags=note.subject_tags,
                max_cards=max_cards
            )

            # Create flashcards in database
            created_flashcards = []
            for flashcard_data in flashcard_creates:
                try:
                    flashcard = await self.create_flashcard(db, user_id, flashcard_data)
                    created_flashcards.append(flashcard)
                except Exception as e:
                    logger.warning(f"Failed to create individual flashcard: {str(e)}")
                    continue

            logger.info(f"Generated {len(created_flashcards)} flashcards from note {note_id}")
            return created_flashcards

        except NotFoundError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate flashcards from note: {str(e)}")
            raise ValidationError(f"Failed to generate flashcards from note: {str(e)}")

    async def generate_flashcards_from_notes(
            self,
            db: AsyncSession,
            user_id: UUID,
            note_ids: List[UUID],
            max_cards: int = 10
    ) -> List[FlashcardResponse]:
        """Generate flashcards from multiple notes"""
        try:
            if not note_ids:
                raise ValidationError("At least one note ID is required")

            # Get all notes
            query = select(Note).where(
                and_(Note.id.in_(note_ids), Note.user_id == user_id)
            )
            result = await db.execute(query)
            notes = result.scalars().all()

            if len(notes) != len(note_ids):
                raise NotFoundError("One or more notes not found")

            # Combine text from all notes
            combined_text = "\n\n".join([
                note.extracted_text for note in notes if note.extracted_text
            ])

            if not combined_text.strip():
                raise ValidationError("No extracted text available from selected notes")

            # Get combined subject tags
            all_tags = []
            for note in notes:
                if note.subject_tags:
                    all_tags.extend(note.subject_tags)
            unique_tags = list(set(all_tags)) if all_tags else None

            # Generate flashcards from combined text
            flashcard_creates = await self.generator.generate_from_text(
                text=combined_text,
                source_type="notes",
                source_id=note_ids[0] if note_ids else None,  # Use first note ID as primary source
                subject_tags=unique_tags,
                max_cards=max_cards
            )

            # Create flashcards in database
            created_flashcards = []
            for flashcard_data in flashcard_creates:
                try:
                    flashcard = await self.create_flashcard(db, user_id, flashcard_data)
                    created_flashcards.append(flashcard)
                except Exception as e:
                    logger.warning(f"Failed to create individual flashcard: {str(e)}")
                    continue

            logger.info(f"Generated {len(created_flashcards)} flashcards from {len(note_ids)} notes")
            return created_flashcards

        except NotFoundError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate flashcards from notes: {str(e)}")
            raise ValidationError(f"Failed to generate flashcards from notes: {str(e)}")

    async def generate_flashcards_from_conversation(
            self,
            db: AsyncSession,
            user_id: UUID,
            conversation_id: UUID
    ) -> List[FlashcardResponse]:
        """Generate flashcards from conversation"""
        try:
            # Get conversation with messages
            query = select(Conversation).options(
                selectinload(Conversation.messages)
            ).where(
                and_(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id
                )
            )
            result = await db.execute(query)
            conversation = result.scalar_one_or_none()

            if not conversation:
                raise NotFoundError("Conversation not found")

            if not conversation.messages:
                raise ValidationError("Conversation has no messages")

            # Generate flashcards
            flashcard_creates = await self.generator.generate_from_conversation(
                conversation_messages=conversation.messages,
                conversation_id=conversation_id
            )

            # Create flashcards in database
            created_flashcards = []
            for flashcard_data in flashcard_creates:
                try:
                    flashcard = await self.create_flashcard(db, user_id, flashcard_data)
                    created_flashcards.append(flashcard)
                except Exception as e:
                    logger.warning(f"Failed to create individual flashcard: {str(e)}")
                    continue

            logger.info(f"Generated {len(created_flashcards)} flashcards from conversation {conversation_id}")
            return created_flashcards

        except Exception as e:
            logger.error(f"Failed to generate flashcards from conversation: {str(e)}")
            raise ValidationError(f"Failed to generate flashcards from conversation: {str(e)}")

    async def delete_flashcard(
            self,
            db: AsyncSession,
            user_id: UUID,
            flashcard_id: UUID
    ) -> bool:
        """Delete a flashcard"""
        try:
            query = select(Flashcard).where(
                and_(
                    Flashcard.id == flashcard_id,
                    Flashcard.user_id == user_id
                )
            )
            result = await db.execute(query)
            flashcard = result.scalar_one_or_none()

            if not flashcard:
                raise NotFoundError("Flashcard not found")

            await db.delete(flashcard)
            await db.commit()

            return True

        except NotFoundError:
            raise
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to delete flashcard: {str(e)}")
            raise ValidationError(f"Failed to delete flashcard: {str(e)}")