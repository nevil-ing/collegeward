from typing import List, Dict, Optional
from datetime import datetime
from decimal import Decimal
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.models.quiz_question import Quiz
from app.models.flashcard import Flashcard
from app.models.subject_mastery import SubjectMastery
from app.models.quiz_question import QuizQuestion

from app.core.logging import get_logger

logger = get_logger(__name__)


class MasteryCalculationService:
    """Service for calculating and updating subject mastery"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def update_mastery_from_quiz(self, user_id: str, quiz_id: str):
        """Update subject mastery based on quiz performance"""
        try:
            # Convert string UUIDs to UUID objects
            user_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
            quiz_uuid = uuid.UUID(quiz_id) if isinstance(quiz_id, str) else quiz_id

            # Get quiz with questions
            quiz_query = select(Quiz).where(Quiz.id == quiz_uuid)
            result = await self.db.execute(quiz_query)
            quiz = result.scalar_one_or_none()

            if not quiz or not quiz.subject_tags:
                return

            # Get quiz questions and performance
            questions_query = select(QuizQuestion).where(QuizQuestion.quiz_id == quiz_uuid)
            result = await self.db.execute(questions_query)
            questions = result.scalars().all()

            if not questions:
                return

            # Calculate performance by subject
            subject_performance = {}
            for subject_tag in quiz.subject_tags:
                total_questions = len(questions)
                correct_answers = sum(1 for q in questions if q.is_correct)

                subject_performance[subject_tag] = {
                    'total_questions': total_questions,
                    'correct_answers': correct_answers,
                    'accuracy': correct_answers / total_questions if total_questions > 0 else 0
                }

            # Update mastery for each subject
            for subject_tag, performance in subject_performance.items():
                await self._update_subject_mastery(
                    user_id=user_uuid,
                    subject_tag=subject_tag,
                    quiz_questions=performance['total_questions'],
                    quiz_correct=performance['correct_answers']
                )

        except Exception as e:
            logger.error(f"Error updating mastery from quiz {quiz_id}: {e}")
            raise

    async def update_mastery_from_flashcard_review(
            self,
            user_id: str,
            flashcard_id: str,
            was_correct: bool
    ):
        """Update subject mastery based on flashcard review"""
        try:
            # Convert string UUIDs to UUID objects
            user_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
            flashcard_uuid = uuid.UUID(flashcard_id) if isinstance(flashcard_id, str) else flashcard_id

            # Get flashcard
            flashcard_query = select(Flashcard).where(Flashcard.id == flashcard_uuid)
            result = await self.db.execute(flashcard_query)
            flashcard = result.scalar_one_or_none()

            if not flashcard or not flashcard.subject_tags:
                return

            # Update mastery for each subject tag
            for subject_tag in flashcard.subject_tags:
                await self._update_subject_mastery(
                    user_id=user_uuid,
                    subject_tag=subject_tag,
                    flashcard_reviews=1,
                    flashcard_correct=1 if was_correct else 0
                )

        except Exception as e:
            logger.error(f"Error updating mastery from flashcard {flashcard_id}: {e}")
            raise

    async def recalculate_all_masteries(self, user_id: str):
        """Recalculate all subject masteries for a user from scratch"""
        try:
            # Get all subjects for the user
            subjects = await self._get_user_subjects(user_id)

            for subject_tag in subjects:
                await self._recalculate_subject_mastery(user_id, subject_tag)

        except Exception as e:
            logger.error(f"Error recalculating masteries for user {user_id}: {e}")
            raise

    async def get_mastery_trends(self, user_id: str, days: int = 30) -> Dict[str, List[Dict]]:
        """Get mastery trends over time for visualization"""
        try:
            # This is a simplified implementation
            # In a real system, you'd track historical mastery values
            current_masteries = await self._get_current_masteries(user_id)

            # Return current values as trend data
            trends = {}
            for subject, mastery in current_masteries.items():
                trends[subject] = [{
                    'date': datetime.utcnow().isoformat(),
                    'mastery_percentage': float(mastery)
                }]

            return trends

        except Exception as e:
            logger.error(f"Error getting mastery trends: {e}")
            raise

    # Private helper methods

    async def _update_subject_mastery(
            self,
            user_id,
            subject_tag: str,
            quiz_questions: int = 0,
            quiz_correct: int = 0,
            flashcard_reviews: int = 0,
            flashcard_correct: int = 0
    ):
        """Update or create subject mastery record"""
        try:
            # Get existing mastery record
            query = select(SubjectMastery).where(
                and_(
                    SubjectMastery.user_id == user_id,
                    SubjectMastery.subject_tag == subject_tag
                )
            )
            result = await self.db.execute(query)
            mastery = result.scalar_one_or_none()

            if not mastery:
                # Create new mastery record
                mastery = SubjectMastery(
                    user_id=user_id,
                    subject_tag=subject_tag,
                    mastery_percentage=Decimal(0),
                    total_questions_answered=0,
                    correct_answers=0,
                    flashcards_mastered=0,
                    total_flashcards=0,
                    last_activity_date=datetime.utcnow()
                )
                self.db.add(mastery)

            # Update counters
            mastery.total_questions_answered += quiz_questions
            mastery.correct_answers += quiz_correct
            mastery.last_activity_date = datetime.utcnow()

            # Update flashcard stats if provided
            if flashcard_reviews > 0:
                # Get total flashcards for this subject
                flashcard_count_query = select(func.count(Flashcard.id)).where(
                    and_(
                        Flashcard.user_id == user_id,
                        Flashcard.subject_tags.contains([subject_tag])
                    )
                )
                total_flashcards = await self.db.scalar(flashcard_count_query) or 0
                mastery.total_flashcards = total_flashcards

                # Count mastered flashcards (leitner_box >= 4 indicates mastery)
                mastered_query = select(func.count(Flashcard.id)).where(
                    and_(
                        Flashcard.user_id == user_id,
                        Flashcard.subject_tags.contains([subject_tag]),
                        Flashcard.leitner_box >= 4
                    )
                )
                mastered_count = await self.db.scalar(mastered_query) or 0
                mastery.flashcards_mastered = mastered_count

            # Calculate overall mastery percentage
            mastery.mastery_percentage = await self._calculate_mastery_percentage(
                user_id, subject_tag
            )

            await self.db.commit()

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating subject mastery: {e}")
            raise

    async def _calculate_mastery_percentage(self, user_id: str, subject_tag: str) -> Decimal:
        """Calculate mastery percentage based on quiz and flashcard performance"""
        try:
            # Get quiz performance
            quiz_accuracy = await self._get_quiz_accuracy_for_subject(user_id, subject_tag)

            # Get flashcard performance
            flashcard_mastery = await self._get_flashcard_mastery_for_subject(user_id, subject_tag)

            # Weighted average: 60% quiz performance, 40% flashcard mastery
            if quiz_accuracy is not None and flashcard_mastery is not None:
                mastery = (quiz_accuracy * Decimal(0.6)) + (flashcard_mastery * Decimal(0.4))
            elif quiz_accuracy is not None:
                mastery = quiz_accuracy
            elif flashcard_mastery is not None:
                mastery = flashcard_mastery
            else:
                mastery = Decimal(0)

            return min(mastery, Decimal(100))  # Cap at 100%

        except Exception as e:
            logger.error(f"Error calculating mastery percentage: {e}")
            return Decimal(0)

    async def _get_quiz_accuracy_for_subject(self, user_id: str, subject_tag: str) -> Optional[Decimal]:
        """Get quiz accuracy for a specific subject"""
        try:
            # Get all quiz questions for this subject
            query = select(
                func.count(QuizQuestion.id).label('total'),
                func.sum(func.cast(QuizQuestion.is_correct, func.INTEGER)).label('correct')
            ).select_from(
                QuizQuestion.join(Quiz)
            ).where(
                and_(
                    Quiz.user_id == user_id,
                    Quiz.subject_tags.contains([subject_tag]),
                    QuizQuestion.is_correct.isnot(None)
                )
            )

            result = await self.db.execute(query)
            row = result.fetchone()

            if not row or not row.total:
                return None

            accuracy = (Decimal(row.correct or 0) / Decimal(row.total)) * Decimal(100)
            return accuracy

        except Exception as e:
            logger.error(f"Error getting quiz accuracy for subject {subject_tag}: {e}")
            return None

    async def _get_flashcard_mastery_for_subject(self, user_id: str, subject_tag: str) -> Optional[Decimal]:
        """Get flashcard mastery percentage for a specific subject"""
        try:
            # Count total and mastered flashcards
            total_query = select(func.count(Flashcard.id)).where(
                and_(
                    Flashcard.user_id == user_id,
                    Flashcard.subject_tags.contains([subject_tag])
                )
            )
            total_flashcards = await self.db.scalar(total_query) or 0

            if total_flashcards == 0:
                return None

            mastered_query = select(func.count(Flashcard.id)).where(
                and_(
                    Flashcard.user_id == user_id,
                    Flashcard.subject_tags.contains([subject_tag]),
                    Flashcard.leitner_box >= 4  # Consider box 4+ as mastered
                )
            )
            mastered_flashcards = await self.db.scalar(mastered_query) or 0

            mastery = (Decimal(mastered_flashcards) / Decimal(total_flashcards)) * Decimal(100)
            return mastery

        except Exception as e:
            logger.error(f"Error getting flashcard mastery for subject {subject_tag}: {e}")
            return None

    async def _get_user_subjects(self, user_id: str) -> List[str]:
        """Get all subjects for a user from their activities"""
        subjects = set()

        # Get subjects from quizzes
        quiz_query = select(Quiz.subject_tags).where(
            and_(
                Quiz.user_id == user_id,
                Quiz.subject_tags.isnot(None)
            )
        )
        result = await self.db.execute(quiz_query)
        for tag_list in result.scalars():
            if tag_list:
                subjects.update(tag_list)

        # Get subjects from flashcards
        flashcard_query = select(Flashcard.subject_tags).where(
            and_(
                Flashcard.user_id == user_id,
                Flashcard.subject_tags.isnot(None)
            )
        )
        result = await self.db.execute(flashcard_query)
        for tag_list in result.scalars():
            if tag_list:
                subjects.update(tag_list)

        return list(subjects)

    async def _recalculate_subject_mastery(self, user_id: str, subject_tag: str):
        """Recalculate mastery for a specific subject from scratch"""
        try:
            # Delete existing mastery record
            delete_query = select(SubjectMastery).where(
                and_(
                    SubjectMastery.user_id == user_id,
                    SubjectMastery.subject_tag == subject_tag
                )
            )
            result = await self.db.execute(delete_query)
            existing = result.scalar_one_or_none()
            if existing:
                await self.db.delete(existing)

            # Get quiz statistics
            quiz_stats = await self._get_quiz_stats_for_subject(user_id, subject_tag)

            # Create new mastery record
            mastery = SubjectMastery(
                user_id=user_id,
                subject_tag=subject_tag,
                total_questions_answered=quiz_stats['total_questions'],
                correct_answers=quiz_stats['correct_answers'],
                last_activity_date=datetime.utcnow()
            )

            # Calculate flashcard stats
            flashcard_stats = await self._get_flashcard_stats_for_subject(user_id, subject_tag)
            mastery.total_flashcards = flashcard_stats['total_flashcards']
            mastery.flashcards_mastered = flashcard_stats['mastered_flashcards']

            # Calculate mastery percentage
            mastery.mastery_percentage = await self._calculate_mastery_percentage(
                user_id, subject_tag
            )

            self.db.add(mastery)
            await self.db.commit()

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error recalculating subject mastery: {e}")
            raise

    async def _get_quiz_stats_for_subject(self, user_id: str, subject_tag: str) -> Dict[str, int]:
        """Get quiz statistics for a subject"""
        query = select(
            func.count(QuizQuestion.id).label('total'),
            func.sum(func.cast(QuizQuestion.is_correct, func.INTEGER)).label('correct')
        ).select_from(
            QuizQuestion.join(Quiz)
        ).where(
            and_(
                Quiz.user_id == user_id,
                Quiz.subject_tags.contains([subject_tag]),
                QuizQuestion.is_correct.isnot(None)
            )
        )

        result = await self.db.execute(query)
        row = result.fetchone()

        return {
            'total_questions': row.total or 0,
            'correct_answers': row.correct or 0
        }

    async def _get_flashcard_stats_for_subject(self, user_id: str, subject_tag: str) -> Dict[str, int]:
        """Get flashcard statistics for a subject"""
        total_query = select(func.count(Flashcard.id)).where(
            and_(
                Flashcard.user_id == user_id,
                Flashcard.subject_tags.contains([subject_tag])
            )
        )
        total_flashcards = await self.db.scalar(total_query) or 0

        mastered_query = select(func.count(Flashcard.id)).where(
            and_(
                Flashcard.user_id == user_id,
                Flashcard.subject_tags.contains([subject_tag]),
                Flashcard.leitner_box >= 4
            )
        )
        mastered_flashcards = await self.db.scalar(mastered_query) or 0

        return {
            'total_flashcards': total_flashcards,
            'mastered_flashcards': mastered_flashcards
        }

    async def _get_current_masteries(self, user_id: str) -> Dict[str, Decimal]:
        """Get current mastery percentages for all subjects"""
        query = select(SubjectMastery.subject_tag, SubjectMastery.mastery_percentage).where(
            SubjectMastery.user_id == user_id
        )

        result = await self.db.execute(query)
        masteries = {}

        for subject_tag, mastery_percentage in result.fetchall():
            masteries[subject_tag] = mastery_percentage

        return masteries