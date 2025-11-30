import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from app.models.quiz import Quiz
from app.models.note import Note
from app.models.quiz_question import QuizQuestion
from app.schemas.quiz_schema import (
    QuizCreate, QuizResponse, QuizSubmission, QuizResult,
    QuizStats, QuizQuestionResponse
)
from app.ai.quiz_generator import MedicalQuizGenerator
from app.services.ai_service import ai_service_manager
from app.utils.exceptions import NotFoundError, ValidationError

logger = logging.getLogger(__name__)


class QuizService:
    """Service for quiz generation, management, and assessment"""

    def __init__(self):
        self.quiz_generator = MedicalQuizGenerator(ai_service_manager)

    def _to_str_id(self, id_value):
        """Convert UUID to string for database queries"""
        if isinstance(id_value, UUID):
            return str(id_value)
        return id_value

    def _to_str_id(self, id_value: Union[UUID, str]) -> str:
        """Convert UUID or string to string for database compatibility"""
        return str(id_value) if id_value else None

    async def generate_quiz_from_notes(
            self,
            db: AsyncSession,
            user_id: UUID,
            note_ids: Optional[List[UUID]] = None,
            num_questions: int = 10,
            difficulty_level: Optional[int] = None,
            focus_subjects: Optional[List[str]] = None,
            quiz_title: Optional[str] = None
    ) -> QuizResponse:
        """
        Generate a quiz from user's notes

        Args:
            db: Database session
            user_id: User ID
            note_ids: Specific note IDs to use (if None, uses all user notes)
            num_questions: Number of questions to generate
            difficulty_level: Target difficulty (1-5)
            focus_subjects: Specific subjects to focus on
            quiz_title: Custom quiz title

        Returns:
            Created quiz with questions
        """
        try:
            # Get user's notes content
            notes_content = await self._get_notes_content(db, user_id, note_ids)

            if not notes_content:
                raise ValidationError("No study materials found to generate quiz from")

            # Generate quiz using AI
            quiz_create = await self.quiz_generator.generate_quiz_from_materials(
                text_content=notes_content,
                num_questions=num_questions,
                difficulty_level=difficulty_level,
                focus_subjects=focus_subjects,
                quiz_title=quiz_title,
                include_clinical_scenarios=True
            )

            # Save quiz to database
            quiz = await self._create_quiz_in_db(db, user_id, quiz_create)

            logger.info(f"Generated quiz {quiz.id} for user {user_id}")
            return quiz

        except Exception as e:
            logger.error(f"Quiz generation failed for user {user_id}: {str(e)}")
            raise

    async def generate_targeted_quiz(
            self,
            db: AsyncSession,
            user_id: UUID,
            weak_areas: List[str],
            num_questions: int = 15
    ) -> QuizResponse:
        """
        Generate a quiz targeting user's weak areas

        Args:
            db: Database session
            user_id: User ID
            weak_areas: Subject areas where user needs improvement
            num_questions: Number of questions to generate

        Returns:
            Targeted quiz focusing on weak areas
        """
        try:
            # Get user's notes content
            notes_content = await self._get_notes_content(db, user_id)

            if not notes_content:
                raise ValidationError("No study materials found for targeted quiz")

            # Generate targeted quiz
            quiz_create = await self.quiz_generator.generate_targeted_quiz(
                weak_areas=weak_areas,
                user_notes_content=notes_content,
                num_questions=num_questions,
                focus_on_weaknesses=True
            )

            # Save quiz to database
            quiz = await self._create_quiz_in_db(db, user_id, quiz_create)

            logger.info(f"Generated targeted quiz {quiz.id} for weak areas: {weak_areas}")
            return quiz

        except Exception as e:
            logger.error(f"Targeted quiz generation failed: {str(e)}")
            raise

    async def get_quiz(
            self,
            db: AsyncSession,
            quiz_id: UUID,
            user_id: UUID,
            include_answers: bool = False
    ) -> QuizResponse:
        """
        Get a quiz by ID

        Args:
            db: Database session
            quiz_id: Quiz ID
            user_id: User ID (for authorization)
            include_answers: Whether to include correct answers

        Returns:
            Quiz with questions
        """
        try:
            # Query quiz with questions
            stmt = (
                select(Quiz)
                .options(selectinload(Quiz.questions))
                .where(and_(Quiz.id == self._to_str_id(quiz_id), Quiz.user_id == self._to_str_id(user_id)))
            )
            result = await db.execute(stmt)
            quiz = result.scalar_one_or_none()

            if not quiz:
                raise NotFoundError(f"Quiz {quiz_id} not found")

            # Convert to response format
            questions = []
            for q in sorted(quiz.questions, key=lambda x: x.question_order):
                question_response = QuizQuestionResponse(
                    id=q.id,
                    quiz_id=q.quiz_id,
                    question_text=q.question_text,
                    options=q.options if isinstance(q.options, list) else q.options.get('options', []),
                    correct_answer=q.correct_answer if include_answers else None,
                    explanation=q.explanation if include_answers else None,
                    user_answer=q.user_answer,
                    is_correct=q.is_correct,
                    question_order=q.question_order
                )
                questions.append(question_response)

            return QuizResponse(
                id=quiz.id,
                user_id=quiz.user_id,
                title=quiz.title,
                subject_tags=quiz.subject_tags,
                total_questions=quiz.total_questions,
                score=quiz.score,
                percentage=quiz.percentage,
                time_taken=quiz.time_taken,
                completed_at=quiz.completed_at,
                questions=questions,
                created_at=quiz.created_at,
                updated_at=quiz.updated_at
            )

        except Exception as e:
            logger.error(f"Failed to get quiz {quiz_id}: {str(e)}")
            raise

    async def save_quiz_answer(
            self,
            db: AsyncSession,
            quiz_id: UUID,
            user_id: UUID,
            question_order: int,
            answer: int
    ) -> None:
        """
        Save a partial answer for a quiz question (allows resuming quizzes)

        Args:
            db: Database session
            quiz_id: Quiz ID
            user_id: User ID
            question_order: Question order number (1-based)
            answer: Selected answer index (0-based)
        """
        try:
            # Get quiz with questions
            stmt = (
                select(Quiz)
                .options(selectinload(Quiz.questions))
                .where(and_(Quiz.id == self._to_str_id(quiz_id), Quiz.user_id == self._to_str_id(user_id)))
            )
            result = await db.execute(stmt)
            quiz = result.scalar_one_or_none()

            if not quiz:
                raise NotFoundError(f"Quiz {quiz_id} not found")

            if quiz.completed_at:
                raise ValidationError("Cannot modify answers for a completed quiz")

            # Find the question by order
            question = next((q for q in quiz.questions if q.question_order == question_order), None)

            if not question:
                raise NotFoundError(f"Question with order {question_order} not found in quiz")

            # Validate answer index
            options = question.options if isinstance(question.options, list) else question.options.get('options', [])
            if answer < 0 or answer >= len(options):
                raise ValidationError(f"Invalid answer index: {answer}. Must be between 0 and {len(options) - 1}")

            # Save the answer
            question.user_answer = answer
            # Don't set is_correct until quiz is submitted

            await db.commit()
            await db.refresh(question)

            logger.info(f"Saved answer for quiz {quiz_id}, question {question_order}: {answer}")

        except NotFoundError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to save quiz answer: {str(e)}")
            await db.rollback()
            raise

    async def submit_quiz(
            self,
            db: AsyncSession,
            quiz_id: UUID,
            user_id: UUID,
            submission: QuizSubmission
    ) -> QuizResult:
        """
        Submit quiz answers and calculate results

        Args:
            db: Database session
            quiz_id: Quiz ID
            user_id: User ID
            submission: Quiz submission with answers

        Returns:
            Quiz results with scoring and explanations
        """
        try:
            # Get quiz with questions
            stmt = (
                select(Quiz)
                .options(selectinload(Quiz.questions))
                .where(and_(Quiz.id == self._to_str_id(quiz_id), Quiz.user_id == self._to_str_id(user_id)))
            )
            result = await db.execute(stmt)
            quiz = result.scalar_one_or_none()

            if not quiz:
                raise NotFoundError(f"Quiz {quiz_id} not found")

            if quiz.completed_at:
                raise ValidationError("Quiz has already been completed")

            # Convert string keys to int keys for processing
            answers_int_keys = submission.get_answers_as_int_keys()

            # Process answers and calculate score
            score_data = await self._calculate_quiz_score(
                quiz, answers_int_keys, submission.time_taken
            )

            # Update quiz with results
            quiz.score = score_data['score']
            quiz.percentage = score_data['percentage']
            quiz.time_taken = submission.time_taken
            quiz.completed_at = datetime.utcnow()

            # Convert string keys to int keys for processing
            answers_int_keys = submission.get_answers_as_int_keys()

            # Update questions with user answers
            for question in quiz.questions:
                user_answer = answers_int_keys.get(question.question_order)
                if user_answer is not None:
                    question.user_answer = user_answer
                    question.is_correct = (user_answer == question.correct_answer)

            await db.commit()
            await db.refresh(quiz)

            # Create result response
            questions_with_answers = []
            for q in sorted(quiz.questions, key=lambda x: x.question_order):
                question_response = QuizQuestionResponse(
                    id=q.id,
                    quiz_id=q.quiz_id,
                    question_text=q.question_text,
                    options=q.options if isinstance(q.options, list) else q.options.get('options', []),
                    correct_answer=q.correct_answer,
                    explanation=q.explanation,
                    user_answer=q.user_answer,
                    is_correct=q.is_correct,
                    question_order=q.question_order
                )
                questions_with_answers.append(question_response)

            result = QuizResult(
                quiz_id=quiz.id,
                score=quiz.score,
                percentage=quiz.percentage,
                time_taken=quiz.time_taken,
                correct_answers=score_data['correct_count'],
                total_questions=quiz.total_questions,
                questions_with_answers=questions_with_answers,
                completed_at=quiz.completed_at
            )

            logger.info(f"Quiz {quiz_id} completed with score {quiz.score}/{quiz.total_questions}")
            return result

        except Exception as e:
            logger.error(f"Quiz submission failed for {quiz_id}: {str(e)}")
            raise

    async def get_user_quizzes(
            self,
            db: AsyncSession,
            user_id: UUID,
            limit: int = 20,
            offset: int = 0,
            completed_only: bool = False
    ) -> List[QuizResponse]:
        """
        Get user's quizzes with pagination

        Args:
            db: Database session
            user_id: User ID
            limit: Maximum number of quizzes to return
            offset: Number of quizzes to skip
            completed_only: Whether to return only completed quizzes

        Returns:
            List of user's quizzes
        """
        try:
            # Load quizzes with questions to show progress
            stmt = (
                select(Quiz)
                .options(selectinload(Quiz.questions))
                .where(Quiz.user_id == self._to_str_id(user_id))
            )

            if completed_only:
                stmt = stmt.where(Quiz.completed_at.isnot(None))

            stmt = stmt.order_by(Quiz.created_at.desc()).limit(limit).offset(offset)

            result = await db.execute(stmt)
            quizzes = result.scalars().all()

            quiz_responses = []
            for quiz in quizzes:
                # Convert questions to response format
                questions = []
                for q in sorted(quiz.questions, key=lambda x: x.question_order):
                    question_response = QuizQuestionResponse(
                        id=q.id,
                        quiz_id=q.quiz_id,
                        question_text=q.question_text,
                        options=q.options if isinstance(q.options, list) else q.options.get('options', []),
                        correct_answer=q.correct_answer if quiz.completed_at else None,  # Hide answer if not completed
                        explanation=q.explanation if quiz.completed_at else None,  # Hide explanation if not completed
                        user_answer=q.user_answer,
                        is_correct=q.is_correct,
                        question_order=q.question_order
                    )
                    questions.append(question_response)

                quiz_response = QuizResponse(
                    id=quiz.id,
                    user_id=quiz.user_id,
                    title=quiz.title,
                    subject_tags=quiz.subject_tags,
                    total_questions=quiz.total_questions,
                    score=quiz.score,
                    percentage=quiz.percentage,
                    time_taken=quiz.time_taken,
                    completed_at=quiz.completed_at,
                    questions=questions,  # Include questions to show progress
                    created_at=quiz.created_at,
                    updated_at=quiz.updated_at
                )
                quiz_responses.append(quiz_response)

            return quiz_responses

        except Exception as e:
            logger.error(f"Failed to get quizzes for user {user_id}: {str(e)}")
            raise

    async def get_quiz_statistics(
            self,
            db: AsyncSession,
            user_id: UUID,
            days: int = 30
    ) -> QuizStats:
        """
        Get user's quiz statistics and performance analytics

        Args:
            db: Database session
            user_id: User ID
            days: Number of days to analyze

        Returns:
            Quiz statistics and weak areas
        """
        try:
            # Convert to string for compatibility with test models
            user_id_str = str(user_id)

            # Date range for analysis
            since_date = datetime.utcnow() - timedelta(days=days)

            # Get completed quizzes in date range
            stmt = (
                select(Quiz)
                .options(selectinload(Quiz.questions))
                .where(
                    and_(
                        Quiz.user_id == user_id_str,
                        Quiz.completed_at.isnot(None),
                        Quiz.completed_at >= since_date
                    )
                )
            )
            result = await db.execute(stmt)
            quizzes = result.scalars().all()

            if not quizzes:
                return QuizStats(
                    total_quizzes=0,
                    completed_quizzes=0,
                    average_score=0.0,
                    best_score=0.0,
                    total_time_spent=0,
                    weak_areas=[]
                )

            # Calculate statistics
            total_quizzes = len(quizzes)
            completed_quizzes = len([q for q in quizzes if q.completed_at])

            scores = [float(q.percentage) for q in quizzes if q.percentage is not None]
            average_score = sum(scores) / len(scores) if scores else 0.0
            best_score = max(scores) if scores else 0.0

            total_time = sum(q.time_taken for q in quizzes if q.time_taken) or 0

            # Analyze weak areas
            weak_areas = await self._analyze_weak_areas(quizzes)

            return QuizStats(
                total_quizzes=total_quizzes,
                completed_quizzes=completed_quizzes,
                average_score=average_score,
                best_score=best_score,
                total_time_spent=total_time,
                weak_areas=weak_areas
            )

        except Exception as e:
            logger.error(f"Failed to get quiz statistics for user {user_id}: {str(e)}")
            raise

    async def identify_weak_areas(
            self,
            db: AsyncSession,
            user_id: UUID,
            min_questions: int = 5
    ) -> List[str]:
        """
        Identify subject areas where user needs improvement

        Args:
            db: Database session
            user_id: User ID
            min_questions: Minimum questions needed to consider a subject

        Returns:
            List of weak subject areas
        """
        try:
            # Get completed quizzes with questions
            stmt = (
                select(Quiz)
                .options(selectinload(Quiz.questions))
                .where(
                    and_(
                        Quiz.user_id == self._to_str_id(user_id),
                        Quiz.completed_at.isnot(None)
                    )
                )
            )
            result = await db.execute(stmt)
            quizzes = result.scalars().all()

            return await self._analyze_weak_areas(quizzes, min_questions)

        except Exception as e:
            logger.error(f"Failed to identify weak areas for user {user_id}: {str(e)}")
            raise

    async def _get_notes_content(
            self,
            db: AsyncSession,
            user_id: UUID,
            note_ids: Optional[List[UUID]] = None
    ) -> str:
        """Get concatenated content from user's notes"""
        try:
            stmt = select(Note).where(
                and_(
                    Note.user_id == self._to_str_id(user_id),
                    Note.processing_status == "completed",
                    Note.extracted_text.isnot(None)
                )
            )

            if note_ids:
                note_ids_str = [self._to_str_id(nid) for nid in note_ids]
                stmt = stmt.where(Note.id.in_(note_ids_str))

            result = await db.execute(stmt)
            notes = result.scalars().all()

            # Concatenate all note content
            content_parts = []
            for note in notes:
                if note.extracted_text:
                    content_parts.append(f"=== {note.filename} ===\n{note.extracted_text}")

            return "\n\n".join(content_parts)

        except Exception as e:
            logger.error(f"Failed to get notes content: {str(e)}")
            return ""

    async def _create_quiz_in_db(
            self,
            db: AsyncSession,
            user_id: UUID,
            quiz_create: QuizCreate
    ) -> QuizResponse:
        """Create quiz and questions in database"""
        try:
            # Create quiz
            quiz = Quiz(
                user_id=self._to_str_id(user_id),
                title=quiz_create.title,
                subject_tags=quiz_create.subject_tags,
                total_questions=len(quiz_create.questions)
            )

            db.add(quiz)
            await db.flush()
            await db.refresh(quiz)

            # Create questions
            for question_create in quiz_create.questions:
                question = QuizQuestion(
                    quiz_id=quiz.id,
                    question_text=question_create.question_text,
                    options={"options": question_create.options},  # Store as JSONB
                    correct_answer=question_create.correct_answer,
                    explanation=question_create.explanation,
                    question_order=question_create.question_order
                )
                db.add(question)

            await db.commit()

            # Reload quiz with questions
            stmt = (
                select(Quiz)
                .options(selectinload(Quiz.questions))
                .where(Quiz.id == quiz.id)
            )
            result = await db.execute(stmt)
            quiz = result.scalar_one()

            # Convert questions to response format
            questions = []
            for q in sorted(quiz.questions, key=lambda x: x.question_order):
                question_response = QuizQuestionResponse(
                    id=q.id,
                    quiz_id=q.quiz_id,
                    question_text=q.question_text,
                    options=q.options if isinstance(q.options, list) else q.options.get('options', []),
                    correct_answer=q.correct_answer,  # Include correct answer (frontend should hide until submission)
                    explanation=q.explanation,  # Include explanation (frontend should hide until submission)
                    user_answer=None,
                    is_correct=None,
                    question_order=q.question_order
                )
                questions.append(question_response)

            # Return quiz response with questions
            return QuizResponse(
                id=quiz.id,
                user_id=quiz.user_id,
                title=quiz.title,
                subject_tags=quiz.subject_tags,
                total_questions=quiz.total_questions,
                score=quiz.score,
                percentage=quiz.percentage,
                time_taken=quiz.time_taken,
                completed_at=quiz.completed_at,
                questions=questions,
                created_at=quiz.created_at,
                updated_at=quiz.updated_at
            )

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create quiz in database: {str(e)}")
            raise

    async def _calculate_quiz_score(
            self,
            quiz: Quiz,
            answers: Dict[int, int],
            time_taken: Optional[int]
    ) -> Dict[str, Any]:
        """Calculate quiz score and statistics"""
        correct_count = 0
        total_questions = len(quiz.questions)

        for question in quiz.questions:
            user_answer = answers.get(question.question_order)
            if user_answer == question.correct_answer:
                correct_count += 1

        percentage = Decimal((correct_count / total_questions) * 100) if total_questions > 0 else Decimal(0)

        return {
            'score': correct_count,
            'correct_count': correct_count,
            'total_questions': total_questions,
            'percentage': percentage
        }

    async def _analyze_weak_areas(
            self,
            quizzes: List[Quiz],
            min_questions: int = 5
    ) -> List[str]:
        """Analyze quiz performance to identify weak subject areas"""
        subject_performance = {}

        for quiz in quizzes:
            if not quiz.subject_tags:
                continue

            # Calculate quiz performance
            if quiz.questions:
                correct_count = sum(1 for q in quiz.questions if q.is_correct)
                total_count = len(quiz.questions)
                performance = correct_count / total_count if total_count > 0 else 0

                # Track performance by subject
                for subject in quiz.subject_tags:
                    if subject not in subject_performance:
                        subject_performance[subject] = {'scores': [], 'question_count': 0}

                    subject_performance[subject]['scores'].append(performance)
                    subject_performance[subject]['question_count'] += total_count

        # Identify weak areas (below 70% average performance)
        weak_areas = []
        for subject, data in subject_performance.items():
            if data['question_count'] >= min_questions:
                avg_performance = sum(data['scores']) / len(data['scores'])
                if avg_performance < 0.7:  # Below 70%
                    weak_areas.append(subject)

        # Sort by performance (worst first)
        weak_areas.sort(key=lambda s: sum(subject_performance[s]['scores']) / len(subject_performance[s]['scores']))

        return weak_areas[:5]  # Return top 5 weak areas