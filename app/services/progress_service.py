from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc, Integer
from sqlalchemy.orm import selectinload


from app.models.user import User
from app.models.study_session import StudySession
from app.models.subject_mastery import SubjectMastery
from app.models.study_recommendation import StudyRecommendation
from app.models.flashcard import Flashcard
from app.models.quiz import Quiz
from app.models.quiz_question import QuizQuestion
from app.models.conversation import Conversation
from app.models.message import Message

from app.schemas.progress_schema import (
    StudySessionCreate, StudySessionResponse, SubjectMasteryResponse,
    StudyRecommendationResponse, ProgressAnalyticsResponse,
    StudyTimeAnalytics, PerformanceAnalytics, ReviewReminderResponse
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class ProgressService:
    """Service for progress analytics and recommendations"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_study_session(
            self,
            user_id: str,
            session_data: StudySessionCreate
    ) -> StudySessionResponse:
        """Create a new study session record"""
        try:
            # Simple approach: try UUID first, fall back to string
            user_id_value = user_id
            activity_id_value = session_data.activity_id

            # Check if we're dealing with UUID models by trying to create a StudySession
            # and seeing what type of user_id it expects
            try:
                # Try UUID conversion first
                if isinstance(user_id, str) and len(user_id.replace('-', '')) == 32:
                    user_id_value = uuid.UUID(user_id)
                if session_data.activity_id and isinstance(session_data.activity_id, str):
                    try:
                        activity_id_value = uuid.UUID(session_data.activity_id)
                    except ValueError:
                        activity_id_value = None
            except (ValueError, TypeError):
                # Fall back to string values
                user_id_value = str(user_id)
                activity_id_value = str(session_data.activity_id) if session_data.activity_id else None

            study_session = StudySession(
                user_id=user_id_value,
                activity_type=session_data.activity_type,
                activity_id=activity_id_value,
                subject_tags=session_data.subject_tags,
                duration_seconds=session_data.duration_seconds,
                started_at=session_data.started_at,
                ended_at=session_data.ended_at
            )

            self.db.add(study_session)
            await self.db.commit()
            await self.db.refresh(study_session)

            # Update subject mastery after creating session
            if session_data.subject_tags:
                await self._update_subject_activity(user_id_value, session_data.subject_tags)

            return StudySessionResponse(
                id=str(study_session.id),
                activity_type=study_session.activity_type,
                activity_id=str(study_session.activity_id) if study_session.activity_id else None,
                subject_tags=study_session.subject_tags,
                duration_seconds=study_session.duration_seconds,
                started_at=study_session.started_at,
                ended_at=study_session.ended_at,
                created_at=study_session.created_at
            )

        except Exception as e:
            logger.error(f"Error creating study session: {e}")
            await self.db.rollback()
            raise

    async def get_progress_analytics(self, user_id: str) -> ProgressAnalyticsResponse:
        """Get comprehensive progress analytics for a user"""
        try:
            # Handle both UUID objects (production) and string IDs (tests)
            user_id_value = self._convert_user_id(user_id)

            # Get study time analytics
            total_study_time = await self._get_total_study_time(user_id_value)
            study_time_7_days = await self._get_study_time_period(user_id_value, days=7)
            study_time_30_days = await self._get_study_time_period(user_id_value, days=30)

            # Calculate average daily study time (last 30 days)
            avg_daily_time = Decimal(study_time_30_days) / Decimal(30) if study_time_30_days > 0 else Decimal(0)

            # Get session counts
            total_sessions = await self._get_total_sessions(user_id_value)
            sessions_7_days = await self._get_sessions_period(user_id_value, days=7)

            # Get subject masteries
            subject_masteries = await self._get_subject_masteries(user_id_value)

            # Identify weak and strong areas
            weak_areas, strong_areas = await self._identify_performance_areas(user_id_value)

            # Calculate study streak
            study_streak = await self._calculate_study_streak(user_id_value)

            # Get last study date
            last_study_date = await self._get_last_study_date(user_id_value)

            # Get activity breakdown
            activity_breakdown = await self._get_activity_breakdown(user_id_value)

            return ProgressAnalyticsResponse(
                total_study_time_seconds=total_study_time,
                study_time_last_7_days=study_time_7_days,
                study_time_last_30_days=study_time_30_days,
                average_daily_study_time=avg_daily_time,
                total_sessions=total_sessions,
                sessions_last_7_days=sessions_7_days,
                subject_masteries=subject_masteries,
                weak_areas=weak_areas,
                strong_areas=strong_areas,
                study_streak_days=study_streak,
                last_study_date=last_study_date,
                activity_breakdown=activity_breakdown
            )

        except Exception as e:
            logger.error(f"Error getting progress analytics: {e}")
            raise

    async def get_study_time_analytics(self, user_id: str) -> StudyTimeAnalytics:
        """Get detailed study time analytics"""
        try:
            user_id_value = self._convert_user_id(user_id)

            # Get daily study times for last 30 days
            daily_times = await self._get_daily_study_times(user_id_value, days=30)

            # Get weekly totals for last 12 weeks
            weekly_totals = await self._get_weekly_study_times(user_id_value, weeks=12)

            # Get monthly totals for last 6 months
            monthly_totals = await self._get_monthly_study_times(user_id_value, months=6)

            # Get activity distribution
            activity_distribution = await self._get_activity_breakdown(user_id_value)

            return StudyTimeAnalytics(
                daily_study_times=daily_times,
                weekly_totals=weekly_totals,
                monthly_totals=monthly_totals,
                activity_distribution=activity_distribution
            )

        except Exception as e:
            logger.error(f"Error getting study time analytics: {e}")
            raise

    async def get_performance_analytics(self, user_id: str) -> PerformanceAnalytics:
        """Get performance analytics including accuracy and trends"""
        try:
            user_id_value = self._convert_user_id(user_id)

            # Calculate overall quiz accuracy
            quiz_accuracy = await self._calculate_quiz_accuracy(user_id_value)

            # Calculate flashcard success rate
            flashcard_success_rate = await self._calculate_flashcard_success_rate(user_id_value)

            # Determine improvement trend
            improvement_trend = await self._calculate_improvement_trend(user_id_value)

            # Get subject-specific performance
            subject_performance = await self._get_subject_performance(user_id_value)

            # Calculate recent performance change
            recent_change = await self._calculate_recent_performance_change(user_id_value)

            return PerformanceAnalytics(
                overall_quiz_accuracy=quiz_accuracy,
                flashcard_success_rate=flashcard_success_rate,
                improvement_trend=improvement_trend,
                subject_performance=subject_performance,
                recent_performance_change=recent_change
            )

        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            raise

    async def generate_study_recommendations(
            self,
            user_id: str,
            max_recommendations: int = 5,
            focus_areas: Optional[List[str]] = None
    ) -> List[StudyRecommendationResponse]:
        """Generate personalized study recommendations"""
        try:
            user_id_value = self._convert_user_id(user_id)

            # Clear old recommendations
            await self._clear_old_recommendations(user_id_value)

            recommendations = []

            # Get weak areas that need attention
            weak_areas = await self._get_weak_areas_for_recommendations(user_id_value, focus_areas)

            # Generate flashcard review recommendations
            flashcard_recs = await self._generate_flashcard_recommendations(user_id_value, weak_areas)
            recommendations.extend(flashcard_recs)

            # Generate quiz recommendations
            quiz_recs = await self._generate_quiz_recommendations(user_id_value, weak_areas)
            recommendations.extend(quiz_recs)

            # Generate study session recommendations
            session_recs = await self._generate_study_session_recommendations(user_id_value)
            recommendations.extend(session_recs)

            # Sort by priority and limit
            recommendations.sort(key=lambda x: x.priority_score, reverse=True)
            recommendations = recommendations[:max_recommendations]

            # Save recommendations to database
            for rec in recommendations:
                await self._save_recommendation(user_id_value, rec)

            return recommendations

        except Exception as e:
            logger.error(f"Error generating study recommendations: {e}")
            raise

    async def get_review_reminders(self, user_id: str) -> ReviewReminderResponse:
        """Get review reminders for flashcards and weak areas"""
        try:
            user_id_value = self._convert_user_id(user_id)
            now = datetime.utcnow()

            # Count flashcards due for review
            flashcards_due_query = select(func.count(Flashcard.id)).where(
                and_(
                    Flashcard.user_id == user_id_value,
                    Flashcard.next_review_date <= now
                )
            )
            flashcards_due = await self.db.scalar(flashcards_due_query) or 0

            # Count overdue flashcards (more than 1 day past due date)
            overdue_date = now - timedelta(days=1)
            overdue_query = select(func.count(Flashcard.id)).where(
                and_(
                    Flashcard.user_id == user_id_value,
                    Flashcard.next_review_date <= overdue_date
                )
            )
            overdue_flashcards = await self.db.scalar(overdue_query) or 0

            # Get next review time
            next_review_query = select(func.min(Flashcard.next_review_date)).where(
                and_(
                    Flashcard.user_id == user_id_value,
                    Flashcard.next_review_date > now
                )
            )
            next_review_time = await self.db.scalar(next_review_query)

            # Get subjects needing review
            subjects_query = select(Flashcard.subject_tags).where(
                and_(
                    Flashcard.user_id == user_id_value,
                    Flashcard.next_review_date <= now,
                    Flashcard.subject_tags.isnot(None)
                )
            ).distinct()

            result = await self.db.execute(subjects_query)
            subject_lists = result.scalars().all()
            subjects_needing_review = list(set(
                tag for tag_list in subject_lists if tag_list
                for tag in tag_list
            ))

            # Estimate review time (2 minutes per flashcard)
            estimated_time = (flashcards_due * 2) if flashcards_due > 0 else 0

            return ReviewReminderResponse(
                flashcards_due=flashcards_due,
                overdue_flashcards=overdue_flashcards,
                next_review_time=next_review_time,
                subjects_needing_review=subjects_needing_review,
                estimated_review_time_minutes=estimated_time
            )

        except Exception as e:
            logger.error(f"Error getting review reminders: {e}")
            raise

    # Private helper methods

    def _convert_user_id(self, user_id: str):
        """Convert user_id to appropriate type based on model"""
        # Simple approach: try UUID first, fall back to string
        try:
            if isinstance(user_id, str) and len(user_id.replace('-', '')) == 32:
                return uuid.UUID(user_id)
            else:
                return str(user_id)
        except (ValueError, TypeError):
            return str(user_id)

    async def _get_total_study_time(self, user_id) -> int:
        """Get total study time in seconds"""
        query = select(func.sum(StudySession.duration_seconds)).where(
            StudySession.user_id == user_id
        )
        result = await self.db.scalar(query)
        return result or 0

    async def _get_study_time_period(self, user_id, days: int) -> int:
        """Get study time for a specific period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = select(func.sum(StudySession.duration_seconds)).where(
            and_(
                StudySession.user_id == user_id,
                StudySession.started_at >= cutoff_date
            )
        )
        result = await self.db.scalar(query)
        return result or 0

    async def _get_total_sessions(self, user_id) -> int:
        """Get total number of study sessions"""
        query = select(func.count(StudySession.id)).where(
            StudySession.user_id == user_id
        )
        result = await self.db.scalar(query)
        return result or 0

    async def _get_sessions_period(self, user_id, days: int) -> int:
        """Get session count for a specific period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = select(func.count(StudySession.id)).where(
            and_(
                StudySession.user_id == user_id,
                StudySession.started_at >= cutoff_date
            )
        )
        result = await self.db.scalar(query)
        return result or 0

    async def _get_subject_masteries(self, user_id) -> List[SubjectMasteryResponse]:
        """Get all subject masteries for a user"""
        query = select(SubjectMastery).where(SubjectMastery.user_id == user_id)
        result = await self.db.execute(query)
        masteries = result.scalars().all()

        return [
            SubjectMasteryResponse(
                subject_tag=mastery.subject_tag,
                mastery_percentage=mastery.mastery_percentage,
                total_questions_answered=mastery.total_questions_answered,
                correct_answers=mastery.correct_answers,
                flashcards_mastered=mastery.flashcards_mastered,
                total_flashcards=mastery.total_flashcards,
                last_activity_date=mastery.last_activity_date
            )
            for mastery in masteries
        ]

    async def _identify_performance_areas(self, user_id) -> Tuple[List[str], List[str]]:
        """Identify weak and strong subject areas"""
        query = select(SubjectMastery).where(SubjectMastery.user_id == user_id)
        result = await self.db.execute(query)
        masteries = result.scalars().all()

        weak_areas = []
        strong_areas = []

        for mastery in masteries:
            if mastery.mastery_percentage < 60:
                weak_areas.append(mastery.subject_tag)
            elif mastery.mastery_percentage >= 80:
                strong_areas.append(mastery.subject_tag)

        return weak_areas, strong_areas

    async def _calculate_study_streak(self, user_id) -> int:
        """Calculate current study streak in days"""
        # Get distinct study dates in descending order
        query = select(func.date(StudySession.started_at)).where(
            StudySession.user_id == user_id
        ).distinct().order_by(desc(func.date(StudySession.started_at)))

        result = await self.db.execute(query)
        study_dates = [row[0] for row in result.fetchall()]

        if not study_dates:
            return 0

        # Calculate streak
        streak = 0
        current_date = datetime.utcnow().date()

        for study_date in study_dates:
            if study_date == current_date or study_date == current_date - timedelta(days=streak):
                streak += 1
                current_date = study_date
            else:
                break

        return streak

    async def _get_last_study_date(self, user_id) -> Optional[datetime]:
        """Get the last study session date"""
        query = select(func.max(StudySession.started_at)).where(
            StudySession.user_id == user_id
        )
        result = await self.db.scalar(query)
        return result

    async def _get_activity_breakdown(self, user_id) -> Dict[str, int]:
        """Get breakdown of study time by activity type"""
        query = select(
            StudySession.activity_type,
            func.sum(StudySession.duration_seconds)
        ).where(
            StudySession.user_id == user_id
        ).group_by(StudySession.activity_type)

        result = await self.db.execute(query)
        breakdown = {}

        for activity_type, total_seconds in result.fetchall():
            breakdown[activity_type] = total_seconds or 0

        return breakdown

    async def _update_subject_activity(self, user_id, subject_tags: List[str]):
        """Update subject mastery records after activity"""
        for subject_tag in subject_tags:
            # Check if mastery record exists
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
                    last_activity_date=datetime.utcnow()
                )
                self.db.add(mastery)
            else:
                # Update last activity date
                mastery.last_activity_date = datetime.utcnow()

        await self.db.commit()

    async def _get_daily_study_times(self, user_id: str, days: int) -> List[Dict[str, Any]]:
        """Get daily study times for the specified number of days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        query = select(
            func.date(StudySession.started_at).label('study_date'),
            func.sum(StudySession.duration_seconds).label('total_seconds')
        ).where(
            and_(
                StudySession.user_id == user_id,
                StudySession.started_at >= cutoff_date
            )
        ).group_by(func.date(StudySession.started_at)).order_by('study_date')

        result = await self.db.execute(query)
        daily_times = []

        for study_date, total_seconds in result.fetchall():
            daily_times.append({
                'date': study_date.isoformat(),
                'seconds': total_seconds or 0
            })

        return daily_times

    async def _get_weekly_study_times(self, user_id: str, weeks: int) -> List[Dict[str, Any]]:
        """Get weekly study time totals"""
        cutoff_date = datetime.utcnow() - timedelta(weeks=weeks)

        query = select(
            func.date_trunc('week', StudySession.started_at).label('week_start'),
            func.sum(StudySession.duration_seconds).label('total_seconds')
        ).where(
            and_(
                StudySession.user_id == user_id,
                StudySession.started_at >= cutoff_date
            )
        ).group_by(func.date_trunc('week', StudySession.started_at)).order_by('week_start')

        result = await self.db.execute(query)
        weekly_totals = []

        for week_start, total_seconds in result.fetchall():
            weekly_totals.append({
                'week_start': week_start.isoformat(),
                'seconds': total_seconds or 0
            })

        return weekly_totals

    async def _get_monthly_study_times(self, user_id: str, months: int) -> List[Dict[str, Any]]:
        """Get monthly study time totals"""
        cutoff_date = datetime.utcnow() - timedelta(days=months * 30)

        query = select(
            func.date_trunc('month', StudySession.started_at).label('month_start'),
            func.sum(StudySession.duration_seconds).label('total_seconds')
        ).where(
            and_(
                StudySession.user_id == user_id,
                StudySession.started_at >= cutoff_date
            )
        ).group_by(func.date_trunc('month', StudySession.started_at)).order_by('month_start')

        result = await self.db.execute(query)
        monthly_totals = []

        for month_start, total_seconds in result.fetchall():
            monthly_totals.append({
                'month_start': month_start.isoformat(),
                'seconds': total_seconds or 0
            })

        return monthly_totals

    async def _calculate_quiz_accuracy(self, user_id: str) -> Decimal:
        """Calculate overall quiz accuracy percentage"""
        query = select(
            func.count(QuizQuestion.id).label('total'),
            func.sum(func.cast(QuizQuestion.is_correct, Integer)).label('correct')
        ).select_from(
            QuizQuestion.join(Quiz)
        ).where(Quiz.user_id == user_id)

        result = await self.db.execute(query)
        row = result.fetchone()

        if not row or not row.total:
            return Decimal(0)

        return Decimal(row.correct or 0) / Decimal(row.total) * Decimal(100)

    async def _calculate_flashcard_success_rate(self, user_id: str) -> Decimal:
        """Calculate flashcard success rate"""
        query = select(
            func.sum(Flashcard.times_reviewed).label('total_reviews'),
            func.sum(Flashcard.times_correct).label('total_correct')
        ).where(Flashcard.user_id == user_id)

        result = await self.db.execute(query)
        row = result.fetchone()

        if not row or not row.total_reviews:
            return Decimal(0)

        return Decimal(row.total_correct or 0) / Decimal(row.total_reviews) * Decimal(100)

    async def _calculate_improvement_trend(self, user_id: str) -> str:
        """Calculate improvement trend based on recent performance"""
        # Compare last 30 days vs previous 30 days
        now = datetime.utcnow()
        recent_start = now - timedelta(days=30)
        previous_start = now - timedelta(days=60)

        # Recent performance
        recent_query = select(
            func.count(QuizQuestion.id).label('total'),
            func.sum(func.cast(QuizQuestion.is_correct, Integer)).label('correct')
        ).select_from(
            QuizQuestion.join(Quiz)
        ).where(
            and_(
                Quiz.user_id == user_id,
                Quiz.completed_at >= recent_start
            )
        )

        recent_result = await self.db.execute(recent_query)
        recent_row = recent_result.fetchone()

        # Previous performance
        previous_query = select(
            func.count(QuizQuestion.id).label('total'),
            func.sum(func.cast(QuizQuestion.is_correct, Integer)).label('correct')
        ).select_from(
            QuizQuestion.join(Quiz)
        ).where(
            and_(
                Quiz.user_id == user_id,
                Quiz.completed_at >= previous_start,
                Quiz.completed_at < recent_start
            )
        )

        previous_result = await self.db.execute(previous_query)
        previous_row = previous_result.fetchone()

        # Calculate trend
        if not recent_row or not recent_row.total:
            return "stable"

        recent_accuracy = (recent_row.correct or 0) / recent_row.total

        if not previous_row or not previous_row.total:
            return "stable"

        previous_accuracy = (previous_row.correct or 0) / previous_row.total

        if recent_accuracy > previous_accuracy + 0.05:  # 5% improvement
            return "improving"
        elif recent_accuracy < previous_accuracy - 0.05:  # 5% decline
            return "declining"
        else:
            return "stable"

    async def _get_subject_performance(self, user_id: str) -> Dict[str, Decimal]:
        """Get performance by subject"""
        query = select(SubjectMastery.subject_tag, SubjectMastery.mastery_percentage).where(
            SubjectMastery.user_id == user_id
        )

        result = await self.db.execute(query)
        performance = {}

        for subject_tag, mastery_percentage in result.fetchall():
            performance[subject_tag] = mastery_percentage

        return performance

    async def _calculate_recent_performance_change(self, user_id: str) -> Decimal:
        """Calculate recent performance change percentage"""
        # This is a simplified implementation
        # In a real system, you'd track historical mastery percentages
        return Decimal(0)  # Placeholder

    async def _clear_old_recommendations(self, user_id: str):
        """Clear old or expired recommendations"""
        now = datetime.utcnow()

        # Mark expired recommendations as inactive
        query = select(StudyRecommendation).where(
            and_(
                StudyRecommendation.user_id == user_id,
                or_(
                    StudyRecommendation.expires_at <= now,
                    StudyRecommendation.created_at <= now - timedelta(days=7)
                )
            )
        )

        result = await self.db.execute(query)
        old_recommendations = result.scalars().all()

        for rec in old_recommendations:
            rec.is_active = False

        await self.db.commit()

    async def _get_weak_areas_for_recommendations(
            self,
            user_id: str,
            focus_areas: Optional[List[str]] = None
    ) -> List[str]:
        """Get weak areas that need attention"""
        query = select(SubjectMastery.subject_tag).where(
            and_(
                SubjectMastery.user_id == user_id,
                SubjectMastery.mastery_percentage < 70
            )
        )

        if focus_areas:
            query = query.where(SubjectMastery.subject_tag.in_(focus_areas))

        result = await self.db.execute(query)
        return [row[0] for row in result.fetchall()]

    async def _generate_flashcard_recommendations(
            self,
            user_id: str,
            weak_areas: List[str]
    ) -> List[StudyRecommendationResponse]:
        """Generate flashcard review recommendations"""
        recommendations = []

        # Check for due flashcards
        now = datetime.utcnow()
        due_query = select(func.count(Flashcard.id)).where(
            and_(
                Flashcard.user_id == user_id,
                Flashcard.next_review_date <= now
            )
        )
        due_count = await self.db.scalar(due_query) or 0

        if due_count > 0:
            recommendations.append(StudyRecommendationResponse(
                id="",  # Will be set when saved
                recommendation_type="review_flashcards",
                subject_tag=None,
                priority_score=Decimal(90),
                reason=f"You have {due_count} flashcards due for review",
                action_data={"flashcard_count": due_count},
                is_active=True,
                expires_at=now + timedelta(hours=24),
                created_at=now
            ))

        # Recommend flashcard creation for weak areas
        for subject in weak_areas[:2]:  # Limit to top 2 weak areas
            recommendations.append(StudyRecommendationResponse(
                id="",
                recommendation_type="create_flashcards",
                subject_tag=subject,
                priority_score=Decimal(75),
                reason=f"Create more flashcards for {subject} to improve mastery",
                action_data={"subject": subject},
                is_active=True,
                expires_at=now + timedelta(days=3),
                created_at=now
            ))

        return recommendations

    async def _generate_quiz_recommendations(
            self,
            user_id: str,
            weak_areas: List[str]
    ) -> List[StudyRecommendationResponse]:
        """Generate quiz recommendations"""
        recommendations = []
        now = datetime.utcnow()

        # Recommend quizzes for weak areas
        for subject in weak_areas[:3]:  # Limit to top 3 weak areas
            recommendations.append(StudyRecommendationResponse(
                id="",
                recommendation_type="take_quiz",
                subject_tag=subject,
                priority_score=Decimal(80),
                reason=f"Take a quiz on {subject} to identify knowledge gaps",
                action_data={"subject": subject, "difficulty": "medium"},
                is_active=True,
                expires_at=now + timedelta(days=2),
                created_at=now
            ))

        return recommendations

    async def _generate_study_session_recommendations(self, user_id: str) -> List[StudyRecommendationResponse]:
        """Generate general study session recommendations"""
        recommendations = []
        now = datetime.utcnow()

        # Check if user hasn't studied recently
        last_session_query = select(func.max(StudySession.started_at)).where(
            StudySession.user_id == user_id
        )
        last_session = await self.db.scalar(last_session_query)

        if not last_session or last_session < now - timedelta(days=2):
            recommendations.append(StudyRecommendationResponse(
                id="",
                recommendation_type="start_study_session",
                subject_tag=None,
                priority_score=Decimal(85),
                reason="It's been a while since your last study session",
                action_data={"suggested_duration": 30},
                is_active=True,
                expires_at=now + timedelta(hours=12),
                created_at=now
            ))

        return recommendations

    async def _save_recommendation(self, user_id: str, rec: StudyRecommendationResponse):
        """Save a recommendation to the database"""
        recommendation = StudyRecommendation(
            user_id=user_id,
            recommendation_type=rec.recommendation_type,
            subject_tag=rec.subject_tag,
            priority_score=rec.priority_score,
            reason=rec.reason,
            action_data=rec.action_data,
            is_active=rec.is_active,
            expires_at=rec.expires_at
        )

        self.db.add(recommendation)
        await self.db.commit()
        await self.db.refresh(recommendation)

        # Update the ID in the response
        rec.id = str(recommendation.id)