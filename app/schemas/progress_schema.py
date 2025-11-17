from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, ConfigDict


class StudySessionCreate(BaseModel):
    activity_type: str = Field(..., description="Type of study activity")
    activity_id: Optional[str] = Field(None, description="ID of the specific activity")
    subject_tags: Optional[List[str]] = Field(None, description="Subject tags for the session")
    duration_seconds: int = Field(..., gt=0, description="Duration of the session in seconds")
    started_at: datetime = Field(..., description="When the session started")
    ended_at: datetime = Field(..., description="When the session ended")


class StudySessionResponse(BaseModel):
    id: str
    activity_type: str
    activity_id: Optional[str]
    subject_tags: Optional[List[str]]
    duration_seconds: int
    started_at: datetime
    ended_at: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SubjectMasteryResponse(BaseModel):
    subject_tag: str
    mastery_percentage: Decimal
    total_questions_answered: int
    correct_answers: int
    flashcards_mastered: int
    total_flashcards: int
    last_activity_date: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class StudyRecommendationResponse(BaseModel):
    id: str
    recommendation_type: str
    subject_tag: Optional[str]
    priority_score: Decimal
    reason: str
    action_data: Optional[Dict[str, Any]]
    is_active: bool
    expires_at: Optional[datetime]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProgressAnalyticsResponse(BaseModel):
    total_study_time_seconds: int
    study_time_last_7_days: int
    study_time_last_30_days: int
    average_daily_study_time: Decimal
    total_sessions: int
    sessions_last_7_days: int
    subject_masteries: List[SubjectMasteryResponse]
    weak_areas: List[str]
    strong_areas: List[str]
    study_streak_days: int
    last_study_date: Optional[datetime]
    activity_breakdown: Dict[str, int]


class StudyTimeAnalytics(BaseModel):
    daily_study_times: List[Dict[str, Any]]
    weekly_totals: List[Dict[str, Any]]
    monthly_totals: List[Dict[str, Any]]
    activity_distribution: Dict[str, int]


class PerformanceAnalytics(BaseModel):
    overall_quiz_accuracy: Decimal
    flashcard_success_rate: Decimal
    improvement_trend: str
    subject_performance: Dict[str, Decimal]
    recent_performance_change: Decimal


class RecommendationRequest(BaseModel):
    max_recommendations: Optional[int] = Field(5, ge=1, le=20)
    focus_areas: Optional[List[str]] = Field(None, description="Specific subjects to focus on")
    exclude_types: Optional[List[str]] = Field(None, description="Recommendation types to exclude")


class ReviewReminderResponse(BaseModel):
    flashcards_due: int
    overdue_flashcards: int
    next_review_time: Optional[datetime]
    subjects_needing_review: List[str]
    estimated_review_time_minutes: int