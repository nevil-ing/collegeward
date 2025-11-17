from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


class XPTransactionBase(BaseModel):
    activity_type: str = Field(..., max_length=30)
    activity_id: Optional[UUID] = None
    xp_earned: int = Field(..., ge=0)
    multiplier: Decimal = Field(default=Decimal("1.0"), ge=0)
    reason: str = Field(..., max_length=200)
    metadata: Optional[Dict[str, Any]] = None


class XPTransactionCreate(XPTransactionBase):
    pass


class XPTransactionResponse(XPTransactionBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    created_at: datetime


class UserGameProfileBase(BaseModel):
    total_xp: int = Field(default=0, ge=0)
    current_level: int = Field(default=1, ge=1)
    current_streak: int = Field(default=0, ge=0)
    longest_streak: int = Field(default=0, ge=0)
    last_activity_date: Optional[datetime] = None
    streak_freeze_count: int = Field(default=0, ge=0)
    leaderboard_visible: bool = Field(default=True)


class UserGameProfileCreate(UserGameProfileBase):
    pass


class UserGameProfileUpdate(BaseModel):
    leaderboard_visible: Optional[bool] = None
    streak_freeze_count: Optional[int] = Field(None, ge=0)


class UserGameProfileResponse(UserGameProfileBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime


class AchievementBase(BaseModel):
    code: str = Field(..., max_length=50)
    name: str = Field(..., max_length=100)
    description: str
    category: str = Field(..., max_length=30)
    badge_icon: str = Field(..., max_length=50)
    badge_color: str = Field(..., max_length=20)
    xp_reward: int = Field(default=0, ge=0)
    criteria: Dict[str, Any]
    is_active: bool = Field(default=True)


class AchievementCreate(AchievementBase):
    pass


class AchievementUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    badge_icon: Optional[str] = Field(None, max_length=50)
    badge_color: Optional[str] = Field(None, max_length=20)
    xp_reward: Optional[int] = Field(None, ge=0)
    criteria: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class AchievementResponse(AchievementBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime


class UserAchievementBase(BaseModel):
    achievement_id: UUID
    progress_data: Optional[Dict[str, Any]] = None


class UserAchievementCreate(UserAchievementBase):
    pass


class UserAchievementResponse(UserAchievementBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    earned_at: datetime
    achievement: AchievementResponse


class StudyStreakBase(BaseModel):
    streak_date: datetime
    activities_completed: int = Field(default=1, ge=0)
    total_study_time: int = Field(default=0, ge=0)  # seconds
    streak_maintained: bool = Field(default=True)
    freeze_used: bool = Field(default=False)


class StudyStreakCreate(StudyStreakBase):
    pass


class StudyStreakResponse(StudyStreakBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    created_at: datetime


class LeaderboardEntry(BaseModel):
    user_id: UUID
    display_name: Optional[str]
    total_xp: int
    current_level: int
    current_streak: int
    rank: int


class LeaderboardResponse(BaseModel):
    entries: List[LeaderboardEntry]
    user_rank: Optional[int] = None
    total_participants: int


class GamificationStats(BaseModel):
    total_xp: int
    current_level: int
    xp_to_next_level: int
    current_streak: int
    longest_streak: int
    streak_freeze_count: int
    achievements_earned: int
    total_achievements: int
    recent_achievements: List[UserAchievementResponse]
    recent_xp_transactions: List[XPTransactionResponse]


class ActivityXPReward(BaseModel):
    activity_type: str
    base_xp: int
    multiplier: Decimal = Field(default=Decimal("1.0"))
    bonus_reason: Optional[str] = None
    total_xp: int


class StreakStatus(BaseModel):
    current_streak: int
    longest_streak: int
    last_activity_date: Optional[datetime]
    streak_at_risk: bool
    hours_until_break: Optional[int]
    freeze_available: bool
    freeze_count: int


class NotificationPreferences(BaseModel):
    streak_reminders: bool = Field(default=True)
    achievement_notifications: bool = Field(default=True)
    xp_milestones: bool = Field(default=True)
    leaderboard_updates: bool = Field(default=False)
    reminder_time: Optional[str] = Field(None, description="Time in HH:MM format for daily reminders")


class GamificationSummary(BaseModel):
    profile: UserGameProfileResponse
    stats: GamificationStats
    streak_status: StreakStatus
    leaderboard_position: Optional[int] = None
    pending_achievements: List[AchievementResponse] = Field(default_factory=list)