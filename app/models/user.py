from typing import Optional, List
from sqlalchemy import String, Integer, Boolean
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.db.base import BaseModel
from app.models.conversation import Conversation
from app.models.flashcard import Flashcard
from app.models.note import Note
from app.models.quiz import Quiz
from app.models.study_recommendation import StudyRecommendation
from app.models.study_session import StudySession
from app.models.study_streak import StudyStreak
from app.models.subject_mastery import SubjectMastery
from app.models.user_game_profile import UserGameProfile


class User(BaseModel):
    __tablename__ = "users"

    firebase_uid: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(100))
    study_level: Mapped[Optional[str]] = mapped_column(String(50))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

#relationships
    notes: Mapped[List["Note"]] = relationship("Note", back_populates="user", cascade="all, delete-orphan")
    conversation: Mapped[List["Conversation"]] = relationship("Conversation", back_populates="user",cascade="all, delete-orphan")
    flashcards: Mapped[List["Flashcard"]] = relationship("Flashcard", back_populates="user", cascade="all, delete-orphan")
    quizzes: Mapped[List["Quiz"]] = relationship("Quiz", back_populates="user", cascade="all, delete-orphan")
    study_sessions: Mapped[List["StudySession"]] = relationship("StudySession", back_populates="user",cascade="all, delete-orphan")
    subject_masteries: Mapped[List["SubjectMastery"]] = relationship("SubjectMastery", back_populates="user",  cascade="all, delete-orphan")
    study_recommendations: Mapped[List["StudyRecommendation"]] = relationship("StudyRecommendation",back_populates="user", cascade="all, delete-orphan")
    game_profile: Mapped[Optional["UserGameProfile"]] = relationship("UserGameProfile", back_populates="user",                                                                cascade="all, delete-orphan", uselist=False)
    study_streaks: Mapped[List["StudyStreak"]] = relationship("StudyStreak", back_populates="user", cascade="all, delete-orphan")
