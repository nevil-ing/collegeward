# Import all models for Alembic to detect
from app.models.user import User
from app.models.note import Note
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.flashcard import Flashcard
from app.models.quiz import Quiz
from app.models.quiz_question import QuizQuestion
from app.models.study_session import StudySession
from app.models.subject_mastery import SubjectMastery
from app.models.study_recommendation import StudyRecommendation
from app.models.study_streak import StudyStreak
from app.models.user_game_profile import UserGameProfile
from app.models.achievement import Achievement
from app.models.user_achievement import UserAchievement
from app.models.xp_transaction import XPTransaction
from app.models.ai_response import AIResponse
from app.models.topic_taxonomy import TopicCategory, Topic, TopicSyncLog
from app.models.subscription import Subscription
