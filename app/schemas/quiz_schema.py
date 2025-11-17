from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


class QuizQuestionBase(BaseModel):
    question_text: str = Field(..., min_length=1, description="Question text")
    options: List[str] = Field(..., min_length=2, max_length=6, description="Answer options")
    correct_answer: int = Field(..., ge=0, description="Index of correct option")
    explanation: Optional[str] = Field(None, description="Explanation for the answer")


class QuizQuestionCreate(QuizQuestionBase):
    question_order: int = Field(..., ge=1, description="Question order in quiz")


class QuizQuestionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    quiz_id: UUID
    question_text: str = Field(..., min_length=1, description="Question text")
    options: List[str] = Field(..., min_length=2, max_length=6, description="Answer options")
    correct_answer: Optional[int] = Field(None, ge=0, description="Index of correct option (None if not revealed)")
    explanation: Optional[str] = Field(None, description="Explanation for the answer (None if not revealed)")
    user_answer: Optional[int]
    is_correct: Optional[bool]
    question_order: int


class QuizBase(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    subject_tags: Optional[List[str]] = Field(None, description="Subject classifications")


class QuizCreate(QuizBase):
    questions: List[QuizQuestionCreate] = Field(..., min_length=1, max_length=50)

    @property
    def total_questions(self) -> int:
        return len(self.questions)


class QuizResponse(QuizBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    total_questions: int
    score: Optional[int]
    percentage: Optional[Decimal]
    time_taken: Optional[int]  # seconds
    completed_at: Optional[datetime]
    questions: Optional[List[QuizQuestionResponse]] = None
    created_at: datetime
    updated_at: datetime


class QuizSubmission(BaseModel):

    quiz_id: UUID
    answers: Dict[str, int] = Field(..., description="")
    time_taken: Optional[int] = Field(None, ge=0, description="Time taken in seconds")

    def get_answers_as_int_keys(self) -> Dict[int, int]:

        return {int(k): v for k, v in self.answers.items()}


class QuizResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    quiz_id: UUID
    score: int = Field(..., ge=0)
    percentage: Decimal = Field(..., ge=0, le=100)
    time_taken: Optional[int]
    correct_answers: int
    total_questions: int
    questions_with_answers: List[QuizQuestionResponse]
    completed_at: datetime


class QuizStats(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total_quizzes: int
    completed_quizzes: int
    average_score: float = Field(..., ge=0.0, le=100.0)
    best_score: float = Field(..., ge=0.0, le=100.0)
    total_time_spent: int = Field(..., ge=0, description="Total time in seconds")
    weak_areas: List[str] = Field(..., description="Subject areas needing improvement")