from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


class FlashcardBase(BaseModel):
    question: str = Field(..., min_length=1, description="Flashcard question")
    answer: str = Field(..., min_length=1, description="")
    subject_tags: Optional[List[str]] = Field(None, description="")
    difficulty_level: int = Field(1, ge=1, le=5, description="Difficulty level")


class FlashcardCreate(FlashcardBase):
    created_from: Optional[str] = Field(None, description="S")
    source_reference: Optional[UUID] = Field(None, description="Reference to source note or conversation")


class FlashcardUpdate(BaseModel):
    question: Optional[str] = Field(None, min_length=1)
    answer: Optional[str] = Field(None, min_length=1)
    subject_tags: Optional[List[str]] = None
    difficulty_level: Optional[int] = Field(None, ge=1, le=5)


class FlashcardResponse(FlashcardBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    leitner_box: int
    next_review_date: Optional[datetime]
    times_reviewed: int
    times_correct: int
    created_from: Optional[str]
    source_reference: Optional[UUID]
    created_at: datetime
    updated_at: datetime


class FlashcardReview(BaseModel):
    flashcard_id: UUID
    is_correct: bool = Field(..., description="Whether the user answered correctly")


class FlashcardStats(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total_flashcards: int
    due_for_review: int
    mastered: int
    accuracy_rate: float = Field(..., ge=0.0, le=1.0, description="Accuracy rate (0.0-1.0)")
    average_difficulty: float = Field(..., ge=1.0, le=5.0, description="Average difficulty level")