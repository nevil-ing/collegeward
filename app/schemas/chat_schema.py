from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


class MessageBase(BaseModel):
    role: str = Field(..., description="")
    content: str = Field(..., min_length=1, description="")


class MessageCreate(MessageBase):
    sources: Optional[Dict[str, Any]] = Field(None, description="Citations and source references")


class MessageResponse(MessageBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    sources: Optional[Dict[str, Any]]
    created_at: datetime


class ConversationBase(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    mode: str = Field(..., description="")


class ConversationCreate(ConversationBase):
    pass


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=200)


class ConversationResponse(ConversationBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    messages: Optional[List[MessageResponse]] = None
    created_at: datetime
    updated_at: datetime


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="")
    conversation_id: Optional[UUID] = Field(None, description="")
    mode: str = Field("ai_mode", description="")
    enable_step_by_step: bool = Field(True, description="")
    enable_guided_questions: bool = Field(True, description="")


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message: MessageResponse
    conversation: ConversationResponse


class TutorModeInfo(BaseModel):
    id: str
    name: str
    description: str
    features: List[str]
    response_time: str


class ClinicalScenarioRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    difficulty_level: str = Field("intermediate", description="")


class ClinicalScenarioResponse(BaseModel):
    topic: str
    difficulty_level: str
    scenario: Dict[str, Any]
    context_used: bool


class AssessmentRequest(BaseModel):
    student_response: str = Field(..., min_length=1, max_length=2000)
    expected_concepts: List[str] = Field(..., min_items=1)
    conversation_id: Optional[UUID] = None


class AssessmentResponse(BaseModel):
    student_response: str
    expected_concepts: List[str]
    assessment: Dict[str, Any]