from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


class NoteBase(BaseModel):
    filename: str = Field(..., max_length=255)
    file_type: str = Field(..., max_length=10, description="File type: pdf, docx, image")
    subject_tags: Optional[List[str]] = Field(None, description="")


class NoteCreate(NoteBase):
    file_size: int = Field(..., gt=0, description="")
    storage_path: str = Field(..., max_length=500, description="")


class NoteUpdate(BaseModel):
    filename: Optional[str] = Field(None, max_length=255)
    subject_tags: Optional[List[str]] = None
    processing_status: Optional[str] = Field(None, description="Processing status")
    extracted_text: Optional[str] = None


class NoteResponse(NoteBase):

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    file_size: int
    storage_path: str
    processing_status: str
    extracted_text: Optional[str]
    processed_date: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class NoteUpload(BaseModel):
    filename: str = Field(..., max_length=255)
    file_type: str = Field(..., max_length=10)
    file_size: int = Field(..., gt=0, le=50 * 1024 * 1024, description="")
    subject_tags: Optional[List[str]] = None


class NoteProcessingStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    processing_status: str = Field(..., description="Current processing status")
    extracted_text: Optional[str] = None
    processed_date: Optional[datetime] = None