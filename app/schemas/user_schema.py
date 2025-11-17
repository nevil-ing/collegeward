from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from uuid import UUID

class UserBase(BaseModel):
    email: EmailStr
    display_name: Optional[str] = Field(None, max_length=100)
    study_level: Optional[str] = Field(None, max_length=50, description="e.g., 'medical_student', 'nursing_student'")


class UserCreate(UserBase):
    firebase_uid: str = Field(..., max_length=128, description="Firebase authentication UID")


class UserUpdate(BaseModel):
    display_name: Optional[str] = Field(None, max_length=100)
    study_level: Optional[str] = Field(None, max_length=50)
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    firebase_uid: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class UserProfile(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    display_name: Optional[str]
    study_level: Optional[str]
    created_at: datetime


class LoginResponse(BaseModel):
    user: UserProfile
    session: Dict[str, Any]
    message: str


class TokenInfo(BaseModel):
    valid: bool
    uid: str
    email: str
    email_verified: bool
    token_expires_at: Optional[datetime]
    should_refresh: bool


class AuthHealthCheck(BaseModel):
    status: str
    authenticated: bool
    timestamp: Optional[datetime]