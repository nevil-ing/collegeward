
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
from app.ai.clients.medical_sources import MedicalSourcesClient, MedicalSource
from pydantic import BaseModel, ConfigDict


class AIResponse(BaseModel):
    """AI service response"""
    model_config = ConfigDict(protected_namespaces=())

    content: str
    model_used: str
    response_time: float
    sources: List[MedicalSource] = []
    citations: List[str] = []
    mode: str = "ai_mode"


class ConversationMessage(BaseModel):
    """Conversation message format"""
    role: str
    content: str
    timestamp: Optional[datetime] = None