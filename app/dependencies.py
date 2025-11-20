from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user, get_current_user_optional
from app.db.session import get_db
from app.models.user import User
from app.services.user_service import user_service
from app.utils.exceptions import AuthenticationError, NotFoundError
from app.core.logging import get_logger

logger = get_logger(__name__)

