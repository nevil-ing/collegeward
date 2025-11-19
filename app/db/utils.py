from typing import Optional, Type, TypeVar, Generic, List, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError, NoResultFound

from app.db.base import BaseModel
from app.utils.exceptions import DatabaseError, NotFoundError
from app.core.logging import get_logger

logger = get_logger(__name__)

ModelType = TypeVar("ModelType", bound=BaseModel)

class BaseRepository(Generic[ModelType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model
