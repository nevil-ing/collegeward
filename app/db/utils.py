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
    async def create(self, db: AsyncSession, **kwargs) -> ModelType:
        """Create a new record"""
        try:
            db_obj = self.model(**kwargs)
            db.add(db_obj)
            await db.flush()
            await db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            await db.rollback()
            logger.error(f"Database integrity error creating {self.model.__name__}: {e}")
            raise DatabaseError(f"Failed to create {self.model.__name__}: {str(e)}")
        except Exception as e:
            await db.rollback()
            logger.error(f"Unexpected error creating {self.model.__name__}: {e}")
            raise DatabaseError(f"Failed to create {self.model.__name__}")

    async def get_by_id(self, db: AsyncSession, id: UUID) -> Optional[ModelType]:
            """Get record by ID"""
            try:
                result = await db.execute(select(self.model).where(self.model.id == id))
                return result.scalar_one_or_none()
            except Exception as e:
                logger.error(f"Error getting {self.model.__name__} by ID {id}: {e}")
                raise DatabaseError(f"Failed to get {self.model.__name__}")


    async def get_by_id_or_404(self, db: AsyncSession, id: UUID) -> ModelType:
            """Get record by ID or raise 404"""
            obj = await self.get_by_id(db, id)
            if not obj:
                raise NotFoundError(f"{self.model.__name__} not found")
            return obj

    async def get_multi(
                self,
                db: AsyncSession,
                skip: int = 0,
                limit: int = 100,
                filters: Optional[Dict[str, Any]] = None
        ) -> List[ModelType]:
            """Get multiple records with pagination and filtering"""
            try:
                query = select(self.model)

                # Apply filters
                if filters:
                    for key, value in filters.items():
                        if hasattr(self.model, key):
                            query = query.where(getattr(self.model, key) == value)

                query = query.offset(skip).limit(limit)
                result = await db.execute(query)
                return result.scalars().all()
            except Exception as e:
                logger.error(f"Error getting multiple {self.model.__name__}: {e}")
                raise DatabaseError(f"Failed to get {self.model.__name__} records")

    async def update(self, db: AsyncSession, id: UUID, **kwargs) -> Optional[ModelType]:
            """Update record by ID"""
            try:
                # Remove None values
                update_data = {k: v for k, v in kwargs.items() if v is not None}

                if not update_data:
                    return await self.get_by_id(db, id)

                await db.execute(
                    update(self.model)
                    .where(self.model.id == id)
                    .values(**update_data)
                )
                await db.flush()
                return await self.get_by_id(db, id)
            except IntegrityError as e:
                await db.rollback()
                logger.error(f"Database integrity error updating {self.model.__name__}: {e}")
                raise DatabaseError(f"Failed to update {self.model.__name__}: {str(e)}")
            except Exception as e:
                await db.rollback()
                logger.error(f"Unexpected error updating {self.model.__name__}: {e}")
                raise DatabaseError(f"Failed to update {self.model.__name__}")

    async def delete(self, db: AsyncSession, id: UUID) -> bool:
            """Delete record by ID"""
            try:
                result = await db.execute(delete(self.model).where(self.model.id == id))
                await db.flush()
                return result.rowcount > 0
            except Exception as e:
                await db.rollback()
                logger.error(f"Error deleting {self.model.__name__} {id}: {e}")
                raise DatabaseError(f"Failed to delete {self.model.__name__}")

    async def count(self, db: AsyncSession, filters: Optional[Dict[str, Any]] = None) -> int:
            """Count records with optional filtering"""
            try:
                query = select(func.count(self.model.id))

                # Apply filters
                if filters:
                    for key, value in filters.items():
                        if hasattr(self.model, key):
                            query = query.where(getattr(self.model, key) == value)

                result = await db.execute(query)
                return result.scalar()
            except Exception as e:
                logger.error(f"Error counting {self.model.__name__}: {e}")
                raise DatabaseError(f"Failed to count {self.model.__name__} records")

async def check_database_connection(db: AsyncSession) -> bool:
        """Check if database connection is working"""
        try:
            await db.execute(select(1))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

async def get_table_info(db: AsyncSession, table_name: str) -> Dict[str, Any]:
        """Get information about a database table"""
        try:
            # Get table row count
            result = await db.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = result.scalar()

            return {
                "table_name": table_name,
                "row_count": row_count,
                "status": "healthy"
            }
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            return {
                "table_name": table_name,
                "row_count": 0,
                "status": "error",
                "error": str(e)
            }