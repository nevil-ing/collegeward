import asyncio
from multiprocessing.util import get_logger

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import engine, init_db
from app.db.base import Base
from app.core.config import settings
#from app.core.logging import get_logger

logger = get_logger(__name__)

async  def create_tables():
    try:
        await init_db()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


