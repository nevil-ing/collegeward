import asyncio
from multiprocessing.util import get_logger

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import engine, init_db, AsyncSessionLocal
from app.db.base import Base
from app.core.config import settings
from app.db.utils import check_database_connection, get_table_info

#from app.core.logging import get_logger

logger = get_logger(__name__)

async  def create_tables():
    try:
        await init_db()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

async def drop_tables():
    try:
        async with engine.begin() as conn:
            await  conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise


async def reset_tables():
    try:
        await drop_tables()
        await  create_tables()
        logger.info("Database tables reset successfully")
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise

async  def check_database_health():
    from app.db.session import AsyncSessionLocal
    try:
        async with AsyncSessionLocal() as session:
            is_connected = await  check_database_connection(session)
            if not is_connected:
                logger.error("Database connection failed")
                return False

            tables = ["users", "notes", "conversations", "messages", "flashcards", "quizzes", "quiz_questions"]
            for table in tables:
                info = await get_table_info(session, table)
                logger.info(f"{table}: {info['row_count']} rows, status: {info['status']}")
            logger.info("Database health check completed successfully")
            return True
    except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
