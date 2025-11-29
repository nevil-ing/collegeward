import datetime
from functools import lru_cache

import requests
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.security import RateLimiter, verify_firebase_token
from app.db.base import Base
from app.core.logging import get_logger
import logging
from app.db.session import engine
from app.middleware.auth_middleware import SecurityHeadersMiddleware, AuthenticationLoggingMiddleware
from app.middleware.monitoring_middleware import ResponseEnhancementMiddleware, ComprehensiveMonitoringMiddleware, \
    RequestValidationMiddleware
from app.utils.error_handler import setup_exception_handlers
from app.core.rate_limiter import RateLimitMiddleware
from app.api.v1 import auth

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting CollegeWard API")

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        logger.info("API will continue without database functionality")

    yield
    logger.info("Shutting down CollegeWard API")


@lru_cache
def get_settings():
    return settings()
app = FastAPI(
    lifespan=lifespan,
    title="Collegeward API",
    description="AI-powered interactive study companion for medical students",
    version="0.1.0",
    docs_url="/",
    redoc_url="/redoc",
)

setup_exception_handlers(app)

#register middlewares
app.add_middleware(ResponseEnhancementMiddleware)
app.add_middleware(ComprehensiveMonitoringMiddleware)
app.add_middleware(SecurityHeadersMiddleware,
                   verify_token_fn=verify_firebase_token)
app.add_middleware(RateLimitMiddleware)
#app.add_middleware(RequestValidationMiddleware)
app.add_middleware(AuthenticationLoggingMiddleware)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#api routes
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])

storage_path = Path("./storage")
storage_path.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(storage_path)), name="storage")
logger.info(f"Static file server mounted at /storage (directory: {storage_path.absolute()})")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
"""
@app.get("/health/detailed")
async def health_check_detailed():
    from app.core.monitoring import health_checker

    try:
        results = await health_checker.run_all_checks()
        overall_status = health_checker.get_overall_status(results)

        return {
            "status": overall_status,
            "service": "StudyBlitzAI API",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {name: asdict(result) for name, result in results.items()}
        }
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "StudyBlitzAI API",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Health check system failure"
            }
        )"""


