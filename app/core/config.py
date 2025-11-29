import os
from typing import List, Optional, Dict
from pydantic import field_validator, ConfigDict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENVIRONMENT: str = ""
    DEBUG: bool

    # Security
    SECRET_KEY: str
    ALLOWED_HOSTS: str
    CORS_ORIGINS: str

    # Database
    DB_USER: str
    DB_NAME: str
    DB_PASSWORD: str
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int
    DATABASE_MAX_OVERFLOW: int

    # Qdrant Vector Database
    QDRANT_HOST: str
    QDRANT_PORT: int
    QDRANT_API_KEY: Optional[str]
    QDRANT_COLLECTION_NAME: str

    # Firebase Configuration
    FIREBASE_PROJECT_ID: str
    FIREBASE_PRIVATE_KEY_ID: str
    FIREBASE_PRIVATE_KEY: str
    FIREBASE_CLIENT_EMAIL: str
    FIREBASE_CLIENT_ID: str
    FIREBASE_AUTH_URI: str
    FIREBASE_TOKEN_URI: str
    FIREBASE_STORAGE_BUCKET: str = ""

    # Local Storage
    LOCAL_STORAGE_ROOT: str = "./storage"

    # AI Services
    GROQ_API_KEY: str
    GROQ_BASE_URL: str
    GROQ_DEFAULT_MODEL: str
    GROQ_RATE_LIMIT_PER_MINUTE: int

    # External Medical APIs
    PUBMED_API_BASE_URL: str
    PUBMED_EMAIL: str
    MEDLINEPLUS_API_BASE_URL: str

    # AI Response Settings
    AI_MAX_TOKENS: int = 2048
    AI_TEMPERATURE: float = 0.7
    AI_RESPONSE_TIMEOUT: int = 30

    # File Upload Settings
    MAX_FILE_SIZE: int
    ALLOWED_FILE_TYPES: List[str] = ["pdf", "docx", "png", "jpg", "jpeg"]

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int
    RATE_LIMIT_PER_HOUR: int

    # Input Validation
    MAX_TEXT_LENGTH: int = 10000
    MAX_SUBJECT_TAGS: int = 10
    MAX_SUBJECT_TAG_LENGTH: int = 50

    # Security
    ENABLE_SECURITY_HEADERS: bool = True
    ENABLE_RATE_LIMITING: bool = True
    ENABLE_REQUEST_LOGGING: bool = True
    ENABLE_INPUT_VALIDATION: bool = True
    ENABLE_SECURITY_MONITORING: bool = True

    # Error Handling
    ENABLE_DETAILED_ERRORS: bool = False
    ENABLE_ERROR_TRACKING: bool = True
    ERROR_NOTIFICATION_THRESHOLD: int = 10  # Errors per minute before notification

    # Request Validation
    MAX_REQUEST_SIZE: int = 100 * 1024 * 1024  # 100MB
    MAX_QUERY_STRING_LENGTH: int = 2048
    MAX_HEADER_SIZE: int = 8192
    ALLOWED_CONTENT_TYPES: List[str] = [
        "application/json",
        "multipart/form-data",
        "application/x-www-form-urlencoded",
        "text/plain"
    ]

    # Rate Limiting Tiers
    RATE_LIMIT_TIERS: Dict[str, Dict[str, int]] = {
        "free": {"minute": 30, "hour": 500, "day": 2000},
        "standard": {"minute": 60, "hour": 1000, "day": 5000},
        "premium": {"minute": 120, "hour": 2000, "day": 10000},
        "enterprise": {"minute": 300, "hour": 5000, "day": 25000}
    }

    # Circuit Breaker Settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 60
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = 3

    # Monitoring
    ENABLE_HEALTH_CHECKS: bool = True
    HEALTH_CHECK_TIMEOUT: int = 30
    ENABLE_PERFORMANCE_MONITORING: bool = True
    SLOW_REQUEST_THRESHOLD: float = 1.0
    VERY_SLOW_REQUEST_THRESHOLD: float = 5.0

    # Graceful Degradation
    ENABLE_GRACEFUL_DEGRADATION: bool = True
    FALLBACK_CACHE_TTL: int = 3600
    SERVICE_RECOVERY_TIME: int = 300

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    model_config = ConfigDict(
        extra="ignore"
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def assemble_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v


# Global settings instance
settings = Settings()
