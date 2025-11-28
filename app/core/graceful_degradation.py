import asyncio
import time
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from app.utils.exceptions import (
    ServiceDegradationError,
    ExternalServiceError,
    CircuitBreakerError
)
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


@dataclass
class FallbackResponse:
    """Fallback response structure"""
    success: bool
    data: Any
    message: str
    source: str  # "cache", "fallback_service", "default_response"
    degraded: bool = True
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ServiceHealth:
    """Service health tracking"""
    service_name: str
    status: ServiceStatus
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    failure_count: int
    success_count: int
    response_time_ms: Optional[float]
    error_message: Optional[str]
    fallback_available: bool


class FallbackCache:
    """Simple in-memory cache for fallback responses"""

    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key not in self.cache:
            return None

        entry = self.cache[key]
        if time.time() > entry['expires_at']:
            del self.cache[key]
            return None

        return entry['data']

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with TTL"""
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            'data': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }

    def clear(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()

    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class GracefulDegradationManager:
    """Manages graceful degradation for external services"""

    def __init__(self):
        self.service_health: Dict[str, ServiceHealth] = {}
        self.fallback_cache = FallbackCache()
        self.fallback_handlers: Dict[str, Callable] = {}
        self.default_responses: Dict[str, Any] = {}

        # Configuration
        self.max_failure_count = 5
        self.degradation_threshold = 3
        self.recovery_time = timedelta(minutes=5)
        self.cache_ttl = 3600  # 1 hour

    def register_service(
            self,
            service_name: str,
            fallback_handler: Optional[Callable] = None,
            default_response: Optional[Any] = None
    ) -> None:
        """Register a service for degradation management"""

        self.service_health[service_name] = ServiceHealth(
            service_name=service_name,
            status=ServiceStatus.HEALTHY,
            last_success=None,
            last_failure=None,
            failure_count=0,
            success_count=0,
            response_time_ms=None,
            error_message=None,
            fallback_available=fallback_handler is not None or default_response is not None
        )

        if fallback_handler:
            self.fallback_handlers[service_name] = fallback_handler

        if default_response:
            self.default_responses[service_name] = default_response

        logger.info(f"Registered service '{service_name}' for degradation management")

    def record_success(self, service_name: str, response_time_ms: float) -> None:
        """Record successful service call"""

        if service_name not in self.service_health:
            self.register_service(service_name)

        health = self.service_health[service_name]
        health.last_success = datetime.utcnow()
        health.success_count += 1
        health.response_time_ms = response_time_ms
        health.error_message = None

        # Reset failure count on success
        if health.failure_count > 0:
            health.failure_count = max(0, health.failure_count - 1)

        # Update status based on recent performance
        if health.failure_count == 0:
            health.status = ServiceStatus.HEALTHY
        elif health.failure_count < self.degradation_threshold:
            health.status = ServiceStatus.DEGRADED

    def record_failure(self, service_name: str, error_message: str) -> None:
        """Record failed service call"""

        if service_name not in self.service_health:
            self.register_service(service_name)

        health = self.service_health[service_name]
        health.last_failure = datetime.utcnow()
        health.failure_count += 1
        health.error_message = error_message

        # Update status based on failure count
        if health.failure_count >= self.max_failure_count:
            health.status = ServiceStatus.UNAVAILABLE
        elif health.failure_count >= self.degradation_threshold:
            health.status = ServiceStatus.DEGRADED

        logger.warning(
            f"Service '{service_name}' failure recorded. "
            f"Count: {health.failure_count}, Status: {health.status.value}"
        )

    def get_service_status(self, service_name: str) -> ServiceStatus:
        """Get current service status"""

        if service_name not in self.service_health:
            return ServiceStatus.HEALTHY

        health = self.service_health[service_name]

        # Check if service should recover from unavailable status
        if (health.status == ServiceStatus.UNAVAILABLE and
                health.last_failure and
                datetime.utcnow() - health.last_failure > self.recovery_time):
            # Reset to degraded status for recovery attempt
            health.status = ServiceStatus.DEGRADED
            health.failure_count = self.degradation_threshold
            logger.info(f"Service '{service_name}' attempting recovery from unavailable status")

        return health.status

    async def call_with_fallback(
            self,
            service_name: str,
            primary_func: Callable,
            *args,
            cache_key: Optional[str] = None,
            cache_ttl: Optional[int] = None,
            **kwargs
    ) -> FallbackResponse:
        """
        Call service with automatic fallback on failure

        Args:
            service_name: Name of the service
            primary_func: Primary function to call
            cache_key: Key for caching successful responses
            cache_ttl: Cache TTL in seconds
            *args, **kwargs: Arguments for the primary function

        Returns:
            FallbackResponse with result or fallback data
        """

        status = self.get_service_status(service_name)

        # If service is unavailable, go straight to fallback
        if status == ServiceStatus.UNAVAILABLE:
            return await self._get_fallback_response(service_name, cache_key)

        # Try primary service
        start_time = time.time()
        try:
            result = await primary_func(*args, **kwargs)
            response_time = (time.time() - start_time) * 1000

            # Record success
            self.record_success(service_name, response_time)

            # Cache successful response
            if cache_key:
                self.fallback_cache.set(cache_key, result, cache_ttl or self.cache_ttl)

            return FallbackResponse(
                success=True,
                data=result,
                message="Service call successful",
                source="primary_service",
                degraded=False
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            # Record failure
            self.record_failure(service_name, str(e))

            logger.warning(
                f"Primary service '{service_name}' failed: {e}",
                extra={
                    "service_name": service_name,
                    "response_time_ms": response_time,
                    "error": str(e)
                }
            )

            # Try fallback
            return await self._get_fallback_response(service_name, cache_key)

    async def _get_fallback_response(
            self,
            service_name: str,
            cache_key: Optional[str] = None
    ) -> FallbackResponse:
        """Get fallback response for failed service"""

        # Try cached response first
        if cache_key:
            cached_data = self.fallback_cache.get(cache_key)
            if cached_data is not None:
                return FallbackResponse(
                    success=True,
                    data=cached_data,
                    message="Using cached response due to service unavailability",
                    source="cache"
                )

        # Try custom fallback handler
        if service_name in self.fallback_handlers:
            try:
                fallback_data = await self.fallback_handlers[service_name]()
                return FallbackResponse(
                    success=True,
                    data=fallback_data,
                    message="Using fallback service response",
                    source="fallback_service"
                )
            except Exception as e:
                logger.error(f"Fallback handler for '{service_name}' failed: {e}")

        # Use default response
        if service_name in self.default_responses:
            return FallbackResponse(
                success=True,
                data=self.default_responses[service_name],
                message="Using default response due to service unavailability",
                source="default_response"
            )

        # No fallback available
        health = self.service_health.get(service_name)
        error_msg = health.error_message if health else "Service unavailable"

        return FallbackResponse(
            success=False,
            data=None,
            message=f"Service '{service_name}' unavailable and no fallback configured",
            source="none",
            degraded=True
        )

    def get_all_service_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all services"""

        return {
            name: {
                "status": health.status.value,
                "last_success": health.last_success.isoformat() if health.last_success else None,
                "last_failure": health.last_failure.isoformat() if health.last_failure else None,
                "failure_count": health.failure_count,
                "success_count": health.success_count,
                "response_time_ms": health.response_time_ms,
                "error_message": health.error_message,
                "fallback_available": health.fallback_available
            }
            for name, health in self.service_health.items()
        }

    def reset_service_health(self, service_name: str) -> None:
        """Reset service health status"""

        if service_name in self.service_health:
            health = self.service_health[service_name]
            health.status = ServiceStatus.HEALTHY
            health.failure_count = 0
            health.error_message = None
            logger.info(f"Reset health status for service '{service_name}'")


# Global degradation manager instance
degradation_manager = GracefulDegradationManager()


# Default fallback handlers for common services
async def groq_fallback_handler() -> Dict[str, Any]:
    """Fallback handler for Groq AI service"""
    return {
        "choices": [{
            "message": {
                "content": "I apologize, but the AI service is temporarily unavailable. Please try again later or contact support if the issue persists.",
                "role": "assistant"
            }
        }],
        "usage": {"total_tokens": 0},
        "model": "fallback",
        "degraded": True
    }


async def pubmed_fallback_handler() -> Dict[str, Any]:
    """Fallback handler for PubMed service"""
    return {
        "articles": [],
        "total_count": 0,
        "message": "Medical research database temporarily unavailable. Using cached information.",
        "degraded": True
    }


async def medlineplus_fallback_handler() -> Dict[str, Any]:
    """Fallback handler for MedlinePlus service"""
    return {
        "results": [],
        "total": 0,
        "message": "Medical information service temporarily unavailable.",
        "degraded": True
    }


# Register default services
def setup_default_services():
    """Setup default services with fallback handlers"""

    degradation_manager.register_service(
        "groq_ai",
        fallback_handler=groq_fallback_handler,
        default_response={
            "error": "AI service unavailable",
            "message": "Please try again later"
        }
    )

    degradation_manager.register_service(
        "pubmed",
        fallback_handler=pubmed_fallback_handler,
        default_response={
            "articles": [],
            "message": "Research database unavailable"
        }
    )

    degradation_manager.register_service(
        "medlineplus",
        fallback_handler=medlineplus_fallback_handler,
        default_response={
            "results": [],
            "message": "Medical information unavailable"
        }
    )

    degradation_manager.register_service(
        "firebase_storage",
        default_response={
            "error": "File storage temporarily unavailable",
            "message": "Please try uploading again later"
        }
    )

    degradation_manager.register_service(
        "qdrant",
        default_response={
            "results": [],
            "message": "Search service temporarily unavailable"
        }
    )


# Initialize default services
setup_default_services()