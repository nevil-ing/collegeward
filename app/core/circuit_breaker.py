import asyncio
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.utils.exceptions import CircuitBreakerError, ServiceDegradationError
from app.core.logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Service unavailable
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: int = 30  # Request timeout in seconds


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """Circuit breaker implementation for external services"""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            self.stats.total_requests += 1

            # Check if circuit is open
            if self.stats.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.stats.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN state")
                else:
                    self._log_circuit_open()
                    raise CircuitBreakerError(
                        self.name,
                        retry_after=self.config.recovery_timeout
                    )

        # Execute the function
        try:
            # Apply timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )

            await self._on_success()
            return result

        except asyncio.TimeoutError:
            await self._on_failure("timeout")
            raise CircuitBreakerError(self.name)
        except Exception as e:
            await self._on_failure(str(e))
            raise

    async def _on_success(self):
        """Handle successful request"""
        async with self._lock:
            self.stats.success_count += 1
            self.stats.total_successes += 1
            self.stats.last_success_time = datetime.utcnow()

            if self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.success_count >= self.config.success_threshold:
                    self.stats.state = CircuitState.CLOSED
                    self.stats.failure_count = 0
                    self.stats.success_count = 0
                    logger.info(f"Circuit breaker {self.name} closed after recovery")
            elif self.stats.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.stats.failure_count = 0

    async def _on_failure(self, error: str):
        """Handle failed request"""
        async with self._lock:
            self.stats.failure_count += 1
            self.stats.total_failures += 1
            self.stats.last_failure_time = datetime.utcnow()

            logger.warning(f"Circuit breaker {self.name} failure: {error}")

            if self.stats.state == CircuitState.HALF_OPEN:
                # Go back to open on any failure in half-open state
                self.stats.state = CircuitState.OPEN
                self.stats.success_count = 0
                logger.warning(f"Circuit breaker {self.name} opened from HALF_OPEN")
            elif self.stats.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self.stats.failure_count >= self.config.failure_threshold:
                    self.stats.state = CircuitState.OPEN
                    logger.error(f"Circuit breaker {self.name} opened due to failures")

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if not self.stats.last_failure_time:
            return True

        time_since_failure = datetime.utcnow() - self.stats.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout

    def _log_circuit_open(self):
        """Log circuit open state"""
        time_until_retry = 0
        if self.stats.last_failure_time:
            elapsed = datetime.utcnow() - self.stats.last_failure_time
            time_until_retry = max(0, self.config.recovery_timeout - elapsed.total_seconds())

        logger.warning(
            f"Circuit breaker {self.name} is OPEN. "
            f"Retry in {time_until_retry:.0f} seconds"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_requests": self.stats.total_requests,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "success_rate": (
                self.stats.total_successes / self.stats.total_requests
                if self.stats.total_requests > 0 else 0
            ),
            "last_failure_time": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            "last_success_time": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None
        }


class ServiceRegistry:
    """Registry for managing circuit breakers for different services"""

    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()

    def get_circuit_breaker(self, service_name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self._circuit_breakers:
            circuit_config = config or self._default_config
            self._circuit_breakers[service_name] = CircuitBreaker(service_name, circuit_config)

        return self._circuit_breakers[service_name]

    async def call_service(
            self,
            service_name: str,
            func: Callable[..., Awaitable[Any]],
            *args,
            fallback: Optional[Callable[..., Awaitable[Any]]] = None,
            config: Optional[CircuitBreakerConfig] = None,
            **kwargs
    ) -> Any:
        """Call service with circuit breaker protection and optional fallback"""
        circuit_breaker = self.get_circuit_breaker(service_name, config)

        try:
            return await circuit_breaker.call(func, *args, **kwargs)
        except CircuitBreakerError:
            if fallback:
                logger.info(f"Using fallback for service {service_name}")
                try:
                    result = await fallback(*args, **kwargs)
                    # Raise degradation error to inform client
                    raise ServiceDegradationError(
                        f"Service {service_name} degraded, using fallback",
                        service_name,
                        fallback_used=True
                    )
                except Exception as e:
                    logger.error(f"Fallback failed for service {service_name}: {e}")
                    raise CircuitBreakerError(service_name)
            else:
                raise

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {
            name: cb.get_stats()
            for name, cb in self._circuit_breakers.items()
        }

    def reset_circuit_breaker(self, service_name: str) -> bool:
        """Manually reset a circuit breaker"""
        if service_name in self._circuit_breakers:
            cb = self._circuit_breakers[service_name]
            cb.stats.state = CircuitState.CLOSED
            cb.stats.failure_count = 0
            cb.stats.success_count = 0
            logger.info(f"Circuit breaker {service_name} manually reset")
            return True
        return False


# Global service registry instance
service_registry = ServiceRegistry()


# Decorator for easy circuit breaker usage
def circuit_breaker(
        service_name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable] = None
):
    """Decorator to add circuit breaker protection to async functions"""

    def decorator(func: Callable[..., Awaitable[Any]]):
        async def wrapper(*args, **kwargs):
            return await service_registry.call_service(
                service_name, func, *args, fallback=fallback, config=config, **kwargs
            )

        return wrapper

    return decorator