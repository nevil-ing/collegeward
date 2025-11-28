import functools
import time
from typing import Callable, Any, Dict, List, Optional
from fastapi import Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.validation import (
    InputValidator,
    InputSanitizer,
    RequestValidator,
    sanitize_and_validate_input
)
from app.utils.exceptions import (
    InputValidationError,
    SecurityValidationError,
    RateLimitError
)
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


def validate_request(
        text_fields: Optional[List[str]] = None,
        html_fields: Optional[List[str]] = None,
        required_fields: Optional[List[str]] = None,
        max_request_size: Optional[int] = None,
        allowed_content_types: Optional[List[str]] = None
):
    """
    Decorator for comprehensive request validation

    Args:
        text_fields: Fields to sanitize as plain text
        html_fields: Fields to sanitize as HTML
        required_fields: Fields that are required
        max_request_size: Maximum request size in bytes
        allowed_content_types: Allowed content types
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs if present
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                for key, value in kwargs.items():
                    if isinstance(value, Request):
                        request = value
                        break

            if request:
                # Validate request size
                if max_request_size:
                    content_length = int(request.headers.get("content-length", 0))
                    RequestValidator.validate_request_size(content_length, max_request_size)

                # Validate content type
                if allowed_content_types:
                    content_type = request.headers.get("content-type", "")
                    # Extract base content type (ignore charset, boundary, etc.)
                    base_content_type = content_type.split(';')[0].strip()
                    if base_content_type and base_content_type not in allowed_content_types:
                        RequestValidator.validate_content_type(base_content_type, allowed_content_types)

            # If we have form data or JSON data to validate
            if hasattr(request, 'form') or hasattr(request, 'json'):
                try:
                    # This would need to be implemented based on the specific request type
                    # For now, we'll skip automatic data extraction and let the endpoint handle it
                    pass
                except Exception as e:
                    logger.warning(f"Request validation warning: {e}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def monitor_performance(
        log_slow_requests: bool = True,
        slow_request_threshold: float = 1.0,
        track_metrics: bool = True
):
    """
    Decorator for performance monitoring

    Args:
        log_slow_requests: Whether to log slow requests
        slow_request_threshold: Threshold in seconds for slow requests
        track_metrics: Whether to track performance metrics
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Log slow requests
                if log_slow_requests and execution_time > slow_request_threshold:
                    logger.warning(
                        f"Slow request detected: {func.__name__} took {execution_time:.3f}s"
                    )

                # Track metrics (could be extended to send to monitoring system)
                if track_metrics:
                    logger.debug(f"Performance: {func.__name__} executed in {execution_time:.3f}s")

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Error in {func.__name__} after {execution_time:.3f}s: {e}"
                )
                raise

        return wrapper

    return decorator


def handle_database_errors(
        rollback_on_error: bool = True,
        log_errors: bool = True
):
    """
    Decorator for database error handling

    Args:
        rollback_on_error: Whether to rollback transaction on error
        log_errors: Whether to log database errors
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Find database session in arguments
            db_session = None
            for arg in args:
                if isinstance(arg, AsyncSession):
                    db_session = arg
                    break

            if not db_session:
                for key, value in kwargs.items():
                    if isinstance(value, AsyncSession):
                        db_session = value
                        break

            try:
                return await func(*args, **kwargs)

            except Exception as e:
                if log_errors:
                    logger.error(f"Database error in {func.__name__}: {e}")

                # Rollback transaction if database session is available
                if rollback_on_error and db_session:
                    try:
                        await db_session.rollback()
                        logger.debug("Database transaction rolled back")
                    except Exception as rollback_error:
                        logger.error(f"Failed to rollback transaction: {rollback_error}")

                raise

        return wrapper

    return decorator


def validate_user_input(
        sanitize_text: bool = True,
        check_security: bool = True,
        validate_length: bool = True,
        max_length: int = 1000
):
    """
    Decorator for user input validation and sanitization

    Args:
        sanitize_text: Whether to sanitize text inputs
        check_security: Whether to check for security violations
        validate_length: Whether to validate text length
        max_length: Maximum text length
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # This decorator would need to be customized based on the specific
            # input parameters of each endpoint. For now, it serves as a template.

            try:
                return await func(*args, **kwargs)
            except (InputValidationError, SecurityValidationError) as e:
                logger.warning(f"Input validation error in {func.__name__}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


def circuit_breaker_protection(
        service_name: str,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[int] = None,
        fallback_func: Optional[Callable] = None
):
    """
    Decorator for circuit breaker protection

    Args:
        service_name: Name of the service to protect
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        fallback_func: Fallback function to call when circuit is open
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            from app.core.circuit_breaker import service_registry, CircuitBreakerConfig

            # Create config with custom values or defaults
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold or settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                recovery_timeout=recovery_timeout or settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
                success_threshold=settings.CIRCUIT_BREAKER_SUCCESS_THRESHOLD
            )

            try:
                return await service_registry.call_service(
                    service_name,
                    func,
                    *args,
                    config=config,
                    fallback=fallback_func,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Circuit breaker error for {service_name}: {e}")
                raise

        return wrapper

    return decorator


def comprehensive_api_protection(
        validate_input: bool = True,
        monitor_performance: bool = True,
        handle_db_errors: bool = True,
        circuit_breaker: Optional[str] = None,
        max_request_size: Optional[int] = None,
        required_fields: Optional[List[str]] = None,
        text_fields: Optional[List[str]] = None
):
    """
    Comprehensive API protection decorator combining multiple protections

    Args:
        validate_input: Enable input validation
        monitor_performance: Enable performance monitoring
        handle_db_errors: Enable database error handling
        circuit_breaker: Service name for circuit breaker protection
        max_request_size: Maximum request size
        required_fields: Required fields for validation
        text_fields: Text fields to sanitize
    """

    def decorator(func: Callable) -> Callable:
        # Apply decorators in reverse order (innermost first)
        protected_func = func

        if handle_db_errors:
            protected_func = handle_database_errors()(protected_func)

        if monitor_performance:
            protected_func = monitor_performance()(protected_func)

        if validate_input:
            protected_func = validate_request(
                text_fields=text_fields,
                required_fields=required_fields,
                max_request_size=max_request_size
            )(protected_func)

        if circuit_breaker:
            protected_func = circuit_breaker_protection(circuit_breaker)(protected_func)

        return protected_func

    return decorator