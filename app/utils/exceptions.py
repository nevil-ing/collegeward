from datetime import datetime, UTC
from datetime import timezone
from typing import Optional, Dict, Any, List, Union
from fastapi import HTTPException
import traceback
import uuid

class CustomException(Exception):

    def __init__(self,
                 message: str,
                 status_code: int = 500,
                 error_code: str = "INTERNAL_SERVER_ERROR",
                 details: Optional[Dict[str, Any]] = None,
                 user_message: Optional[str] = None,
                 correlation_id: Optional[str] = None,
                 retry_after: Optional[int] = None,
                 help_url: Optional[str] = None,
                 ):
               self.message = message
               self.status_code = status_code
               self.error_code = error_code
               self.details = details or {}
               self.timestamp = datetime.now(UTC)
               self.user_message = user_message or message
               self.correlation_id = correlation_id or str(uuid.uuid4())
               self.retry_after = retry_after
               self.help_url = help_url
               self.traceback = traceback.format_exc() if status_code >= 500 else None
               super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        error_dict = {
            "code": self.error_code,
            "message": self.user_message,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "type": "error"
        }

        if self.retry_after:
            error_dict["retry_after"] = self.retry_after

        if self.details:
            error_dict["details"] = self.details
        if self.help_url:
            error_dict["help_url"] = self.help_url
        if self.traceback and self.status_code >= 500:
            error_dict["traceback"] = self.traceback.split('\n')
        return {"error": error_dict}

    def get_response_headers(self) -> Dict[str, str]:
        """Get additional response headers for this exception"""
        headers = {"X-Correlation-ID": self.correlation_id}

        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)

        if self.status_code == 429:  # Rate limit
            headers["X-RateLimit-Reset"] = str(int(self.timestamp.timestamp()) + (self.retry_after or 60))

        return headers


class ValidationError(CustomException):
    """Validation error exception"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details=details
        )


class AuthenticationError(CustomException):
    """Authentication error exception"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(CustomException):
    """Authorization error exception"""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR"
        )


class NotFoundError(CustomException):
    """Resource not found exception"""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND"
        )


class FileProcessingError(CustomException):
    """File processing error exception"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=422,
            error_code="FILE_PROCESSING_ERROR",
            details=details
        )


class ExternalServiceError(CustomException):
    """External service error exception"""

    def __init__(self, message: str, service_name: str):
        super().__init__(
            message=message,
            status_code=503,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service_name}
        )


class RateLimitError(CustomException):
    """Rate limit exceeded exception"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_ERROR"
        )


class DatabaseError(CustomException):
    """Database operation error exception"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="DATABASE_ERROR",
            details=details
        )


class ProcessingError(CustomException):
    """General processing error exception"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=422,
            error_code="PROCESSING_ERROR",
            details=details
        )


class StorageError(CustomException):
    """Storage operation error exception"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="STORAGE_ERROR",
            details=details
        )


class AIServiceError(CustomException):
    """AI service error exception"""

    def __init__(self, message: str, service_name: str = "AI Service", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=503,
            error_code="AI_SERVICE_ERROR",
            details={**(details or {}), "service": service_name}
        )


class InputValidationError(ValidationError):
    """Input validation error with field-specific details"""

    def __init__(self, message: str, field: str, value: Any = None, allowed_values: Optional[List[str]] = None):
        details = {"field": field}
        if value is not None:
            details["provided_value"] = str(value)
        if allowed_values:
            details["allowed_values"] = allowed_values

        super().__init__(
            message=message,
            details=details
        )


class ServiceDegradationError(CustomException):
    """Service degradation error for graceful fallbacks"""

    def __init__(self, message: str, service_name: str, fallback_used: bool = False):
        super().__init__(
            message=message,
            status_code=206,  # Partial Content - service degraded but functional
            error_code="SERVICE_DEGRADED",
            user_message=f"Service temporarily degraded. {'Using fallback.' if fallback_used else 'Limited functionality available.'}",
            details={
                "service": service_name,
                "fallback_used": fallback_used
            }
        )


class CircuitBreakerError(CustomException):
    """Circuit breaker error for external service failures"""

    def __init__(self, service_name: str, retry_after: Optional[int] = None):
        details = {"service": service_name}
        if retry_after:
            details["retry_after_seconds"] = retry_after

        super().__init__(
            message=f"Service {service_name} is temporarily unavailable",
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
            user_message="This feature is temporarily unavailable. Please try again later.",
            details=details
        )


class SecurityValidationError(CustomException):
    """Security validation error for malicious input detection"""

    def __init__(self, message: str, violation_type: str):
        super().__init__(
            message=message,
            status_code=400,
            error_code="SECURITY_VIOLATION",
            user_message="Invalid input detected",
            details={"violation_type": violation_type},
            help_url="/docs/security-guidelines"
        )


class RequestTimeoutError(CustomException):
    """Request timeout error"""

    def __init__(self, message: str = "Request timeout", timeout_seconds: int = 30):
        super().__init__(
            message=message,
            status_code=408,
            error_code="REQUEST_TIMEOUT",
            user_message="Request took too long to process",
            details={"timeout_seconds": timeout_seconds},
            retry_after=5
        )


class ServiceUnavailableError(CustomException):
    """Service unavailable error with retry information"""

    def __init__(self, message: str, service_name: str, retry_after: int = 60):
        super().__init__(
            message=message,
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
            user_message=f"The {service_name} service is temporarily unavailable",
            details={"service": service_name},
            retry_after=retry_after
        )


class PayloadTooLargeError(CustomException):
    """Payload too large error"""

    def __init__(self, message: str, max_size: int, actual_size: int):
        super().__init__(
            message=message,
            status_code=413,
            error_code="PAYLOAD_TOO_LARGE",
            user_message=f"Request size exceeds maximum allowed size",
            details={
                "max_size_bytes": max_size,
                "actual_size_bytes": actual_size,
                "max_size_mb": round(max_size / (1024 * 1024), 2)
            }
        )


class UnsupportedMediaTypeError(CustomException):
    """Unsupported media type error"""

    def __init__(self, message: str, provided_type: str, allowed_types: List[str]):
        super().__init__(
            message=message,
            status_code=415,
            error_code="UNSUPPORTED_MEDIA_TYPE",
            user_message="File type not supported",
            details={
                "provided_type": provided_type,
                "allowed_types": allowed_types
            }
        )


class ConflictError(CustomException):
    """Resource conflict error"""

    def __init__(self, message: str, resource_type: str = "resource"):
        super().__init__(
            message=message,
            status_code=409,
            error_code="CONFLICT",
            user_message=f"The {resource_type} already exists or conflicts with existing data",
            details={"resource_type": resource_type}
        )


class UnprocessableEntityError(CustomException):
    """Unprocessable entity error for semantic validation failures"""

    def __init__(self, message: str, validation_errors: Optional[List[Dict[str, Any]]] = None):
        super().__init__(
            message=message,
            status_code=422,
            error_code="UNPROCESSABLE_ENTITY",
            user_message="The request contains invalid data",
            details={"validation_errors": validation_errors or []}
        )