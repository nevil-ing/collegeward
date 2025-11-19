import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.utils.exceptions import CustomException
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class ErrorResponseFormatter:
    """error responses with a consistent structure"""

    @staticmethod
    def format_error_response(
            error_code: str,
            message: str,
            status_code: int = 500,
            details: Optional[Dict[str, Any]] = None,
            correlation_id: Optional[str] = None,
            retry_after: Optional[int] = None,
            help_url: Optional[str] = None,
            request_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """a standardized error response"""

        error_response = {
            "error": {
                "code": error_code,
                "message": message,
                "status": status_code,
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": correlation_id or str(uuid.uuid4()),
                "type": "error"
            }
        }

        if details:
            error_response["error"]["details"] = details

        if retry_after:
            error_response["error"]["retry_after"] = retry_after

        if help_url:
            error_response["error"]["help_url"] = help_url

        if request_path:
            error_response["error"]["path"] = request_path

        # Add debug information in development
        if settings.ENVIRONMENT == "development" and status_code >= 500:
            error_response["error"]["debug"] = {
                "traceback": traceback.format_exc().split('\n'),
                "environment": settings.ENVIRONMENT
            }

        return error_response

    @staticmethod
    def format_validation_error_response(
            validation_errors: List[Dict[str, Any]],
            correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """a validation error response with field specific detail"""

        formatted_errors = []
        for error in validation_errors:
            formatted_error = {
                "field": error.get("loc", ["unknown"])[-1] if error.get("loc") else "unknown",
                "message": error.get("msg", "Validation failed"),
                "type": error.get("type", "validation_error"),
                "input": error.get("input")
            }

            # Add context for specific validation types
            if error.get("type") == "value_error":
                formatted_error["constraint"] = "Invalid value"
            elif error.get("type") == "type_error":
                formatted_error["constraint"] = "Invalid type"
            elif error.get("type") == "missing":
                formatted_error["constraint"] = "Required field"

            formatted_errors.append(formatted_error)

        return ErrorResponseFormatter.format_error_response(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            status_code=422,
            details={
                "validation_errors": formatted_errors,
                "error_count": len(formatted_errors)
            },
            correlation_id=correlation_id,
            help_url="/docs/api-validation"
        )


class GlobalExceptionHandler:
    """Global exception handler for all application errors"""

    def __init__(self):
        self.formatter = ErrorResponseFormatter()

    async def handle_custom_exception(
            self,
            request: Request,
            exc: CustomException
    ) -> JSONResponse:
        """Handle custom application exceptions"""

        # Log the error with appropriate level
        correlation_id = getattr(request.state, 'correlation_id', exc.correlation_id)

        log_data = {
            "correlation_id": correlation_id,
            "path": request.url.path,
            "method": request.method,
            "error_code": exc.error_code,
            "status_code": exc.status_code
        }

        if exc.status_code >= 500:
            logger.error(f"Server error: {exc.message}", extra=log_data, exc_info=True)
        elif exc.status_code >= 400:
            logger.warning(f"Client error: {exc.message}", extra=log_data)
        else:
            logger.info(f"Service degradation: {exc.message}", extra=log_data)

        # Update correlation ID if needed
        if not exc.correlation_id:
            exc.correlation_id = correlation_id

        response_data = exc.to_dict()
        response_data["error"]["path"] = request.url.path

        return JSONResponse(
            status_code=exc.status_code,
            content=response_data,
            headers=exc.get_response_headers()
        )

    async def handle_http_exception(
            self,
            request: Request,
            exc: Union[HTTPException, StarletteHTTPException]
    ) -> JSONResponse:
        """Handle FastAPI HTTP exceptions"""

        correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))

        logger.warning(
            f"HTTP exception: {exc.detail}",
            extra={
                "correlation_id": correlation_id,
                "path": request.url.path,
                "status_code": exc.status_code
            }
        )

        response_data = self.formatter.format_error_response(
            error_code="HTTP_ERROR",
            message=str(exc.detail),
            status_code=exc.status_code,
            correlation_id=correlation_id,
            request_path=request.url.path
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=response_data,
            headers={"X-Correlation-ID": correlation_id}
        )

    async def handle_validation_exception(
            self,
            request: Request,
            exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation exceptions"""

        correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))

        logger.warning(
            f"Validation error: {len(exc.errors())} validation failures",
            extra={
                "correlation_id": correlation_id,
                "path": request.url.path,
                "errors": exc.errors()
            }
        )

        response_data = self.formatter.format_validation_error_response(
            validation_errors=exc.errors(),
            correlation_id=correlation_id
        )
        response_data["error"]["path"] = request.url.path

        return JSONResponse(
            status_code=422,
            content=response_data,
            headers={"X-Correlation-ID": correlation_id}
        )

    async def handle_database_exception(
            self,
            request: Request,
            exc: SQLAlchemyError
    ) -> JSONResponse:
        """Handle database-related exceptions"""

        correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))

        # Determine specific error type
        if isinstance(exc, IntegrityError):
            error_code = "DATABASE_INTEGRITY_ERROR"
            message = "Data integrity constraint violation"
            status_code = 409
            user_message = "The operation conflicts with existing data"
        else:
            error_code = "DATABASE_ERROR"
            message = "Database operation failed"
            status_code = 500
            user_message = "A database error occurred"

        logger.error(
            f"Database error: {str(exc)}",
            extra={
                "correlation_id": correlation_id,
                "path": request.url.path,
                "error_type": type(exc).__name__
            },
            exc_info=True
        )

        response_data = self.formatter.format_error_response(
            error_code=error_code,
            message=user_message,
            status_code=status_code,
            details={
                "error_type": type(exc).__name__,
                "database_error": True
            } if settings.ENVIRONMENT == "development" else None,
            correlation_id=correlation_id,
            request_path=request.url.path
        )

        return JSONResponse(
            status_code=status_code,
            content=response_data,
            headers={"X-Correlation-ID": correlation_id}
        )

    async def handle_generic_exception(
            self,
            request: Request,
            exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions"""

        correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))

        logger.error(
            f"Unexpected error: {str(exc)}",
            extra={
                "correlation_id": correlation_id,
                "path": request.url.path,
                "error_type": type(exc).__name__
            },
            exc_info=True
        )

        # Don't expose internal error details in production
        if settings.ENVIRONMENT == "production":
            message = "An unexpected error occurred"
            details = None
        else:
            message = f"Unexpected error: {str(exc)}"
            details = {
                "error_type": type(exc).__name__,
                "error_message": str(exc)
            }

        response_data = self.formatter.format_error_response(
            error_code="INTERNAL_SERVER_ERROR",
            message=message,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            request_path=request.url.path,
            help_url="/docs/error-handling"
        )

        return JSONResponse(
            status_code=500,
            content=response_data,
            headers={"X-Correlation-ID": correlation_id}
        )


# Global exception handler instance
global_exception_handler = GlobalExceptionHandler()


def setup_exception_handlers(app):
    """Setup all exception handlers for the FastAPI app"""

    @app.exception_handler(CustomException)
    async def custom_exception_handler(request: Request, exc: CustomException):
        return await global_exception_handler.handle_custom_exception(request, exc)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return await global_exception_handler.handle_http_exception(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
        return await global_exception_handler.handle_http_exception(request, exc)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return await global_exception_handler.handle_validation_exception(request, exc)

    @app.exception_handler(SQLAlchemyError)
    async def database_exception_handler(request: Request, exc: SQLAlchemyError):
        return await global_exception_handler.handle_database_exception(request, exc)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return await global_exception_handler.handle_generic_exception(request, exc)