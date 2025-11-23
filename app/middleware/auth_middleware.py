import time
import uuid
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.sql.util import clause_is_present
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.exceptions import CustomException, RateLimitError

logger = get_logger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, verify_token_fn):
        super() .__init__(app)
        self.verify_token_fn = verify_token_fn

    async def dispatch(self, request: Request, call_next):
        request.state.user_claims = None
        request.state.user_id = None

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1].strip()
            try:
                maybe_coro = self.verify_token_fn(token)
                claims = await maybe_coro if hasattr(maybe_coro, "__await__") else maybe_coro
                request.state.user_claims = claims
                request.state.user_id = claims.get("uid") or claims.get("sub") or claims.get("user_id")
            except Exception as e:
                logger.error("Token verification failed in middleware: ", e)
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Add HSTS header for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """global error handling middleware"""

    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())

        try:
            # Add correlation ID to request state for logging
            request.state.correlation_id = correlation_id

            response = await call_next(request)

            # correlation ID to successful responses
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        except CustomException as e:
            # Set correlation ID if not already set
            if not e.correlation_id:
                e.correlation_id = correlation_id


            if e.status_code >= 500:
                logger.error(f"Server error in {request.url.path}: {e}", extra={"correlation_id": correlation_id})
            elif e.status_code >= 400:
                logger.warning(f"Client error in {request.url.path}: {e}", extra={"correlation_id": correlation_id})
            else:
                logger.info(f"Service degradation in {request.url.path}: {e}", extra={"correlation_id": correlation_id})

            response_content = e.to_dict()
            return JSONResponse(
                status_code=e.status_code,
                content=response_content,
                headers={"X-Correlation-ID": correlation_id}
            )

        except HTTPException as e:
            logger.error(f"HTTP exception in {request.url.path}: {e}", extra={"correlation_id": correlation_id})
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "code": "HTTP_ERROR",
                        "message": e.detail,
                        "correlation_id": correlation_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                },
                headers={"X-Correlation-ID": correlation_id}
            )

        except Exception as e:
            logger.error(
                f"Unexpected error in {request.url.path}: {e}",
                exc_info=True,
                extra={"correlation_id": correlation_id}
            )

            from app.core.config import settings
            error_message = "An unexpected error occurred"
            error_details = {}

            if settings.ENVIRONMENT == "development":
                error_details = {
                    "exception_type": type(e).__name__,
                    "exception_message": str(e)
                }

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": error_message,
                        "correlation_id": correlation_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "details": error_details
                    }
                },
                headers={"X-Correlation-ID": correlation_id}
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests for monitoring and debugging"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Get client info
        client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
        user_agent = request.headers.get("User-Agent", "unknown")

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {client_ip} ({user_agent})"
        )

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        logger.info(
            f"Response: {response.status_code} "
            f"for {request.method} {request.url.path} "
            f"in {process_time:.3f}s"
        )

        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)

        return response


class AuthenticationLoggingMiddleware(BaseHTTPMiddleware):
    """Log authentication events for security monitoring"""

    async def dispatch(self, request: Request, call_next):
        # Check if this is an auth-related endpoint
        auth_endpoints = ["/api/v1/auth/login", "/api/v1/auth/register", "/api/v1/auth/logout"]

        if request.url.path in auth_endpoints:
            client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")

            # Log authentication attempt
            logger.info(f"Authentication attempt: {request.method} {request.url.path} from {client_ip}")

        response = await call_next(request)

        # Log authentication results
        if request.url.path in auth_endpoints:
            if response.status_code == 200:
                logger.info(f"Authentication successful: {request.url.path}")
            else:
                logger.warning(f"Authentication failed: {request.url.path} - Status: {response.status_code}")

        return response