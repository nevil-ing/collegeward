import time
import uuid
import json
from typing import Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_api_logger, get_performance_logger, get_security_logger
from app.core.config import settings

api_logger = get_api_logger(__name__)
performance_logger = get_performance_logger(__name__)
security_logger = get_security_logger(__name__)


class ComprehensiveMonitoringMiddleware(BaseHTTPMiddleware):
    """monitoring middleware for requests, responses, and performance"""

    def __init__(self, app):
        super().__init__(app)
        self.excluded_paths = {
            "/health",
            "/health/detailed",
            "/health/metrics",
            "/health/circuit-breakers",
            "/docs",
            "/redoc",
            "/openapi.json"
        }

    def _get_client_info(self, request: Request) -> Dict[str, str]:
        """Extract client information from request"""
        return {
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "referer": request.headers.get("Referer", ""),
            "origin": request.headers.get("Origin", "")
        }

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP with proxy support"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request if available"""
        # Try to get from request state
        if hasattr(request.state, 'user_id'):
            return request.state.user_id

        # Try to extract from Authorization header
        #auth_header = request.headers.get("Authorization", "")
       # if auth_header.startswith("Bearer "):

           # return "authenticated_user"

        return None

    def _should_monitor(self, request: Request) -> bool:
        """Determine if request should be monitored"""
        return request.url.path not in self.excluded_paths

    def _get_request_size(self, request: Request) -> Optional[int]:
        """Get request content length"""
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass
        return None

    def _get_response_size(self, response: Response) -> Optional[int]:
        """Get response content length"""
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass
        return None

    def _detect_suspicious_activity(
            self,
            request: Request,
            client_info: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Detect potentially suspicious request patterns"""

        suspicious_indicators = []

        # Check for suspicious user agents
        user_agent = client_info["user_agent"].lower()
        suspicious_agents = ["bot", "crawler", "spider", "scraper", "scanner"]
        if any(agent in user_agent for agent in suspicious_agents):
            suspicious_indicators.append("suspicious_user_agent")

        # Check for missing or suspicious referer
        referer = client_info["referer"]
        if request.method == "POST" and not referer and request.url.path not in ["/api/v1/auth/login",
                                                                                 "/api/v1/auth/register"]:
            suspicious_indicators.append("missing_referer_on_post")

        # Check for unusual request patterns
        if len(request.url.query) > 1000:  # Very long query string
            suspicious_indicators.append("long_query_string")

        # Check for potential path traversal
        if "../" in str(request.url.path) or "..\\" in str(request.url.path):
            suspicious_indicators.append("path_traversal_attempt")

        if suspicious_indicators:
            return {
                "indicators": suspicious_indicators,
                "severity": "medium" if len(suspicious_indicators) == 1 else "high"
            }

        return None

    async def dispatch(self, request: Request, call_next):
        """Process request with comprehensive monitoring"""

        # Skip monitoring for excluded paths
        if not self._should_monitor(request):
            return await call_next(request)

        # Generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        # Get client information
        client_info = self._get_client_info(request)
        user_id = self._get_user_id(request)
        request_size = self._get_request_size(request)

        # Start timing
        start_time = time.time()

        # Log request
        if settings.ENABLE_REQUEST_LOGGING:
            api_logger.log_request(
                method=request.method,
                path=request.url.path,
                client_ip=client_info["client_ip"],
                user_agent=client_info["user_agent"],
                user_id=user_id,
                correlation_id=correlation_id,
                request_size=request_size
            )

        # Check for suspicious activity
        suspicious_activity = self._detect_suspicious_activity(request, client_info)
        if suspicious_activity:
            security_logger.log_security_violation(
                violation_type="suspicious_request_pattern",
                details=suspicious_activity,
                client_ip=client_info["client_ip"],
                user_id=user_id
            )

        # Process request
        try:
            response = await call_next(request)

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            response_size = self._get_response_size(response)

            # Add monitoring headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"

            # Log response
            if settings.ENABLE_REQUEST_LOGGING:
                api_logger.log_response(
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    response_time_ms=response_time_ms,
                    response_size=response_size,
                    user_id=user_id,
                    correlation_id=correlation_id
                )

            # Log performance metrics
            with performance_logger.measure_time(
                    operation=f"{request.method} {request.url.path}",
                    service_name="api_endpoint"
            ):
                pass  # Time already measured above

            return response

        except Exception as e:
            # Calculate response time for failed requests
            response_time_ms = (time.time() - start_time) * 1000

            # Log error response
            api_logger.log_response(
                method=request.method,
                path=request.url.path,
                status_code=500,
                response_time_ms=response_time_ms,
                user_id=user_id,
                correlation_id=correlation_id
            )

            # Re-raise the exception to be handled by error middleware
            raise


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation and sanitization"""

    def __init__(self, app):
        super().__init__(app)
        self.max_request_size = settings.MAX_FILE_SIZE
        self.excluded_paths = {"/health", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        """Validate request before processing"""

        # Skip validation for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Validate request size
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    from app.utils.exceptions import PayloadTooLargeError
                    raise PayloadTooLargeError(
                        "Request payload too large",
                        self.max_request_size,
                        size
                    )
            except ValueError:
                pass

        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("Content-Type", "")

            # Allow multipart/form-data for file uploads
            if not content_type:
                from app.utils.exceptions import InputValidationError
                raise InputValidationError(
                    "Content-Type header is required for POST/PUT requests",
                    "content_type"
                )

        return await call_next(request)


class ResponseEnhancementMiddleware(BaseHTTPMiddleware):
    """Middleware to enhance responses with additional headers and information"""

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        """Enhance response with additional headers"""

        response = await call_next(request)

        # Add API version header
        response.headers["X-API-Version"] = "1.0.0"

        # Add server information (minimal for security)
        response.headers["X-Powered-By"] = "StudyBlitzAI"

        # Add cache control for API responses
        if request.url.path.startswith("/api/"):
            if request.method == "GET":
                # Cache GET requests for 5 minutes by default
                response.headers["Cache-Control"] = "public, max-age=300"
            else:
                # Don't cache non-GET requests
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

        # Add CORS headers if not already present
        if "Access-Control-Allow-Origin" not in response.headers:
            origin = request.headers.get("Origin")
            if origin and origin in settings.CORS_ORIGINS:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"

        return response