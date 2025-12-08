import time
import json
import hashlib
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from app.utils.exceptions import RateLimitError
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limit configuration for different user tiers"""
    calls_per_minute: int
    calls_per_hour: int
    calls_per_day: int
    burst_limit: int  # Maximum burst requests
    tier_name: str


@dataclass
class RateLimitStatus:
    """Current rate limit status for a client"""
    minute_calls: int
    hour_calls: int
    day_calls: int
    minute_remaining: int
    hour_remaining: int
    day_remaining: int
    reset_time_minute: int
    reset_time_hour: int
    reset_time_day: int
    tier: str


class InMemoryRateLimitStore:
    """In-memory rate limit storage for when redis fails or unavailable"""

    def __init__(self):
        self.minute_calls: Dict[str, List[float]] = defaultdict(list)
        self.hour_calls: Dict[str, List[float]] = defaultdict(list)
        self.day_calls: Dict[str, List[float]] = defaultdict(list)
        self.user_tiers: Dict[str, str] = defaultdict(lambda: "free")

    def _cleanup_old_calls(self, calls_list: List[float], max_age_seconds: int):
        """Remove calls older than max_age_seconds"""
        cutoff_time = time.time() - max_age_seconds
        while calls_list and calls_list[0] < cutoff_time:
            calls_list.pop(0)

    def get_call_counts(self, key: str) -> Tuple[int, int, int]:
        """Get current call counts for minute, hour, and day"""
        current_time = time.time()

        # Cleanup old calls
        self._cleanup_old_calls(self.minute_calls[key], 60)
        self._cleanup_old_calls(self.hour_calls[key], 3600)
        self._cleanup_old_calls(self.day_calls[key], 86400)

        return (
            len(self.minute_calls[key]),
            len(self.hour_calls[key]),
            len(self.day_calls[key])
        )

    def record_call(self, key: str):
        """Record a new API call"""
        current_time = time.time()
        self.minute_calls[key].append(current_time)
        self.hour_calls[key].append(current_time)
        self.day_calls[key].append(current_time)

    def get_user_tier(self, key: str) -> str:
        """Get user tier for rate limiting"""
        return self.user_tiers.get(key, "free")

    def set_user_tier(self, key: str, tier: str):
        """Set user tier for rate limiting"""
        self.user_tiers[key] = tier


class AdvancedRateLimiter:
    """Advanced rate limiter with user-specific quotas"""

    def __init__(self):
        self.store = InMemoryRateLimitStore()

        # Define rate limit tiers
        self.rate_limit_configs = {
            "free": RateLimitConfig(
                calls_per_minute=30,
                calls_per_hour=500,
                calls_per_day=2000,
                burst_limit=10,
                tier_name="Free"
            ),
            "standard": RateLimitConfig(
                calls_per_minute=60,
                calls_per_hour=1000,
                calls_per_day=5000,
                burst_limit=20,
                tier_name="Standard"
            ),
            "premium": RateLimitConfig(
                calls_per_minute=120,
                calls_per_hour=2000,
                calls_per_day=10000,
                burst_limit=50,
                tier_name="Premium"
            ),

            #"enterprise": RateLimitConfig(
             #   calls_per_minute=300,
              #  calls_per_hour=5000,
               # calls_per_day=25000,
                #burst_limit=100,
                #tier_name="Enterprise"
            #)
        }

    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting"""
        # Try to get user ID from JWT token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                # In production, decode JWT to get user ID
                # For now, create a hash of the token
                token_hash = hashlib.sha256(auth_header.encode()).hexdigest()[:16]
                return f"user:{token_hash}"
            except Exception:
                pass

        # Fallback to IP-based limiting
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address with proxy support"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _get_user_tier(self, request: Request, client_key: str) -> str:
        """Determine user tier from request"""
        # Try to extract tier from JWT token or user database
        # For now, use a simple heuristic based on auth status
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            # Authenticated users get standard tier by default
            return self.store.get_user_tier(client_key) or "standard"
        else:
            # Anonymous users get free tier
            return "free"

    def _calculate_reset_times(self) -> Tuple[int, int, int]:
        """Calculate reset times for minute, hour, and day windows"""
        current_time = int(time.time())

        # Next minute boundary
        reset_minute = current_time + (60 - (current_time % 60))

        # Next hour boundary
        reset_hour = current_time + (3600 - (current_time % 3600))

        # Next day boundary (midnight UTC)
        reset_day = current_time + (86400 - (current_time % 86400))

        return reset_minute, reset_hour, reset_day

    def check_rate_limit(self, request: Request) -> Tuple[bool, RateLimitStatus, Optional[str]]:
        """
        Check if request should be rate limited

        Returns:
            (is_limited, status, error_message)
        """
        client_key = self._get_client_identifier(request)
        user_tier = self._get_user_tier(request, client_key)
        config = self.rate_limit_configs.get(user_tier, self.rate_limit_configs["free"])

        # Get current call counts
        minute_calls, hour_calls, day_calls = self.store.get_call_counts(client_key)

        # Calculate remaining calls
        minute_remaining = max(0, config.calls_per_minute - minute_calls)
        hour_remaining = max(0, config.calls_per_hour - hour_calls)
        day_remaining = max(0, config.calls_per_day - day_calls)

        # Calculate reset times
        reset_minute, reset_hour, reset_day = self._calculate_reset_times()

        # Create status object
        status = RateLimitStatus(
            minute_calls=minute_calls,
            hour_calls=hour_calls,
            day_calls=day_calls,
            minute_remaining=minute_remaining,
            hour_remaining=hour_remaining,
            day_remaining=day_remaining,
            reset_time_minute=reset_minute,
            reset_time_hour=reset_hour,
            reset_time_day=reset_day,
            tier=user_tier
        )

        # Check limits
        if minute_calls >= config.calls_per_minute:
            return True, status, f"Rate limit exceeded: {config.calls_per_minute} calls per minute for {config.tier_name} tier"

        if hour_calls >= config.calls_per_hour:
            return True, status, f"Rate limit exceeded: {config.calls_per_hour} calls per hour for {config.tier_name} tier"

        if day_calls >= config.calls_per_day:
            return True, status, f"Rate limit exceeded: {config.calls_per_day} calls per day for {config.tier_name} tier"

        # Record this call
        self.store.record_call(client_key)

        # Update remaining counts after recording
        status.minute_remaining = max(0, status.minute_remaining - 1)
        status.hour_remaining = max(0, status.hour_remaining - 1)
        status.day_remaining = max(0, status.day_remaining - 1)

        return False, status, None

    def get_rate_limit_headers(self, status: RateLimitStatus) -> Dict[str, str]:
        """Get rate limit headers for response"""
        config = self.rate_limit_configs.get(status.tier, self.rate_limit_configs["free"])

        return {
            "X-RateLimit-Limit-Minute": str(config.calls_per_minute),
            "X-RateLimit-Remaining-Minute": str(status.minute_remaining),
            "X-RateLimit-Reset-Minute": str(status.reset_time_minute),
            "X-RateLimit-Limit-Hour": str(config.calls_per_hour),
            "X-RateLimit-Remaining-Hour": str(status.hour_remaining),
            "X-RateLimit-Reset-Hour": str(status.reset_time_hour),
            "X-RateLimit-Limit-Day": str(config.calls_per_day),
            "X-RateLimit-Remaining-Day": str(status.day_remaining),
            "X-RateLimit-Reset-Day": str(status.reset_time_day),
            "X-RateLimit-Tier": status.tier.title()
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with enhanced features"""

    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = AdvancedRateLimiter()

        # Endpoints to exclude from rate limiting
        self.excluded_paths = {
            "/health",
            "/health/detailed",
            "/health/metrics",
            "/health/circuit-breakers",
            "/docs",
            "/redoc",
            "/openapi.json"
        }

    async def dispatch(self, request: Request, call_next):
        """Process request with advanced rate limiting"""

        # Skip rate limiting for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Skip rate limiting if disabled
        if not settings.ENABLE_RATE_LIMITING:
            return await call_next(request)

        try:
            # Check rate limits
            is_limited, status, error_message = self.rate_limiter.check_rate_limit(request)

            if is_limited:
                # Log rate limit violation
                client_ip = self.rate_limiter._get_client_ip(request)
                logger.warning(
                    f"Rate limit exceeded for {client_ip}: {request.url.path}",
                    extra={
                        "client_ip": client_ip,
                        "path": request.url.path,
                        "tier": status.tier,
                        "minute_calls": status.minute_calls,
                        "hour_calls": status.hour_calls
                    }
                )

                # Calculate retry-after header
                retry_after = min(
                    status.reset_time_minute - int(time.time()),
                    60  # Maximum 1 minute
                )

                # Create rate limit error response
                error_response = {
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": error_message,
                        "status": 429,
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "rate_limit",
                        "details": {
                            "tier": status.tier,
                            "limits": {
                                "minute": self.rate_limiter.rate_limit_configs[status.tier].calls_per_minute,
                                "hour": self.rate_limiter.rate_limit_configs[status.tier].calls_per_hour,
                                "day": self.rate_limiter.rate_limit_configs[status.tier].calls_per_day
                            },
                            "current_usage": {
                                "minute": status.minute_calls,
                                "hour": status.hour_calls,
                                "day": status.day_calls
                            },
                            "reset_times": {
                                "minute": status.reset_time_minute,
                                "hour": status.reset_time_hour,
                                "day": status.reset_time_day
                            },
                            "retry_after_seconds": retry_after
                        },
                        "help_url": "/docs/rate-limits"
                    }
                }

                # Get rate limit headers
                headers = self.rate_limiter.get_rate_limit_headers(status)
                headers["Retry-After"] = str(retry_after)

                return JSONResponse(
                    status_code=429,
                    content=error_response,
                    headers=headers
                )

            # Process request
            response = await call_next(request)

            # Add rate limit headers to successful responses
            rate_limit_headers = self.rate_limiter.get_rate_limit_headers(status)
            for header_name, header_value in rate_limit_headers.items():
                response.headers[header_name] = header_value

            return response

        except Exception as e:
            logger.error(f"Rate limiting error: {e}", exc_info=True)
            # Continue without rate limiting if there's an error
            return await call_next(request)