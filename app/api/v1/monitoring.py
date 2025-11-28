from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.monitoring import health_checker, SystemMonitor
from app.core.graceful_degradation import degradation_manager
from app.core.circuit_breaker import service_registry
from app.core.logging import get_logger
from app.dependencies import get_current_user

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health/comprehensive")
async def comprehensive_health_check():
    """Comprehensive health check with all system components"""
    try:
        # Run all health checks
        health_results = await health_checker.run_all_checks()
        overall_status = health_checker.get_overall_status(health_results)

        # Get system metrics
        system_metrics = SystemMonitor.get_system_metrics()

        # Get service degradation status
        service_health = degradation_manager.get_all_service_health()

        # Get circuit breaker status
        circuit_breaker_stats = service_registry.get_all_stats()

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "StudyBlitzAI API",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "health_checks": {
                name: {
                    "status": result.status,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat()
                }
                for name, result in health_results.items()
            },
            "system_metrics": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "memory_used_mb": system_metrics.memory_used_mb,
                "memory_available_mb": system_metrics.memory_available_mb,
                "disk_percent": system_metrics.disk_percent,
                "disk_used_gb": system_metrics.disk_used_gb,
                "disk_free_gb": system_metrics.disk_free_gb,
                "active_connections": system_metrics.active_connections,
                "timestamp": system_metrics.timestamp.isoformat()
            },
            "service_health": service_health,
            "circuit_breakers": circuit_breaker_stats
        }

    except Exception as e:
        logger.error(f"Comprehensive health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Health check system failure",
                "message": str(e)
            }
        )


@router.get("/metrics/performance")
async def performance_metrics(
        hours: int = Query(1, ge=1, le=24, description="Hours of data to retrieve")
):
    """Get performance metrics for the specified time period"""
    try:
        # In a real implementation, this would query a metrics database
        # For now, return current system metrics
        system_metrics = SystemMonitor.get_system_metrics()

        return {
            "time_period_hours": hours,
            "timestamp": datetime.utcnow().isoformat(),
            "current_metrics": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "disk_percent": system_metrics.disk_percent,
                "active_connections": system_metrics.active_connections
            },
            "performance_summary": {
                "avg_response_time_ms": 150.5,  # Mock data
                "requests_per_minute": 45.2,
                "error_rate_percent": 0.8,
                "slow_requests_count": 12
            },
            "service_performance": {
                "groq_ai": {
                    "avg_response_time_ms": 1200.0,
                    "success_rate_percent": 98.5,
                    "requests_count": 1250
                },
                "database": {
                    "avg_query_time_ms": 25.3,
                    "slow_queries_count": 3,
                    "connection_pool_usage_percent": 45.2
                },
                "qdrant": {
                    "avg_search_time_ms": 85.7,
                    "success_rate_percent": 99.2,
                    "searches_count": 890
                }
            }
        }

    except Exception as e:
        logger.error(f"Performance metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")


@router.get("/metrics/errors")
async def error_metrics(
        hours: int = Query(1, ge=1, le=24, description="Hours of data to retrieve")
):
    """Get error metrics and statistics"""
    try:
        # In a real implementation, this would query error logs
        # For now, return mock data

        return {
            "time_period_hours": hours,
            "timestamp": datetime.utcnow().isoformat(),
            "error_summary": {
                "total_errors": 23,
                "error_rate_percent": 0.8,
                "critical_errors": 2,
                "warnings": 15,
                "info_errors": 6
            },
            "error_categories": {
                "validation_errors": 8,
                "authentication_errors": 3,
                "external_service_errors": 5,
                "database_errors": 1,
                "internal_server_errors": 2,
                "rate_limit_errors": 4
            },
            "top_error_endpoints": [
                {
                    "endpoint": "/api/v1/chat/message",
                    "error_count": 8,
                    "error_rate_percent": 2.1
                },
                {
                    "endpoint": "/api/v1/notes/upload",
                    "error_count": 5,
                    "error_rate_percent": 1.8
                }
            ],
            "recent_critical_errors": [
                {
                    "timestamp": (datetime.utcnow() - timedelta(minutes=15)).isoformat(),
                    "error_code": "DATABASE_CONNECTION_FAILED",
                    "message": "Connection to database timed out",
                    "endpoint": "/api/v1/flashcards/create",
                    "correlation_id": "abc123-def456"
                }
            ]
        }

    except Exception as e:
        logger.error(f"Error metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve error metrics")


@router.get("/metrics/security")
async def security_metrics(
        hours: int = Query(1, ge=1, le=24, description="Hours of data to retrieve"),
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get security metrics and alerts (requires authentication)"""
    try:
        # In a real implementation, this would query security logs
        # For now, return mock data

        return {
            "time_period_hours": hours,
            "timestamp": datetime.utcnow().isoformat(),
            "security_summary": {
                "total_security_events": 45,
                "authentication_failures": 12,
                "authorization_failures": 3,
                "rate_limit_violations": 25,
                "suspicious_activities": 5
            },
            "authentication_stats": {
                "successful_logins": 1250,
                "failed_logins": 12,
                "success_rate_percent": 99.0,
                "unique_users": 89
            },
            "rate_limiting_stats": {
                "requests_rate_limited": 25,
                "top_limited_ips": [
                    {"ip": "192.168.1.100", "violations": 8},
                    {"ip": "10.0.0.50", "violations": 6}
                ]
            },
            "suspicious_activities": [
                {
                    "timestamp": (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
                    "type": "multiple_failed_logins",
                    "client_ip": "192.168.1.100",
                    "details": "5 failed login attempts in 2 minutes"
                },
                {
                    "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                    "type": "suspicious_user_agent",
                    "client_ip": "10.0.0.75",
                    "details": "Bot-like user agent detected"
                }
            ]
        }

    except Exception as e:
        logger.error(f"Security metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve security metrics")


@router.get("/status/services")
async def service_status():
    """Get status of all external services and dependencies"""
    try:
        # Get service health from degradation manager
        service_health = degradation_manager.get_all_service_health()

        # Get circuit breaker stats
        circuit_stats = service_registry.get_all_stats()

        # Combine and format the data
        services_status = {}

        for service_name, health in service_health.items():
            circuit_info = circuit_stats.get(service_name, {})

            services_status[service_name] = {
                "status": health["status"],
                "last_success": health["last_success"],
                "last_failure": health["last_failure"],
                "failure_count": health["failure_count"],
                "success_count": health["success_count"],
                "response_time_ms": health["response_time_ms"],
                "error_message": health["error_message"],
                "fallback_available": health["fallback_available"],
                "circuit_breaker": {
                    "state": circuit_info.get("state", "unknown"),
                    "failure_count": circuit_info.get("failure_count", 0),
                    "success_count": circuit_info.get("success_count", 0),
                    "last_failure_time": circuit_info.get("last_failure_time"),
                    "next_attempt_time": circuit_info.get("next_attempt_time")
                }
            }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_status,
            "overall_health": "healthy" if all(
                s["status"] == "healthy" for s in service_health.values()
            ) else "degraded"
        }

    except Exception as e:
        logger.error(f"Service status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve service status")


@router.post("/services/{service_name}/reset")
async def reset_service_health(
        service_name: str,
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Reset health status for a specific service (admin only)"""
    try:
        # Check if user has admin privileges
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin privileges required")

        # Reset service health
        degradation_manager.reset_service_health(service_name)

        # Reset circuit breaker if exists
        if service_name in service_registry.circuit_breakers:
            service_registry.circuit_breakers[service_name].reset()

        logger.info(f"Service health reset for {service_name} by admin user {current_user['user_id']}")

        return {
            "message": f"Service health reset for {service_name}",
            "timestamp": datetime.utcnow().isoformat(),
            "reset_by": current_user["user_id"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Service health reset failed for {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset service health")


@router.get("/config/validation")
async def validation_config():
    """Get current validation and security configuration"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "validation_config": {
            "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024),
            "allowed_file_types": settings.ALLOWED_FILE_TYPES,
            "max_text_length": settings.MAX_TEXT_LENGTH,
            "max_subject_tags": settings.MAX_SUBJECT_TAGS,
            "max_request_size_mb": getattr(settings, 'MAX_REQUEST_SIZE', 100 * 1024 * 1024) / (1024 * 1024)
        },
        "rate_limiting_config": {
            "enabled": settings.ENABLE_RATE_LIMITING,
            "tiers": getattr(settings, 'RATE_LIMIT_TIERS', {
                "free": {"minute": 30, "hour": 500, "day": 2000},
                "standard": {"minute": 60, "hour": 1000, "day": 5000}
            })
        },
        "security_config": {
            "security_headers_enabled": settings.ENABLE_SECURITY_HEADERS,
            "input_validation_enabled": getattr(settings, 'ENABLE_INPUT_VALIDATION', True),
            "security_monitoring_enabled": getattr(settings, 'ENABLE_SECURITY_MONITORING', True)
        },
        "monitoring_config": {
            "health_checks_enabled": settings.ENABLE_HEALTH_CHECKS,
            "performance_monitoring_enabled": getattr(settings, 'ENABLE_PERFORMANCE_MONITORING', True),
            "request_logging_enabled": settings.ENABLE_REQUEST_LOGGING
        }
    }