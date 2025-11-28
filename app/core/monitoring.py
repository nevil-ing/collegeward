import asyncio
import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.logging import get_logger
from app.core.circuit_breaker import service_registry
from app.db.session import get_db

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """Health status for a component"""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    response_time_ms: Optional[float] = None


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    active_connections: int
    timestamp: datetime


class HealthChecker:
    """Health check manager for system components"""

    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self.last_results: Dict[str, HealthStatus] = {}

    def register_check(self, name: str, check_func: callable):
        """Register a health check function"""
        self.checks[name] = check_func

    async def run_check(self, name: str) -> HealthStatus:
        """Run a specific health check"""
        if name not in self.checks:
            return HealthStatus(
                name=name,
                status="unhealthy",
                message="Health check not found",
                details={},
                timestamp=datetime.utcnow()
            )

        start_time = time.time()
        try:
            result = await self.checks[name]()
            response_time = (time.time() - start_time) * 1000

            if isinstance(result, HealthStatus):
                result.response_time_ms = response_time
                self.last_results[name] = result
                return result
            else:
                # Convert simple result to HealthStatus
                status = HealthStatus(
                    name=name,
                    status="healthy" if result else "unhealthy",
                    message="Check completed",
                    details={"result": result},
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time
                )
                self.last_results[name] = status
                return status

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Health check {name} failed: {e}")

            status = HealthStatus(
                name=name,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow(),
                response_time_ms=response_time
            )
            self.last_results[name] = status
            return status

    async def run_all_checks(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks"""
        results = {}

        # Run checks concurrently
        tasks = [self.run_check(name) for name in self.checks.keys()]
        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(check_results):
            name = list(self.checks.keys())[i]
            if isinstance(result, Exception):
                results[name] = HealthStatus(
                    name=name,
                    status="unhealthy",
                    message=f"Check execution failed: {str(result)}",
                    details={"error": str(result)},
                    timestamp=datetime.utcnow()
                )
            else:
                results[name] = result

        return results

    def get_overall_status(self, results: Dict[str, HealthStatus]) -> str:
        """Determine overall system health status"""
        if not results:
            return "unknown"

        statuses = [result.status for result in results.values()]

        if all(status == "healthy" for status in statuses):
            return "healthy"
        elif any(status == "unhealthy" for status in statuses):
            return "unhealthy"
        else:
            return "degraded"


class SystemMonitor:
    """System performance monitoring"""

    @staticmethod
    def get_system_metrics() -> SystemMetrics:
        """Get current system performance metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_free_gb = disk.free / (1024 * 1024 * 1024)

        # Network connections (approximate active connections)
        try:
            connections = len(psutil.net_connections())
        except (psutil.AccessDenied, PermissionError):
            # Fallback if we don't have permission to access network connections
            connections = 0

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_percent=disk.percent,
            disk_used_gb=disk_used_gb,
            disk_free_gb=disk_free_gb,
            active_connections=connections,
            timestamp=datetime.utcnow()
        )

    @staticmethod
    def check_system_health() -> HealthStatus:
        """Check overall system health based on metrics"""
        try:
            metrics = SystemMonitor.get_system_metrics()

            # Define thresholds
            cpu_warning = 80
            cpu_critical = 95
            memory_warning = 80
            memory_critical = 95
            disk_warning = 80
            disk_critical = 95

            issues = []
            status = "healthy"

            # Check CPU
            if metrics.cpu_percent > cpu_critical:
                issues.append(f"CPU usage critical: {metrics.cpu_percent:.1f}%")
                status = "unhealthy"
            elif metrics.cpu_percent > cpu_warning:
                issues.append(f"CPU usage high: {metrics.cpu_percent:.1f}%")
                if status == "healthy":
                    status = "degraded"

            # Check Memory
            if metrics.memory_percent > memory_critical:
                issues.append(f"Memory usage critical: {metrics.memory_percent:.1f}%")
                status = "unhealthy"
            elif metrics.memory_percent > memory_warning:
                issues.append(f"Memory usage high: {metrics.memory_percent:.1f}%")
                if status == "healthy":
                    status = "degraded"

            # Check Disk
            if metrics.disk_percent > disk_critical:
                issues.append(f"Disk usage critical: {metrics.disk_percent:.1f}%")
                status = "unhealthy"
            elif metrics.disk_percent > disk_warning:
                issues.append(f"Disk usage high: {metrics.disk_percent:.1f}%")
                if status == "healthy":
                    status = "degraded"

            message = "System resources normal" if not issues else "; ".join(issues)

            return HealthStatus(
                name="system",
                status=status,
                message=message,
                details=asdict(metrics),
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return HealthStatus(
                name="system",
                status="unhealthy",
                message=f"System monitoring failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )


# Health check functions
async def check_database_health() -> HealthStatus:
    """Check database connectivity and performance"""
    try:
        start_time = time.time()

        # Get database session
        from app.db.session import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            # Simple query to test connectivity
            result = await db.execute(text("SELECT 1"))
            row = result.fetchone()  # This is synchronous in async SQLAlchemy

            response_time = (time.time() - start_time) * 1000

            return HealthStatus(
                name="database",
                status="healthy",
                message="Database connection successful",
                details={
                    "response_time_ms": response_time,
                    "query_result": str(row) if row else None
                },
                timestamp=datetime.utcnow(),
                response_time_ms=response_time
            )

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return HealthStatus(
            name="database",
            status="unhealthy",
            message=f"Database connection failed: {str(e)}",
            details={"error": str(e)},
            timestamp=datetime.utcnow()
        )


async def check_qdrant_health() -> HealthStatus:
    """Check Qdrant vector database health"""
    try:
        from app.rag.qdrant_client import qdrant_manager

        start_time = time.time()

        # Check if Qdrant is accessible
        if qdrant_manager.client is None:
            raise Exception("Qdrant client not initialized")

        collections = qdrant_manager.client.get_collections()

        response_time = (time.time() - start_time) * 1000

        return HealthStatus(
            name="qdrant",
            status="healthy",
            message="Qdrant connection successful",
            details={
                "collections_count": len(collections.collections) if collections else 0,
                "response_time_ms": response_time,
                "collection_name": qdrant_manager.collection_name
            },
            timestamp=datetime.utcnow(),
            response_time_ms=response_time
        )

    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        return HealthStatus(
            name="qdrant",
            status="unhealthy",
            message=f"Qdrant connection failed: {str(e)}",
            details={"error": str(e)},
            timestamp=datetime.utcnow()
        )


async def check_external_services_health() -> HealthStatus:
    """Check external services health via circuit breakers"""
    try:
        circuit_stats = service_registry.get_all_stats()

        unhealthy_services = []
        degraded_services = []

        for service_name, stats in circuit_stats.items():
            if stats["state"] == "open":
                unhealthy_services.append(service_name)
            elif stats["state"] == "half_open" or stats["success_rate"] < 0.8:
                degraded_services.append(service_name)

        if unhealthy_services:
            status = "unhealthy"
            message = f"Services unavailable: {', '.join(unhealthy_services)}"
        elif degraded_services:
            status = "degraded"
            message = f"Services degraded: {', '.join(degraded_services)}"
        else:
            status = "healthy"
            message = "All external services operational"

        return HealthStatus(
            name="external_services",
            status=status,
            message=message,
            details={
                "circuit_breakers": circuit_stats,
                "unhealthy_services": unhealthy_services,
                "degraded_services": degraded_services
            },
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"External services health check failed: {e}")
        return HealthStatus(
            name="external_services",
            status="unhealthy",
            message=f"External services check failed: {str(e)}",
            details={"error": str(e)},
            timestamp=datetime.utcnow()
        )


# Global health checker instance
health_checker = HealthChecker()

# Register default health checks
health_checker.register_check("database", check_database_health)
health_checker.register_check("qdrant", check_qdrant_health)
health_checker.register_check("external_services", check_external_services_health)


async def check_system_health_async() -> HealthStatus:
    """Async wrapper for system health check"""
    return SystemMonitor.check_system_health()


health_checker.register_check("system", check_system_health_async)