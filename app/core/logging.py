import logging
import logging.handlers
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, asdict

from app.core.config import settings


@dataclass
class LogContext:
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    execution_time_ms: Optional[float] = None
    service_name: Optional[str] = None
    operation: Optional[str] = None


class EnhancedJSONFormatter(logging.Formatter):
    """Enhanced JSON formatter with structured logging and context"""

    def format(self, record: logging.LogRecord) -> str:

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }

        # Add context information
        context_fields = [
            'correlation_id', 'user_id', 'request_id', 'session_id',
            'client_ip', 'user_agent', 'endpoint', 'method',
            'execution_time_ms', 'service_name', 'operation'
        ]

        for field in context_fields:
            if hasattr(record, field) and getattr(record, field) is not None:
                log_entry[field] = getattr(record, field)

        # Add custom extra fields
        if hasattr(record, 'extra_data') and record.extra_data:
            log_entry['extra'] = record.extra_data

        # Add performance metrics
        if hasattr(record, 'metrics') and record.metrics:
            log_entry['metrics'] = record.metrics

        # Add security information
        if hasattr(record, 'security_event') and record.security_event:
            log_entry['security'] = record.security_event

        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info).split('\n')
            }

        # Add stack trace for errors
        if record.levelno >= logging.ERROR and not record.exc_info:
            log_entry['stack_trace'] = traceback.format_stack()

        return json.dumps(log_entry, default=str)


class PerformanceLogger:
        def __init__(self, logger: logging.Logger):
            self.logger = logger
        @contextmanager
        def measure_time(
                self,
                operation: str,
                service_name: Optional[str] = None,
                context: Optional[LogContext] = None
        ):
            """Context manager to measure execution time"""
            start_time = time.time()

            try:
                yield

            finally:
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                extra_data = {
                    'operation': operation,
                    'execution_time_ms': execution_time,
                    'service_name': service_name
                }

                if context:
                    extra_data.update(asdict(context))

                # Log performance based on execution time
                if execution_time > 5000:
                    self.logger.error(
                        f"Very slow operation: {operation} took {execution_time:.2f}ms",
                        extra=extra_data
                    )
                elif execution_time > 1000:
                    self.logger.warning(
                        f"Slow operation: {operation} took {execution_time:.2f}ms",
                        extra=extra_data
                    )
                else:
                    self.logger.debug(
                        f"Operation completed: {operation} took {execution_time:.2f}ms",
                        extra=extra_data
                    )


class SecurityLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_authentication_attempt(
            self,
            success: bool,
            user_id: Optional[str] = None,
            client_ip: Optional[str] = None,
            user_agent: Optional[str] = None,
            method: str = "unknown"
    ):
        """Log authentication attempts"""

        security_event = {
            'event_type': 'authentication',
            'success': success,
            'method': method,
            'client_ip': client_ip,
            'user_agent': user_agent
        }

        if success:
            self.logger.info(
                f"Successful authentication for user {user_id or 'unknown'}",
                extra={
                    'user_id': user_id,
                    'security_event': security_event
                }
            )
        else:
            self.logger.warning(
                f"Failed authentication attempt from {client_ip}",
                extra={
                    'client_ip': client_ip,
                    'security_event': security_event
                }
            )

    def log_authorization_failure(
            self,
            user_id: str,
            resource: str,
            action: str,
            client_ip: Optional[str] = None
    ):
        """Log authorization failures"""

        security_event = {
            'event_type': 'authorization_failure',
            'resource': resource,
            'action': action,
            'client_ip': client_ip
        }

        self.logger.warning(
            f"Authorization denied for user {user_id} accessing {resource}",
            extra={
                'user_id': user_id,
                'security_event': security_event
            }
        )

    def log_security_violation(
            self,
            violation_type: str,
            details: Dict[str, Any],
            client_ip: Optional[str] = None,
            user_id: Optional[str] = None
    ):
        """Log security violations"""

        security_event = {
            'event_type': 'security_violation',
            'violation_type': violation_type,
            'details': details,
            'client_ip': client_ip
        }

        self.logger.error(
            f"Security violation detected: {violation_type}",
            extra={
                'user_id': user_id,
                'client_ip': client_ip,
                'security_event': security_event
            }
        )

    def log_rate_limit_exceeded(
            self,
            client_ip: str,
            endpoint: str,
            limit_type: str,
            user_id: Optional[str] = None
    ):
        """Log rate limit violations"""

        security_event = {
            'event_type': 'rate_limit_exceeded',
            'endpoint': endpoint,
            'limit_type': limit_type,
            'client_ip': client_ip
        }

        self.logger.warning(
            f"Rate limit exceeded for {client_ip} on {endpoint}",
            extra={
                'user_id': user_id,
                'client_ip': client_ip,
                'endpoint': endpoint,
                'security_event': security_event
            }
        )


class APILogger:
    """Logger for API requests and responses"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_request(
            self,
            method: str,
            path: str,
            client_ip: str,
            user_agent: str,
            user_id: Optional[str] = None,
            correlation_id: Optional[str] = None,
            request_size: Optional[int] = None
    ):
        """Log API request"""

        extra_data = {
            'method': method,
            'endpoint': path,
            'client_ip': client_ip,
            'user_agent': user_agent,
            'user_id': user_id,
            'correlation_id': correlation_id,
            'request_size': request_size,
            'event_type': 'api_request'
        }

        self.logger.info(
            f"{method} {path} from {client_ip}",
            extra=extra_data
        )

    def log_response(
            self,
            method: str,
            path: str,
            status_code: int,
            response_time_ms: float,
            response_size: Optional[int] = None,
            user_id: Optional[str] = None,
            correlation_id: Optional[str] = None
    ):
        """Log API response"""

        extra_data = {
            'method': method,
            'endpoint': path,
            'status_code': status_code,
            'execution_time_ms': response_time_ms,
            'response_size': response_size,
            'user_id': user_id,
            'correlation_id': correlation_id,
            'event_type': 'api_response'
        }

        # Log level based on status code and response time
        if status_code >= 500:
            log_level = logging.ERROR
        elif status_code >= 400:
            log_level = logging.WARNING
        elif response_time_ms > 5000:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

        self.logger.log(
            log_level,
            f"{method} {path} -> {status_code} in {response_time_ms:.2f}ms",
            extra=extra_data
        )


class DatabaseLogger:
    """Logger for database operations"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_query(
            self,
            query_type: str,
            table: str,
            execution_time_ms: float,
            rows_affected: Optional[int] = None,
            user_id: Optional[str] = None
    ):
        """Log database query"""

        extra_data = {
            'operation': f"db_{query_type.lower()}",
            'table': table,
            'execution_time_ms': execution_time_ms,
            'rows_affected': rows_affected,
            'user_id': user_id,
            'event_type': 'database_operation'
        }

        if execution_time_ms > 1000:
            self.logger.warning(
                f"Slow {query_type} query on {table}: {execution_time_ms:.2f}ms",
                extra=extra_data
            )
        else:
            self.logger.debug(
                f"{query_type} query on {table}: {execution_time_ms:.2f}ms",
                extra=extra_data
            )

def setup_logging():
        """Setup logging configuration"""

        #create log dir
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))


        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)


        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        if settings.ENVIRONMENT == "development":

            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:

            console_formatter = EnhancedJSONFormatter()

        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Main application log file
        app_handler = logging.handlers.RotatingFileHandler(
            log_dir / "collegeward.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        app_handler.setLevel(logging.DEBUG)
        app_handler.setFormatter(EnhancedJSONFormatter())
        root_logger.addHandler(app_handler)

        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(EnhancedJSONFormatter())
        root_logger.addHandler(error_handler)

        # Security log file
        security_handler = logging.handlers.RotatingFileHandler(
            log_dir / "security.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(EnhancedJSONFormatter())

        #filter for security events
        class SecurityFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'security_event')

        security_handler.addFilter(SecurityFilter())
        root_logger.addHandler(security_handler)

        # Performance log file
        performance_handler = logging.handlers.RotatingFileHandler(
            log_dir / "performance.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )
        performance_handler.setLevel(logging.DEBUG)
        performance_handler.setFormatter(EnhancedJSONFormatter())

        #filter for performance events
        class PerformanceFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'execution_time_ms') or hasattr(record, 'metrics')

        performance_handler.addFilter(PerformanceFilter())
        root_logger.addHandler(performance_handler)

        # API access log file
        api_handler = logging.handlers.RotatingFileHandler(
            log_dir / "api_access.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        api_handler.setLevel(logging.INFO)
        api_handler.setFormatter(EnhancedJSONFormatter())

        #filter for API events
        class APIFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'event_type') and record.event_type in ['api_request', 'api_response']

        api_handler.addFilter(APIFilter())
        root_logger.addHandler(api_handler)

        # Set specific logger levels
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("qdrant_client").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
        """Get logger instance with proper configuration"""
        return logging.getLogger(name)

def get_performance_logger(name: str) -> PerformanceLogger:
        """Get performance logger instance"""
        return PerformanceLogger(get_logger(name))

def get_security_logger(name: str) -> SecurityLogger:
        """Get security logger instance"""
        return SecurityLogger(get_logger(name))

def get_api_logger(name: str) -> APILogger:
        """Get API logger instance"""
        return APILogger(get_logger(name))

def get_database_logger(name: str) -> DatabaseLogger:
        """Get database logger instance"""
        return DatabaseLogger(get_logger(name))