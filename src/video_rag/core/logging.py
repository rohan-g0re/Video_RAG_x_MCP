"""
Enhanced Logging System for MCP Video RAG System.

This module provides a comprehensive logging system with structured logging,
error handling integration, and configurable output formats.
"""

import logging
import logging.handlers
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from contextlib import contextmanager
from enum import Enum

from .interfaces import ILoggingManager
from .exceptions import VideoRAGError, ErrorSeverity, ErrorCategory


class LogFormat(Enum):
    """Available log formats."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    STRUCTURED = "structured"


class LogLevel(Enum):
    """Log levels with numeric values."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, '')
        reset_color = self.COLORS['RESET']
        
        # Store original level name
        original_levelname = record.levelname
        
        # Add color to level name
        record.levelname = f"{level_color}{record.levelname}{reset_color}"
        
        # Format the record
        formatted = super().format(record)
        
        # Restore original level name
        record.levelname = original_levelname
        
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add custom fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class StructuredFormatter(logging.Formatter):
    """Structured formatter with key-value pairs."""
    
    def format(self, record):
        """Format log record as structured text."""
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        parts = [
            f"timestamp={timestamp}",
            f"level={record.levelname}",
            f"logger={record.name}",
            f"message=\"{record.getMessage()}\"",
            f"module={record.module}",
            f"function={record.funcName}",
            f"line={record.lineno}"
        ]
        
        # Add custom fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                if isinstance(value, str):
                    parts.append(f"{key}=\"{value}\"")
                else:
                    parts.append(f"{key}={value}")
        
        # Add exception information
        if record.exc_info:
            exc_type = record.exc_info[0].__name__ if record.exc_info[0] else "Unknown"
            exc_msg = str(record.exc_info[1]) if record.exc_info[1] else ""
            parts.append(f"exception_type={exc_type}")
            parts.append(f"exception_message=\"{exc_msg}\"")
        
        return " ".join(parts)


class VideoRAGLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds context information to log records."""
    
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Process log message and add extra context."""
        # Add extra fields to the log record
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs
    
    def log_error(self, error: Union[Exception, VideoRAGError], message: Optional[str] = None):
        """Log an error with enhanced information."""
        if isinstance(error, VideoRAGError):
            # Log VideoRAG errors with full context
            extra_data = {
                'error_code': error.error_code,
                'error_category': error.category.value if error.category else None,
                'error_severity': error.severity.value,
                'error_recoverable': error.recoverable,
                'error_details': error.details,
                'error_suggestions': error.suggestions
            }
            
            log_message = message or error.message
            log_level = self._get_log_level_for_severity(error.severity)
            
            self.log(log_level, log_message, extra=extra_data, exc_info=True)
        else:
            # Log standard exceptions
            log_message = message or str(error)
            self.error(log_message, exc_info=True)
    
    def _get_log_level_for_severity(self, severity: ErrorSeverity) -> int:
        """Map error severity to log level."""
        mapping = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return mapping.get(severity, logging.ERROR)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        extra_data = {
            'operation': operation,
            'duration_seconds': duration,
            'performance_metric': True
        }
        extra_data.update(kwargs)
        
        self.info(f"Performance: {operation} completed in {duration:.3f}s", extra=extra_data)
    
    def log_audit(self, action: str, user: Optional[str] = None, **kwargs):
        """Log audit trail information."""
        extra_data = {
            'audit_action': action,
            'audit_user': user,
            'audit_timestamp': datetime.now().isoformat()
        }
        extra_data.update(kwargs)
        
        self.info(f"Audit: {action}", extra=extra_data)


class EnhancedLoggingManager(ILoggingManager):
    """Enhanced logging manager with structured logging and error integration."""
    
    def __init__(self):
        self._loggers: Dict[str, VideoRAGLoggerAdapter] = {}
        self._handlers: List[logging.Handler] = []
        self._formatters: Dict[LogFormat, logging.Formatter] = {}
        self._configured = False
        self._log_level = LogLevel.INFO
        self._log_format = LogFormat.DETAILED
        
        # Initialize formatters
        self._init_formatters()
    
    def _init_formatters(self):
        """Initialize log formatters."""
        self._formatters = {
            LogFormat.SIMPLE: logging.Formatter(
                '%(levelname)s - %(message)s'
            ),
            LogFormat.DETAILED: ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ),
            LogFormat.JSON: JSONFormatter(),
            LogFormat.STRUCTURED: StructuredFormatter()
        }
    
    def configure_logging(self, config: Dict[str, Any]) -> None:
        """Configure the logging system from configuration."""
        # Clear existing handlers
        self._clear_handlers()
        
        # Set global log level
        level_str = config.get('level', 'INFO').upper()
        try:
            self._log_level = LogLevel[level_str]
        except KeyError:
            self._log_level = LogLevel.INFO
        
        # Configure console logging
        console_config = config.get('console', {})
        if console_config.get('enabled', True):
            self._configure_console_handler(console_config)
        
        # Configure file logging
        file_config = config.get('file', {})
        if file_config.get('enabled', False):
            self._configure_file_handler(file_config)
        
        # Configure structured logging
        structured_config = config.get('structured', {})
        if structured_config.get('enabled', False):
            self._configure_structured_handler(structured_config)
        
        # Set component-specific log levels
        components = config.get('components', {})
        for component, level_str in components.items():
            try:
                level = LogLevel[level_str.upper()]
                logger = logging.getLogger(component)
                logger.setLevel(level.value)
            except KeyError:
                pass  # Ignore invalid log levels
        
        # Apply configuration to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self._log_level.value)
        
        self._configured = True
    
    def _configure_console_handler(self, config: Dict[str, Any]):
        """Configure console logging handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self._log_level.value)
        
        # Choose formatter
        format_type = config.get('format', 'detailed')
        if format_type == 'json':
            formatter = self._formatters[LogFormat.JSON]
        elif format_type == 'structured':
            formatter = self._formatters[LogFormat.STRUCTURED]
        elif format_type == 'simple':
            formatter = self._formatters[LogFormat.SIMPLE]
        else:
            formatter = self._formatters[LogFormat.DETAILED]
        
        handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)
    
    def _configure_file_handler(self, config: Dict[str, Any]):
        """Configure file logging handler."""
        log_file = Path(config.get('path', 'logs/video_rag.log'))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Choose handler type based on rotation
        rotation = config.get('rotation', 'size')
        if rotation == 'time':
            handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=config.get('backup_count', 7)
            )
        else:
            max_bytes = config.get('max_size_mb', 10) * 1024 * 1024
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=config.get('backup_count', 5)
            )
        
        handler.setLevel(self._log_level.value)
        
        # Use detailed format for file logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)
    
    def _configure_structured_handler(self, config: Dict[str, Any]):
        """Configure structured logging handler."""
        format_type = config.get('format', 'json')
        
        if format_type == 'json':
            formatter = self._formatters[LogFormat.JSON]
        else:
            formatter = self._formatters[LogFormat.STRUCTURED]
        
        # Configure separate file for structured logs
        structured_file = Path(config.get('path', 'logs/structured.log'))
        structured_file.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(structured_file)
        handler.setLevel(self._log_level.value)
        handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)
    
    def _clear_handlers(self):
        """Clear existing handlers."""
        root_logger = logging.getLogger()
        for handler in self._handlers:
            root_logger.removeHandler(handler)
            handler.close()
        self._handlers.clear()
    
    def get_logger(self, name: str, **context) -> VideoRAGLoggerAdapter:
        """Get or create a logger with context."""
        if name not in self._loggers:
            base_logger = logging.getLogger(name)
            
            if not self._configured:
                self._setup_default_logging()
            
            # Create adapter with context
            adapter = VideoRAGLoggerAdapter(base_logger, context)
            self._loggers[name] = adapter
        
        return self._loggers[name]
    
    def set_log_level(self, level: Union[str, LogLevel]) -> None:
        """Set global log level."""
        if isinstance(level, str):
            try:
                level = LogLevel[level.upper()]
            except KeyError:
                level = LogLevel.INFO
        
        self._log_level = level
        
        # Update all handlers
        for handler in self._handlers:
            handler.setLevel(level.value)
        
        # Update root logger
        logging.getLogger().setLevel(level.value)
    
    def _setup_default_logging(self):
        """Set up default logging configuration."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._log_level.value)
        console_handler.setFormatter(self._formatters[LogFormat.DETAILED])
        
        root_logger = logging.getLogger()
        root_logger.setLevel(self._log_level.value)
        root_logger.addHandler(console_handler)
        
        self._handlers.append(console_handler)
        self._configured = True
    
    @contextmanager
    def log_context(self, **context):
        """Context manager for adding context to all logs within the block."""
        # Store original extra data for all loggers
        original_extra = {}
        for name, logger in self._loggers.items():
            original_extra[name] = logger.extra.copy()
            logger.extra.update(context)
        
        try:
            yield
        finally:
            # Restore original extra data
            for name, logger in self._loggers.items():
                logger.extra = original_extra.get(name, {})
    
    def log_system_info(self):
        """Log system information for debugging."""
        import platform
        import psutil
        
        logger = self.get_logger("system")
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_usage': psutil.disk_usage('/').total if hasattr(psutil, 'disk_usage') else 'unknown'
        }
        
        logger.info("System information", extra=system_info)
    
    def create_performance_logger(self, operation: str) -> 'PerformanceLogger':
        """Create a performance logger for timing operations."""
        return PerformanceLogger(self.get_logger(f"performance.{operation}"), operation)


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, logger: VideoRAGLoggerAdapter, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
        self.metadata = {}
    
    def __enter__(self):
        """Start timing the operation."""
        self.start_time = datetime.now()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log the result."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            if exc_type:
                self.logger.error(
                    f"Operation failed: {self.operation}",
                    extra={
                        'operation': self.operation,
                        'duration_seconds': duration,
                        'success': False,
                        **self.metadata
                    }
                )
            else:
                self.logger.log_performance(self.operation, duration, **self.metadata)
    
    def add_metadata(self, **kwargs):
        """Add metadata to the performance log."""
        self.metadata.update(kwargs)


# Global logger instance
_global_logging_manager = None


def get_logging_manager() -> EnhancedLoggingManager:
    """Get the global logging manager instance."""
    global _global_logging_manager
    if _global_logging_manager is None:
        _global_logging_manager = EnhancedLoggingManager()
    return _global_logging_manager


def get_logger(name: str, **context) -> VideoRAGLoggerAdapter:
    """Get a logger with the specified name and context."""
    return get_logging_manager().get_logger(name, **context) 