"""
Custom exception hierarchy for MCP Video RAG System.

This module defines a comprehensive set of exceptions that can occur throughout
the video RAG system, providing clear error categorization and helpful error
messages for debugging and user feedback.
"""

from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in the system."""
    CONFIGURATION = "configuration"
    STORAGE = "storage"
    PROCESSING = "processing"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    RESOURCE = "resource"
    EXTERNAL_SERVICE = "external_service"
    USER_INPUT = "user_input"
    SYSTEM = "system"


class VideoRAGError(Exception):
    """Base exception class for all Video RAG system errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        category: Optional[ErrorCategory] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.category = category or ErrorCategory.SYSTEM
        self.severity = severity
        self.details = details or {}
        self.suggestions = suggestions or []
        self.recoverable = recoverable
        self.timestamp = None  # Will be set by error handler
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "category": self.category.value if self.category else None,
            "severity": self.severity.value,
            "details": self.details,
            "suggestions": self.suggestions,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"[{self.error_code}] {self.message}"]
        
        if self.details:
            parts.append(f"Details: {self.details}")
        
        if self.suggestions:
            parts.append(f"Suggestions: {', '.join(self.suggestions)}")
        
        return " | ".join(parts)


# Configuration Errors
class ConfigurationError(VideoRAGError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        
        suggestions = kwargs.get('suggestions', [])
        if config_key:
            suggestions.append(f"Check configuration value for '{config_key}'")
        
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            details=details,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'suggestions']}
        )


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    
    def __init__(self, validation_errors: List[str], **kwargs):
        message = f"Configuration validation failed: {'; '.join(validation_errors)}"
        details = kwargs.get('details', {})
        details['validation_errors'] = validation_errors
        
        super().__init__(
            message,
            details=details,
            suggestions=["Fix the configuration errors and restart the system"],
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


# Storage Errors
class StorageError(VideoRAGError):
    """Base class for storage-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.STORAGE,
            **kwargs
        )


class DatabaseError(StorageError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check database connection and permissions",
            "Verify database schema is up to date"
        ])
        
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'suggestions']}
        )


class FileSystemError(StorageError):
    """Raised when file system operations fail."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check file/directory permissions",
            "Verify disk space is available",
            "Ensure the path is accessible"
        ])
        
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'suggestions']}
        )


# Processing Errors
class ProcessingError(VideoRAGError):
    """Base class for processing-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PROCESSING,
            **kwargs
        )


class VideoProcessingError(ProcessingError):
    """Raised when video processing operations fail."""
    
    def __init__(self, message: str, video_path: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if video_path:
            details['video_path'] = video_path
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check if the video file is corrupted",
            "Verify FFmpeg is installed and accessible",
            "Ensure the video format is supported"
        ])
        
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'suggestions']}
        )


class TranscriptionError(ProcessingError):
    """Raised when audio transcription fails."""
    
    def __init__(self, message: str, audio_path: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if audio_path:
            details['audio_path'] = audio_path
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check if the audio file is valid",
            "Verify Whisper model is available",
            "Ensure sufficient memory for transcription"
        ])
        
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'suggestions']}
        )


class EmbeddingError(ProcessingError):
    """Raised when embedding generation fails."""
    
    def __init__(self, message: str, text: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if text:
            details['text_length'] = len(text)
            details['text_preview'] = text[:100] + "..." if len(text) > 100 else text
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check if the embedding service is available",
            "Verify API credentials are valid",
            "Ensure text is within length limits"
        ])
        
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'suggestions']}
        )


# Network and External Service Errors
class NetworkError(VideoRAGError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, **kwargs):
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check internet connectivity",
            "Verify firewall settings",
            "Try again after a short delay"
        ])
        
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k != 'suggestions'}
        )


class MCPConnectionError(NetworkError):
    """Raised when MCP (Model Context Protocol) connection fails."""
    
    def __init__(self, message: str, host: Optional[str] = None, port: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if host:
            details['host'] = host
        if port:
            details['port'] = port
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check if the MCP server is running",
            "Verify host and port configuration",
            "Check authentication credentials"
        ])
        
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'suggestions']}
        )


class APIError(NetworkError):
    """Raised when external API calls fail."""
    
    def __init__(self, message: str, api_name: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if api_name:
            details['api_name'] = api_name
        if status_code:
            details['status_code'] = status_code
        
        suggestions = kwargs.get('suggestions', [])
        if status_code:
            if status_code == 401:
                suggestions.append("Check API authentication credentials")
            elif status_code == 429:
                suggestions.append("Rate limit exceeded - wait before retrying")
            elif status_code >= 500:
                suggestions.append("Server error - try again later")
        
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'suggestions']}
        )


# Validation Errors
class ValidationError(VideoRAGError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if field:
            details['field'] = field
        
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class SearchError(ValidationError):
    """Raised when search operations fail."""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if query:
            details['query'] = query
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check search query syntax",
            "Verify search indices are available",
            "Try a simpler query"
        ])
        
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'suggestions']}
        )


# Resource Errors
class ResourceError(VideoRAGError):
    """Raised when system resources are insufficient."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Free up system resources",
            "Reduce processing batch sizes",
            "Consider upgrading hardware"
        ])
        
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            details=details,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'suggestions']}
        )


class MemoryError(ResourceError):
    """Raised when insufficient memory is available."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            resource_type="memory",
            suggestions=[
                "Close other applications to free memory",
                "Reduce video processing batch size",
                "Consider processing smaller segments"
            ],
            **kwargs
        )


class DiskSpaceError(ResourceError):
    """Raised when insufficient disk space is available."""
    
    def __init__(self, message: str, path: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if path:
            details['path'] = path
        
        super().__init__(
            message,
            resource_type="disk_space",
            details=details,
            suggestions=[
                "Free up disk space",
                "Clean up temporary files",
                "Archive old processed videos"
            ],
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


# Authentication Errors
class AuthenticationError(VideoRAGError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Check authentication credentials",
                "Verify API keys are valid",
                "Ensure proper permissions are set"
            ],
            **kwargs
        )


# System Errors
class SystemError(VideoRAGError):
    """Raised when system-level errors occur."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )


class DependencyError(SystemError):
    """Raised when required dependencies are missing or incompatible."""
    
    def __init__(self, message: str, dependency: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if dependency:
            details['dependency'] = dependency
        
        suggestions = kwargs.get('suggestions', [])
        if dependency:
            suggestions.append(f"Install or update the '{dependency}' dependency")
        
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ['details', 'suggestions']}
        )


# Timeout Errors
class TimeoutError(VideoRAGError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        details = kwargs.get('details', {})
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        
        super().__init__(
            message,
            details=details,
            suggestions=[
                "Increase timeout value if appropriate",
                "Check if the operation is hanging",
                "Reduce the scope of the operation"
            ],
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


# Custom error mapping for common exception types
EXCEPTION_MAPPING = {
    "FileNotFoundError": FileSystemError,
    "PermissionError": FileSystemError,
    "OSError": SystemError,
    "ConnectionError": NetworkError,
    "TimeoutError": TimeoutError,
    "MemoryError": MemoryError,
    "ValueError": ValidationError,
    "TypeError": ValidationError,
}


def wrap_exception(exc: Exception, message: Optional[str] = None) -> VideoRAGError:
    """
    Wrap a standard Python exception in a VideoRAG exception.
    
    Args:
        exc: The original exception
        message: Optional custom message
        
    Returns:
        VideoRAGError: Wrapped exception
    """
    exc_type = type(exc).__name__
    wrapper_class = EXCEPTION_MAPPING.get(exc_type, VideoRAGError)
    
    if message is None:
        message = str(exc)
    
    return wrapper_class(
        message=message,
        details={"original_exception": exc_type, "original_message": str(exc)}
    ) 