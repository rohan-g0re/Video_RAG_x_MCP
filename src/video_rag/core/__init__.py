"""
Core module for MCP Video RAG System.

This module contains the foundational classes and interfaces for the video RAG system,
including abstract interfaces, base implementations, and the main system engine.
"""

# Import interfaces
from .interfaces import (
    # Core interfaces
    IProcessor,
    IVideoProcessor,
    ITranscriptionService,
    IVisualAnalyzer,
    IMCPBridge,
    IVectorStore,
    IStorageManager,
    ISearchEngine,
    IRAGGenerator,
    IConfigurationManager,
    ILoggingManager,
    IDependencyContainer,
    
    # Enums and types
    ProcessingState,
    ProcessorFactory,
    ComponentRegistry,
    ConfigDict,
)

# Import base implementations
from .base import (
    BaseProcessor,
    ConfigurationManager,
    LoggingManager,
    DependencyContainer,
    BaseComponent,
    VideoRAGEngine,
)

# Import exceptions
from .exceptions import (
    VideoRAGError,
    ErrorSeverity,
    ErrorCategory,
    ConfigurationError,
    StorageError,
    ProcessingError,
    NetworkError,
    ValidationError,
    ResourceError,
    AuthenticationError,
    SystemError,
    wrap_exception,
)

# Import enhanced logging
from .logging import (
    EnhancedLoggingManager,
    VideoRAGLoggerAdapter,
    PerformanceLogger,
    LogFormat,
    LogLevel,
    get_logging_manager,
    get_logger,
)

# Import dependency injection
from .dependency_injection import (
    EnhancedDependencyContainer,
    ServiceLifetime,
    ServiceScope,
    ServiceRegistry,
    ServiceDescriptor,
    MockContainer,
    ServiceDiscovery,
    ServiceRegistrationError,
    ServiceResolutionError,
    CircularDependencyError,
    service,
    injectable,
)

# Export all public components
__all__ = [
    # Interfaces
    "IProcessor",
    "IVideoProcessor", 
    "ITranscriptionService",
    "IVisualAnalyzer",
    "IMCPBridge",
    "IVectorStore",
    "IStorageManager",
    "ISearchEngine",
    "IRAGGenerator",
    "IConfigurationManager",
    "ILoggingManager",
    "IDependencyContainer",
    
    # Base implementations
    "BaseProcessor",
    "ConfigurationManager",
    "LoggingManager",
    "DependencyContainer",
    "BaseComponent",
    "VideoRAGEngine",
    
    # Exceptions
    "VideoRAGError",
    "ErrorSeverity",
    "ErrorCategory",
    "ConfigurationError",
    "StorageError",
    "ProcessingError",
    "NetworkError",
    "ValidationError",
    "ResourceError",
    "AuthenticationError",
    "SystemError",
    "wrap_exception",
    
    # Enhanced logging
    "EnhancedLoggingManager",
    "VideoRAGLoggerAdapter",
    "PerformanceLogger",
    "LogFormat",
    "LogLevel",
    "get_logging_manager",
    "get_logger",
    
    # Enums and types
    "ProcessingState",
    "ProcessorFactory",
    "ComponentRegistry",
    "ConfigDict",
    
    # Dependency injection
    "EnhancedDependencyContainer",
    "ServiceLifetime",
    "ServiceScope",
    "ServiceRegistry",
    "ServiceDescriptor",
    "MockContainer",
    "ServiceDiscovery",
    "ServiceRegistrationError",
    "ServiceResolutionError",
    "CircularDependencyError",
    "service",
    "injectable",
] 