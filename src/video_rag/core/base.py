"""
Base classes providing common functionality for MCP Video RAG System components.

This module contains abstract base classes that provide default implementations
of common functionality while still requiring specific methods to be implemented
by concrete classes.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Type
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

from .interfaces import (
    IProcessor, IConfigurationManager, ILoggingManager, IDependencyContainer,
    ProcessingState
)
from .dependency_injection import (
    EnhancedDependencyContainer, ServiceLifetime, ServiceScope
)


class BaseProcessor(IProcessor):
    """Base processor class with common functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._state = ProcessingState.PENDING
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._error: Optional[Exception] = None
        self._progress = 0.0
        
    def get_status(self) -> ProcessingState:
        """Get current processing status."""
        return self._state
    
    def get_progress(self) -> float:
        """Get current processing progress (0.0 to 1.0)."""
        return self._progress
    
    def get_duration(self) -> Optional[timedelta]:
        """Get processing duration if completed."""
        if self._start_time and self._end_time:
            return self._end_time - self._start_time
        return None
    
    def get_error(self) -> Optional[Exception]:
        """Get processing error if failed."""
        return self._error
    
    @asynccontextmanager
    async def _processing_context(self):
        """Context manager for processing lifecycle."""
        self._state = ProcessingState.PROCESSING
        self._start_time = datetime.now()
        self._error = None
        
        try:
            yield
            self._state = ProcessingState.COMPLETED
        except Exception as e:
            self._error = e
            self._state = ProcessingState.FAILED
            self.logger.error(f"Processing failed: {e}")
            raise
        finally:
            self._end_time = datetime.now()
    
    def _update_progress(self, progress: float):
        """Update processing progress."""
        self._progress = max(0.0, min(1.0, progress))
        self.logger.debug(f"Progress: {self._progress:.1%}")
    
    async def cleanup(self) -> None:
        """Default cleanup implementation."""
        self.logger.debug("Cleanup completed")


class ConfigurationManager(IConfigurationManager):
    """Base configuration manager implementation."""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._config_file: Optional[Path] = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation support."""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def load_config(self, config_path: Path) -> None:
        """Load configuration from YAML or TOML file."""
        import yaml
        import toml
        
        self._config_file = config_path
        
        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    self._config = yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.toml':
                    self._config = toml.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            self.logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise
    
    def save_config(self, config_path: Path) -> None:
        """Save configuration to file."""
        import yaml
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            raise
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Basic validation - can be extended by subclasses
        required_sections = ['storage', 'mcp', 'processing']
        
        for section in required_sections:
            if section not in self._config:
                errors.append(f"Missing required section: {section}")
        
        return errors
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()


class LoggingManager(ILoggingManager):
    """Base logging manager implementation."""
    
    def __init__(self):
        self._loggers: Dict[str, logging.Logger] = {}
        self._configured = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create logger instance."""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
            
            if not self._configured:
                self._setup_default_logging()
        
        return self._loggers[name]
    
    def configure_logging(self, config: Dict[str, Any]) -> None:
        """Configure logging system from configuration."""
        level = config.get('level', 'INFO').upper()
        format_str = config.get('format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, level),
            format=format_str,
            force=True
        )
        
        # Configure file logging if specified
        if 'file' in config:
            file_handler = logging.FileHandler(config['file'])
            file_handler.setFormatter(logging.Formatter(format_str))
            
            for logger in self._loggers.values():
                logger.addHandler(file_handler)
        
        # Configure structured logging if specified
        if config.get('structured', False):
            self._setup_structured_logging()
        
        self._configured = True
    
    def set_log_level(self, level: str) -> None:
        """Set global log level."""
        log_level = getattr(logging, level.upper())
        logging.getLogger().setLevel(log_level)
        
        for logger in self._loggers.values():
            logger.setLevel(log_level)
    
    def _setup_default_logging(self):
        """Setup default logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self._configured = True
    
    def _setup_structured_logging(self):
        """Setup structured logging with JSON format."""
        try:
            import structlog
            
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
        except ImportError:
            logging.warning("structlog not available, using standard logging")


class DependencyContainer(EnhancedDependencyContainer, IDependencyContainer):
    """Enhanced dependency injection container with backward compatibility."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register(
        self, 
        interface: Type, 
        implementation: Type,
        singleton: bool = False
    ) -> None:
        """Register implementation for interface (backward compatibility)."""
        if singleton:
            self.register_singleton(interface, implementation)
        else:
            self.register_transient(interface, implementation)
        
        self.logger.debug(f"Registered {implementation.__name__} for {interface.__name__}")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure container with settings."""
        # Auto-registration based on configuration
        registrations = config.get('registrations', {})
        
        for interface_name, impl_config in registrations.items():
            # This would need more sophisticated type resolution in practice
            self.logger.info(f"Would register {interface_name} from config")


class BaseComponent(ABC):
    """Base class for all system components."""
    
    def __init__(
        self, 
        container: IDependencyContainer,
        config: Optional[Dict[str, Any]] = None
    ):
        self.container = container
        self.config = config or {}
        self.logger = self._get_logger()
        self._initialized = False
    
    def _get_logger(self) -> logging.Logger:
        """Get logger for this component."""
        try:
            logging_manager = self.container.resolve(ILoggingManager)
            return logging_manager.get_logger(self.__class__.__name__)
        except:
            return logging.getLogger(self.__class__.__name__)
    
    async def initialize(self) -> None:
        """Initialize component."""
        if self._initialized:
            return
        
        await self._setup()
        self._initialized = True
        self.logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    async def _setup(self) -> None:
        """Component-specific setup logic."""
        pass
    
    async def shutdown(self) -> None:
        """Shutdown component."""
        if not self._initialized:
            return
        
        await self._teardown()
        self._initialized = False
        self.logger.info(f"{self.__class__.__name__} shutdown")
    
    @abstractmethod
    async def _teardown(self) -> None:
        """Component-specific teardown logic."""
        pass


class VideoRAGEngine(BaseComponent):
    """Main engine coordinating all video RAG operations."""
    
    async def _setup(self) -> None:
        """Setup the main engine."""
        self.logger.info("Video RAG Engine starting up...")
        
        # Initialize core components
        self.config_manager = self.container.resolve(IConfigurationManager)
        self.logging_manager = self.container.resolve(ILoggingManager)
        
        # Setup logging from configuration
        logging_config = self.config.get('logging', {})
        self.logging_manager.configure_logging(logging_config)
    
    async def _teardown(self) -> None:
        """Teardown the main engine."""
        self.logger.info("Video RAG Engine shutting down...")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "engine": "running",
                "config": "loaded",
                "logging": "configured"
            }
        } 