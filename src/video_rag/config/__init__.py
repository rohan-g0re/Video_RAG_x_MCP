"""
Configuration module for MCP Video RAG System.

This module provides comprehensive configuration management with support for:
- YAML/TOML configuration files
- Environment variable substitution
- Configuration validation and schemas
- Configuration profiles for different environments
- Default value handling and merging
"""

from typing import Any

# Import configuration management classes
from .manager import (
    EnhancedConfigurationManager,
    ConfigurationError,
    ConfigurationProfile,
    ConfigurationSchema,
    DEFAULT_SCHEMA,
)

# Import base configuration manager for backward compatibility
from ..core.base import ConfigurationManager

# Export all public components
__all__ = [
    # Enhanced configuration management
    "EnhancedConfigurationManager",
    "ConfigurationError",
    "ConfigurationProfile",
    "ConfigurationSchema",
    "DEFAULT_SCHEMA",
    
    # Base configuration manager
    "ConfigurationManager",
]

# Default configuration manager instance
default_config_manager = EnhancedConfigurationManager()

# Convenience function to get the default configuration manager
def get_config_manager() -> EnhancedConfigurationManager:
    """Get the default configuration manager instance."""
    return default_config_manager

# Convenience function to load configuration from the default location
def load_default_config() -> None:
    """Load configuration from the default location."""
    default_config_manager.load_profile_config()

# Convenience function to get a configuration value
def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value using the default configuration manager."""
    return default_config_manager.get(key, default)

# Convenience function to set a configuration value
def set_config(key: str, value: Any) -> None:
    """Set a configuration value using the default configuration manager."""
    default_config_manager.set(key, value) 