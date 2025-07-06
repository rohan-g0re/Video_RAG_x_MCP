"""
Enhanced Configuration Management System for MCP Video RAG System.

This module provides a comprehensive configuration management system that supports:
- YAML/TOML configuration files
- Environment variable substitution
- Configuration validation
- Configuration profiles (development, production, etc.)
- Default value handling
- Configuration merging and inheritance
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import re

try:
    import yaml
except ImportError:
    yaml = None

try:
    import toml
except ImportError:
    toml = None

from ..core.interfaces import IConfigurationManager


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigurationProfile(Enum):
    """Configuration profiles for different environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ConfigurationSchema:
    """Schema definition for configuration validation."""
    
    # Required configuration keys
    required_keys: List[str] = field(default_factory=list)
    
    # Optional configuration keys with default values
    optional_keys: Dict[str, Any] = field(default_factory=dict)
    
    # Type specifications for configuration values
    type_specs: Dict[str, type] = field(default_factory=dict)
    
    # Value constraints (min/max for numbers, allowed values for strings)
    constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Nested schema definitions
    nested_schemas: Dict[str, 'ConfigurationSchema'] = field(default_factory=dict)


class EnhancedConfigurationManager(IConfigurationManager):
    """Enhanced configuration manager with validation and environment support."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self._config: Dict[str, Any] = {}
        self._base_path = base_path or Path(".")
        self._profile = ConfigurationProfile.DEVELOPMENT
        self._env_prefix = "VIDEO_RAG_"
        self._config_files: List[Path] = []
        self._schema: Optional[ConfigurationSchema] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load default configuration
        self._load_defaults()
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        defaults = {
            "system": {
                "name": "MCP Video RAG System",
                "version": "1.0.0",
                "profile": self._profile.value
            },
            "storage": {
                "database": {
                    "type": "sqlite",
                    "path": "data/video_rag.db"
                },
                "files": {
                    "video_storage_path": "data/videos",
                    "temp_storage_path": "data/temp"
                }
            },
            "logging": {
                "level": "INFO",
                "console": {"enabled": True},
                "file": {"enabled": True, "path": "logs/video_rag.log"}
            }
        }
        
        self._config = defaults
        
    def set_profile(self, profile: Union[ConfigurationProfile, str]) -> None:
        """Set the configuration profile."""
        if isinstance(profile, str):
            profile = ConfigurationProfile(profile)
        
        self._profile = profile
        self._config["system"]["profile"] = profile.value
        self.logger.info(f"Configuration profile set to: {profile.value}")
    
    def get_profile(self) -> ConfigurationProfile:
        """Get the current configuration profile."""
        return self._profile
    
    def set_environment_prefix(self, prefix: str) -> None:
        """Set the environment variable prefix."""
        self._env_prefix = prefix.upper()
        if not self._env_prefix.endswith('_'):
            self._env_prefix += '_'
    
    def load_config(self, config_path: Path) -> None:
        """Load configuration from file with environment variable substitution."""
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Substitute environment variables
            content = self._substitute_environment_variables(content)
            
            # Parse configuration based on file extension
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                if yaml is None:
                    raise ConfigurationError("PyYAML is required for YAML configuration files")
                loaded_config = yaml.safe_load(content)
            elif config_path.suffix.lower() == '.toml':
                if toml is None:
                    raise ConfigurationError("toml is required for TOML configuration files")
                loaded_config = toml.loads(content)
            else:
                raise ConfigurationError(f"Unsupported configuration format: {config_path.suffix}")
            
            # Merge with existing configuration
            self._config = self._merge_configs(self._config, loaded_config or {})
            self._config_files.append(config_path)
            
            self.logger.info(f"Configuration loaded from: {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}")
    
    def load_profile_config(self, profile: Optional[ConfigurationProfile] = None) -> None:
        """Load configuration files for a specific profile."""
        if profile:
            self.set_profile(profile)
        
        profile_name = self._profile.value
        
        # Load base configuration
        base_config = self._base_path / "config" / "default.yaml"
        if base_config.exists():
            self.load_config(base_config)
        
        # Load profile-specific configuration
        profile_config = self._base_path / "config" / f"{profile_name}.yaml"
        if profile_config.exists():
            self.load_config(profile_config)
        
        # Load local configuration (typically git-ignored)
        local_config = self._base_path / "config" / "local.yaml"
        if local_config.exists():
            self.load_config(local_config)
    
    def _substitute_environment_variables(self, content: str) -> str:
        """Substitute environment variables in configuration content."""
        # Pattern to match ${ENV_VAR} or ${ENV_VAR:default_value}
        pattern = r'\$\{([^}]+)\}'
        
        def replace_env_var(match):
            var_spec = match.group(1)
            
            if ':' in var_spec:
                var_name, default_value = var_spec.split(':', 1)
            else:
                var_name = var_spec
                default_value = None
            
            # Try with and without prefix
            value = os.getenv(var_name) or os.getenv(f"{self._env_prefix}{var_name}")
            
            if value is None:
                if default_value is not None:
                    return default_value
                else:
                    self.logger.warning(f"Environment variable not found: {var_name}")
                    return match.group(0)  # Return original if no default
            
            return value
        
        return re.sub(pattern, replace_env_var, content)
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    return default
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
            elif not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def save_config(self, config_path: Path) -> None:
        """Save current configuration to file."""
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    if yaml is None:
                        raise ConfigurationError("PyYAML is required for YAML configuration files")
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.toml':
                    if toml is None:
                        raise ConfigurationError("toml is required for TOML configuration files")
                    toml.dump(self._config, f)
                else:
                    raise ConfigurationError(f"Unsupported configuration format: {config_path.suffix}")
            
            self.logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def set_schema(self, schema: ConfigurationSchema) -> None:
        """Set the configuration schema for validation."""
        self._schema = schema
    
    def validate_config(self) -> List[str]:
        """Validate configuration against schema."""
        if self._schema is None:
            return []
        
        errors = []
        errors.extend(self._validate_required_keys(self._config, self._schema))
        errors.extend(self._validate_types(self._config, self._schema))
        errors.extend(self._validate_constraints(self._config, self._schema))
        
        return errors
    
    def _validate_required_keys(
        self, 
        config: Dict[str, Any], 
        schema: ConfigurationSchema,
        path: str = ""
    ) -> List[str]:
        """Validate that all required keys are present."""
        errors = []
        
        for key in schema.required_keys:
            full_key = f"{path}.{key}" if path else key
            if key not in config:
                errors.append(f"Missing required configuration key: {full_key}")
        
        # Validate nested schemas
        for key, nested_schema in schema.nested_schemas.items():
            if key in config and isinstance(config[key], dict):
                nested_path = f"{path}.{key}" if path else key
                errors.extend(self._validate_required_keys(config[key], nested_schema, nested_path))
        
        return errors
    
    def _validate_types(
        self, 
        config: Dict[str, Any], 
        schema: ConfigurationSchema,
        path: str = ""
    ) -> List[str]:
        """Validate configuration value types."""
        errors = []
        
        for key, expected_type in schema.type_specs.items():
            if key in config:
                full_key = f"{path}.{key}" if path else key
                value = config[key]
                
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Invalid type for {full_key}: expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
        
        return errors
    
    def _validate_constraints(
        self, 
        config: Dict[str, Any], 
        schema: ConfigurationSchema,
        path: str = ""
    ) -> List[str]:
        """Validate configuration value constraints."""
        errors = []
        
        for key, constraints in schema.constraints.items():
            if key in config:
                full_key = f"{path}.{key}" if path else key
                value = config[key]
                
                # Check minimum value
                if 'min' in constraints and value < constraints['min']:
                    errors.append(f"Value for {full_key} is below minimum: {value} < {constraints['min']}")
                
                # Check maximum value
                if 'max' in constraints and value > constraints['max']:
                    errors.append(f"Value for {full_key} is above maximum: {value} > {constraints['max']}")
                
                # Check allowed values
                if 'allowed' in constraints and value not in constraints['allowed']:
                    errors.append(f"Invalid value for {full_key}: {value} not in {constraints['allowed']}")
        
        return errors
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()
    
    def get_loaded_files(self) -> List[Path]:
        """Get list of loaded configuration files."""
        return self._config_files.copy()
    
    def reload(self) -> None:
        """Reload all configuration files."""
        files_to_reload = self._config_files.copy()
        self._config_files.clear()
        
        # Reset to defaults
        self._load_defaults()
        
        # Reload all files
        for config_file in files_to_reload:
            self.load_config(config_file)
    
    def export_environment_variables(self) -> Dict[str, str]:
        """Export configuration as environment variables."""
        env_vars = {}
        
        def flatten_dict(d: Dict[str, Any], prefix: str = "") -> None:
            for key, value in d.items():
                env_key = f"{prefix}{key}".upper()
                
                if isinstance(value, dict):
                    flatten_dict(value, f"{env_key}_")
                else:
                    env_vars[env_key] = str(value)
        
        flatten_dict(self._config, self._env_prefix)
        return env_vars
    
    def print_config(self, mask_secrets: bool = True) -> None:
        """Print current configuration (for debugging)."""
        config_copy = self._config.copy()
        
        if mask_secrets:
            # Mask sensitive values
            self._mask_sensitive_values(config_copy)
        
        print("Current Configuration:")
        print("=" * 50)
        
        if yaml:
            print(yaml.dump(config_copy, default_flow_style=False, indent=2))
        else:
            import pprint
            pprint.pprint(config_copy)
    
    def _mask_sensitive_values(self, config: Dict[str, Any]) -> None:
        """Mask sensitive configuration values."""
        sensitive_keys = ['password', 'token', 'key', 'secret', 'api_key']
        
        for key, value in config.items():
            if isinstance(value, dict):
                self._mask_sensitive_values(value)
            elif any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                if isinstance(value, str) and value:
                    config[key] = "*" * len(value)


# Default configuration schema
DEFAULT_SCHEMA = ConfigurationSchema(
    required_keys=['system', 'storage', 'logging'],
    type_specs={
        'system.name': str,
        'system.version': str,
        'storage.database.type': str,
        'storage.database.path': str,
        'logging.level': str,
    },
    constraints={
        'logging.level': {
            'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        },
        'storage.database.connection_pool_size': {
            'min': 1,
            'max': 100
        }
    }
) 