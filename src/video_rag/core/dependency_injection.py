"""
Enhanced Dependency Injection System for MCP Video RAG.

This module provides a comprehensive dependency injection framework that supports
constructor injection, factory patterns, lifecycle management, and testing utilities.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Type, TypeVar, Union, Callable, 
    Set, Protocol, runtime_checkable
)
from functools import wraps
from threading import Lock
from collections import defaultdict

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime management options."""
    TRANSIENT = "transient"      # New instance every time
    SINGLETON = "singleton"      # Single instance for application
    SCOPED = "scoped"           # Single instance per scope


class ServiceRegistrationError(Exception):
    """Exception raised when service registration fails."""
    pass


class ServiceResolutionError(Exception):
    """Exception raised when service resolution fails."""
    pass


class CircularDependencyError(ServiceResolutionError):
    """Exception raised when circular dependency is detected."""
    pass


@runtime_checkable
class IServiceScope(Protocol):
    """Protocol for service scopes."""
    
    def dispose(self) -> None:
        """Dispose of all scoped services."""
        ...


@dataclass
class ServiceDescriptor:
    """Describes a registered service."""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    tags: Set[str] = field(default_factory=set)
    dependencies: List[Type] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.implementation_type and not self.factory and not self.instance:
            raise ServiceRegistrationError(
                f"Service {self.service_type} must have implementation, factory, or instance"
            )


class ServiceRegistry:
    """Registry for managing service descriptors."""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._tags: Dict[str, Set[Type]] = defaultdict(set)
        self._lock = Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register(
        self,
        service_type: Type,
        implementation_type: Optional[Type] = None,
        factory: Optional[Callable] = None,
        instance: Optional[Any] = None,
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
        tags: Optional[Set[str]] = None
    ) -> None:
        """Register a service with the registry."""
        with self._lock:
            if service_type in self._services:
                self.logger.warning(f"Overriding existing registration for {service_type}")
            
            # Auto-detect dependencies from constructor
            dependencies = []
            if implementation_type:
                dependencies = self._extract_dependencies(implementation_type)
            elif factory:
                dependencies = self._extract_dependencies(factory)
            
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type,
                factory=factory,
                instance=instance,
                lifetime=lifetime,
                tags=tags or set(),
                dependencies=dependencies
            )
            
            self._services[service_type] = descriptor
            
            # Update tag index
            for tag in descriptor.tags:
                self._tags[tag].add(service_type)
            
            self.logger.debug(f"Registered {service_type} with lifetime {lifetime}")
    
    def get_descriptor(self, service_type: Type) -> Optional[ServiceDescriptor]:
        """Get service descriptor by type."""
        return self._services.get(service_type)
    
    def get_services_by_tag(self, tag: str) -> List[Type]:
        """Get all services with specific tag."""
        return list(self._tags.get(tag, set()))
    
    def get_all_services(self) -> Dict[Type, ServiceDescriptor]:
        """Get all registered services."""
        return self._services.copy()
    
    def remove(self, service_type: Type) -> bool:
        """Remove a service registration."""
        with self._lock:
            if service_type in self._services:
                descriptor = self._services.pop(service_type)
                # Remove from tag index
                for tag in descriptor.tags:
                    self._tags[tag].discard(service_type)
                return True
            return False
    
    def _extract_dependencies(self, target: Union[Type, Callable]) -> List[Type]:
        """Extract dependencies from constructor or factory function."""
        try:
            sig = inspect.signature(target)
            dependencies = []
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                if param.annotation != inspect.Parameter.empty:
                    # Check if it's a class type
                    if inspect.isclass(param.annotation):
                        dependencies.append(param.annotation)
                    # Handle typing annotations
                    elif hasattr(param.annotation, '__origin__'):
                        # For Optional[Type], get the inner type
                        if param.annotation.__origin__ is Union:
                            args = param.annotation.__args__
                            if len(args) == 2 and type(None) in args:
                                non_none_type = args[0] if args[1] is type(None) else args[1]
                                if inspect.isclass(non_none_type):
                                    dependencies.append(non_none_type)
            
            return dependencies
        except Exception as e:
            self.logger.warning(f"Failed to extract dependencies from {target}: {e}")
            return []


class ServiceScope:
    """Service scope for managing scoped service lifetimes."""
    
    def __init__(self, parent_container: 'EnhancedDependencyContainer'):
        self._parent = parent_container
        self._scoped_services: Dict[Type, Any] = {}
        self._disposed = False
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_or_create_scoped_service(self, service_type: Type) -> Any:
        """Get or create a scoped service instance."""
        if self._disposed:
            raise ServiceResolutionError("Cannot resolve service from disposed scope")
        
        if service_type not in self._scoped_services:
            self._scoped_services[service_type] = self._parent._create_instance(service_type)
        
        return self._scoped_services[service_type]
    
    def dispose(self) -> None:
        """Dispose of all scoped services."""
        if self._disposed:
            return
        
        for service_type, instance in self._scoped_services.items():
            try:
                if hasattr(instance, 'dispose'):
                    instance.dispose()
                elif hasattr(instance, '__exit__'):
                    instance.__exit__(None, None, None)
            except Exception as e:
                self.logger.warning(f"Error disposing {service_type}: {e}")
        
        self._scoped_services.clear()
        self._disposed = True
        self.logger.debug("Service scope disposed")


class EnhancedDependencyContainer:
    """Enhanced dependency injection container with advanced features."""
    
    def __init__(self):
        self._registry = ServiceRegistry()
        self._singletons: Dict[Type, Any] = {}
        self._resolution_stack: List[Type] = []
        self._lock = Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_transient(
        self, 
        service_type: Type, 
        implementation_type: Optional[Type] = None,
        factory: Optional[Callable] = None,
        tags: Optional[Set[str]] = None
    ) -> None:
        """Register a transient service (new instance each time)."""
        self._registry.register(
            service_type, 
            implementation_type, 
            factory, 
            lifetime=ServiceLifetime.TRANSIENT,
            tags=tags
        )
    
    def register_singleton(
        self, 
        service_type: Type, 
        implementation_type: Optional[Type] = None,
        factory: Optional[Callable] = None,
        instance: Optional[Any] = None,
        tags: Optional[Set[str]] = None
    ) -> None:
        """Register a singleton service (single instance)."""
        self._registry.register(
            service_type, 
            implementation_type, 
            factory, 
            instance,
            lifetime=ServiceLifetime.SINGLETON,
            tags=tags
        )
    
    def register_scoped(
        self, 
        service_type: Type, 
        implementation_type: Optional[Type] = None,
        factory: Optional[Callable] = None,
        tags: Optional[Set[str]] = None
    ) -> None:
        """Register a scoped service (single instance per scope)."""
        self._registry.register(
            service_type, 
            implementation_type, 
            factory, 
            lifetime=ServiceLifetime.SCOPED,
            tags=tags
        )
    
    def resolve(self, service_type: Type, scope: Optional[ServiceScope] = None) -> Any:
        """Resolve a service instance."""
        with self._lock:
            return self._resolve_internal(service_type, scope)
    
    def _resolve_internal(self, service_type: Type, scope: Optional[ServiceScope] = None) -> Any:
        """Internal resolution method with circular dependency detection."""
        if service_type in self._resolution_stack:
            cycle = " -> ".join([t.__name__ for t in self._resolution_stack]) + f" -> {service_type.__name__}"
            raise CircularDependencyError(f"Circular dependency detected: {cycle}")
        
        self._resolution_stack.append(service_type)
        
        try:
            descriptor = self._registry.get_descriptor(service_type)
            if not descriptor:
                raise ServiceResolutionError(f"Service {service_type} not registered")
            
            # Handle different lifetimes
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                return self._get_or_create_singleton(service_type, descriptor)
            elif descriptor.lifetime == ServiceLifetime.SCOPED:
                if not scope:
                    raise ServiceResolutionError(f"Scoped service {service_type} requires a scope")
                return scope.get_or_create_scoped_service(service_type)
            else:  # TRANSIENT
                return self._create_instance(service_type, descriptor)
        
        finally:
            self._resolution_stack.pop()
    
    def _get_or_create_singleton(self, service_type: Type, descriptor: ServiceDescriptor) -> Any:
        """Get or create singleton instance."""
        if service_type not in self._singletons:
            self._singletons[service_type] = self._create_instance(service_type, descriptor)
        return self._singletons[service_type]
    
    def _create_instance(self, service_type: Type, descriptor: Optional[ServiceDescriptor] = None) -> Any:
        """Create a new instance of the service."""
        if not descriptor:
            descriptor = self._registry.get_descriptor(service_type)
            if not descriptor:
                raise ServiceResolutionError(f"Service {service_type} not registered")
        
        # Use existing instance if provided
        if descriptor.instance is not None:
            return descriptor.instance
        
        # Use factory if provided
        if descriptor.factory:
            return self._invoke_factory(descriptor.factory)
        
        # Use implementation type
        if descriptor.implementation_type:
            return self._construct_instance(descriptor.implementation_type)
        
        raise ServiceResolutionError(f"No way to create instance of {service_type}")
    
    def _construct_instance(self, implementation_type: Type) -> Any:
        """Construct instance using constructor injection."""
        try:
            sig = inspect.signature(implementation_type.__init__)
            kwargs = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation != inspect.Parameter.empty:
                    try:
                        # Try to resolve the dependency
                        dependency = self._resolve_internal(param.annotation)
                        kwargs[param_name] = dependency
                    except ServiceResolutionError:
                        # If dependency is optional and can't be resolved, skip it
                        if param.default != inspect.Parameter.empty:
                            continue
                        # If dependency is Optional[Type], pass None
                        if hasattr(param.annotation, '__origin__') and param.annotation.__origin__ is Union:
                            args = param.annotation.__args__
                            if len(args) == 2 and type(None) in args:
                                kwargs[param_name] = None
                                continue
                        raise
            
            return implementation_type(**kwargs)
        
        except Exception as e:
            raise ServiceResolutionError(f"Failed to construct {implementation_type}: {e}")
    
    def _invoke_factory(self, factory: Callable) -> Any:
        """Invoke factory function with dependency injection."""
        try:
            sig = inspect.signature(factory)
            kwargs = {}
            
            for param_name, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    try:
                        dependency = self._resolve_internal(param.annotation)
                        kwargs[param_name] = dependency
                    except ServiceResolutionError:
                        if param.default != inspect.Parameter.empty:
                            continue
                        raise
            
            return factory(**kwargs)
        
        except Exception as e:
            raise ServiceResolutionError(f"Failed to invoke factory {factory}: {e}")
    
    def create_scope(self) -> ServiceScope:
        """Create a new service scope."""
        return ServiceScope(self)
    
    def get_services_by_tag(self, tag: str) -> List[Any]:
        """Get all services with specific tag."""
        service_types = self._registry.get_services_by_tag(tag)
        return [self.resolve(service_type) for service_type in service_types]
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about all registered services."""
        info = {
            "total_services": len(self._registry.get_all_services()),
            "singletons": len(self._singletons),
            "services": {}
        }
        
        for service_type, descriptor in self._registry.get_all_services().items():
            info["services"][service_type.__name__] = {
                "lifetime": descriptor.lifetime.value,
                "implementation": descriptor.implementation_type.__name__ if descriptor.implementation_type else None,
                "has_factory": descriptor.factory is not None,
                "has_instance": descriptor.instance is not None,
                "tags": list(descriptor.tags),
                "dependencies": [dep.__name__ for dep in descriptor.dependencies]
            }
        
        return info


# Testing utilities
class MockContainer(EnhancedDependencyContainer):
    """Mock container for testing with override capabilities."""
    
    def __init__(self):
        super().__init__()
        self._overrides: Dict[Type, Any] = {}
    
    def override(self, service_type: Type, mock_instance: Any) -> None:
        """Override a service with a mock instance."""
        self._overrides[service_type] = mock_instance
    
    def resolve(self, service_type: Type, scope: Optional[ServiceScope] = None) -> Any:
        """Resolve service with override support."""
        if service_type in self._overrides:
            return self._overrides[service_type]
        return super().resolve(service_type, scope)
    
    def clear_overrides(self) -> None:
        """Clear all overrides."""
        self._overrides.clear()


# Decorators for service registration
def service(
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
    tags: Optional[Set[str]] = None
):
    """Decorator for automatic service registration."""
    def decorator(cls):
        # Store service metadata on the class
        cls._service_lifetime = lifetime
        cls._service_tags = tags or set()
        return cls
    return decorator


def injectable(cls):
    """Decorator to mark a class as injectable."""
    cls._injectable = True
    return cls


# Service discovery utilities
class ServiceDiscovery:
    """Utilities for automatic service discovery and registration."""
    
    @staticmethod
    def auto_register_services(
        container: EnhancedDependencyContainer,
        module_names: List[str]
    ) -> None:
        """Automatically register services from modules."""
        import importlib
        
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
                ServiceDiscovery._register_from_module(container, module)
            except ImportError as e:
                logging.warning(f"Failed to import module {module_name}: {e}")
    
    @staticmethod
    def _register_from_module(container: EnhancedDependencyContainer, module: Any) -> None:
        """Register services from a module."""
        for name in dir(module):
            obj = getattr(module, name)
            
            if inspect.isclass(obj) and hasattr(obj, '_service_lifetime'):
                # Auto-register based on interfaces
                interfaces = [
                    base for base in obj.__bases__ 
                    if hasattr(base, '__abstractmethods__') and base.__abstractmethods__
                ]
                
                for interface in interfaces:
                    if obj._service_lifetime == ServiceLifetime.SINGLETON:
                        container.register_singleton(interface, obj, tags=obj._service_tags)
                    elif obj._service_lifetime == ServiceLifetime.SCOPED:
                        container.register_scoped(interface, obj, tags=obj._service_tags)
                    else:
                        container.register_transient(interface, obj, tags=obj._service_tags) 