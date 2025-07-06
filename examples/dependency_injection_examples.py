"""
Examples demonstrating the MCP Video RAG Dependency Injection System.

This module provides comprehensive examples of how to use the dependency injection
system for various scenarios including basic usage, testing, and advanced patterns.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

# Import the dependency injection system
from src.video_rag.core import (
    EnhancedDependencyContainer,
    ServiceLifetime,
    ServiceScope,
    MockContainer,
    ServiceDiscovery,
    service,
    injectable,
)


# Example 1: Basic Interface and Implementation
class IEmailService(ABC):
    """Interface for email services."""
    
    @abstractmethod
    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send an email."""
        pass


class IUserRepository(ABC):
    """Interface for user data access."""
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[dict]:
        """Get user by ID."""
        pass


class INotificationService(ABC):
    """Interface for notification services."""
    
    @abstractmethod
    async def notify_user(self, user_id: str, message: str) -> bool:
        """Notify a user."""
        pass


# Example 2: Concrete Implementations with Constructor Injection
@injectable
class SMTPEmailService(IEmailService):
    """SMTP email service implementation."""
    
    def __init__(self, smtp_server: str = "localhost", port: int = 587):
        self.smtp_server = smtp_server
        self.port = port
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email via SMTP."""
        self.logger.info(f"Sending email to {to}: {subject}")
        # Simulate email sending
        await asyncio.sleep(0.1)
        return True


@injectable
class DatabaseUserRepository(IUserRepository):
    """Database user repository implementation."""
    
    def __init__(self, connection_string: str = "sqlite:///users.db"):
        self.connection_string = connection_string
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def get_user(self, user_id: str) -> Optional[dict]:
        """Get user from database."""
        self.logger.info(f"Getting user {user_id} from database")
        # Simulate database query
        await asyncio.sleep(0.05)
        return {"id": user_id, "name": f"User {user_id}", "email": f"user{user_id}@example.com"}


@service(lifetime=ServiceLifetime.SINGLETON)
class EmailNotificationService(INotificationService):
    """Email-based notification service with dependency injection."""
    
    def __init__(self, email_service: IEmailService, user_repository: IUserRepository):
        self.email_service = email_service
        self.user_repository = user_repository
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def notify_user(self, user_id: str, message: str) -> bool:
        """Send notification via email."""
        user = await self.user_repository.get_user(user_id)
        if not user:
            self.logger.warning(f"User {user_id} not found")
            return False
        
        return await self.email_service.send_email(
            to=user["email"],
            subject="Notification",
            body=message
        )


# Example 3: Factory Pattern
def create_email_service(smtp_server: str, port: int) -> IEmailService:
    """Factory function for creating email services."""
    return SMTPEmailService(smtp_server, port)


# Example 4: Testing with Mock Container
class MockEmailService(IEmailService):
    """Mock email service for testing."""
    
    def __init__(self):
        self.sent_emails = []
    
    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Mock email sending."""
        self.sent_emails.append({"to": to, "subject": subject, "body": body})
        return True


class MockUserRepository(IUserRepository):
    """Mock user repository for testing."""
    
    def __init__(self):
        self.users = {
            "123": {"id": "123", "name": "Test User", "email": "test@example.com"}
        }
    
    async def get_user(self, user_id: str) -> Optional[dict]:
        """Mock user retrieval."""
        return self.users.get(user_id)


async def example_basic_usage():
    """Example 1: Basic dependency injection usage."""
    print("=== Example 1: Basic Usage ===")
    
    # Create container
    container = EnhancedDependencyContainer()
    
    # Register services
    container.register_transient(IEmailService, SMTPEmailService)
    container.register_transient(IUserRepository, DatabaseUserRepository)
    container.register_singleton(INotificationService, EmailNotificationService)
    
    # Resolve and use services
    notification_service = container.resolve(INotificationService)
    result = await notification_service.notify_user("123", "Hello from DI!")
    
    print(f"Notification sent: {result}")
    
    # Verify singleton behavior
    notification_service2 = container.resolve(INotificationService)
    print(f"Same instance: {notification_service is notification_service2}")


async def example_factory_pattern():
    """Example 2: Factory pattern usage."""
    print("\n=== Example 2: Factory Pattern ===")
    
    container = EnhancedDependencyContainer()
    
    # Register with factory
    container.register_transient(
        IEmailService,
        factory=lambda: create_email_service("smtp.gmail.com", 587)
    )
    
    email_service = container.resolve(IEmailService)
    result = await email_service.send_email("test@example.com", "Test", "Hello!")
    print(f"Email sent via factory: {result}")


async def example_scoped_services():
    """Example 3: Scoped services usage."""
    print("\n=== Example 3: Scoped Services ===")
    
    container = EnhancedDependencyContainer()
    
    # Register scoped service
    container.register_scoped(IUserRepository, DatabaseUserRepository)
    
    # Create scope
    scope = container.create_scope()
    
    # Resolve services within scope
    repo1 = container.resolve(IUserRepository, scope)
    repo2 = container.resolve(IUserRepository, scope)
    
    print(f"Same instance within scope: {repo1 is repo2}")
    
    # Create new scope
    scope2 = container.create_scope()
    repo3 = container.resolve(IUserRepository, scope2)
    
    print(f"Different instance in new scope: {repo1 is repo3}")
    
    # Dispose scopes
    scope.dispose()
    scope2.dispose()


async def example_testing_with_mocks():
    """Example 4: Testing with mock container."""
    print("\n=== Example 4: Testing with Mocks ===")
    
    # Create mock container
    mock_container = MockContainer()
    
    # Register normal services
    mock_container.register_transient(IEmailService, SMTPEmailService)
    mock_container.register_transient(IUserRepository, DatabaseUserRepository)
    mock_container.register_singleton(INotificationService, EmailNotificationService)
    
    # Override with mocks
    mock_email_service = MockEmailService()
    mock_user_repo = MockUserRepository()
    
    mock_container.override(IEmailService, mock_email_service)
    mock_container.override(IUserRepository, mock_user_repo)
    
    # Use service with mocks
    notification_service = mock_container.resolve(INotificationService)
    result = await notification_service.notify_user("123", "Test notification")
    
    print(f"Notification sent (mocked): {result}")
    print(f"Mock emails sent: {len(mock_email_service.sent_emails)}")
    print(f"Email details: {mock_email_service.sent_emails[0]}")


def example_service_discovery():
    """Example 5: Service discovery and auto-registration."""
    print("\n=== Example 5: Service Discovery ===")
    
    container = EnhancedDependencyContainer()
    
    # Manual registration with metadata
    container.register_singleton(
        IEmailService, 
        SMTPEmailService,
        tags={"communication", "email"}
    )
    
    container.register_transient(
        IUserRepository,
        DatabaseUserRepository,
        tags={"storage", "database"}
    )
    
    # Get services by tag
    communication_services = container.get_services_by_tag("communication")
    storage_services = container.get_services_by_tag("storage")
    
    print(f"Communication services: {len(communication_services)}")
    print(f"Storage services: {len(storage_services)}")
    
    # Get service information
    info = container.get_service_info()
    print(f"Total services: {info['total_services']}")
    print(f"Singletons: {info['singletons']}")
    
    for service_name, service_info in info['services'].items():
        print(f"  {service_name}: {service_info['lifetime']}, tags: {service_info['tags']}")


def example_advanced_patterns():
    """Example 6: Advanced dependency injection patterns."""
    print("\n=== Example 6: Advanced Patterns ===")
    
    container = EnhancedDependencyContainer()
    
    # Conditional registration
    use_smtp = True
    
    if use_smtp:
        container.register_transient(IEmailService, SMTPEmailService)
    else:
        # Could register a different implementation
        pass
    
    # Multiple implementations with tags
    container.register_transient(
        IEmailService,
        SMTPEmailService,
        tags={"primary", "smtp"}
    )
    
    # Register with instance
    email_instance = SMTPEmailService("custom.smtp.com", 465)
    container.register_singleton(
        IEmailService,
        instance=email_instance,
        tags={"custom", "preconfigured"}
    )
    
    # The container will use the last registration
    service = container.resolve(IEmailService)
    print(f"Resolved service: {service}")


async def example_error_handling():
    """Example 7: Error handling and debugging."""
    print("\n=== Example 7: Error Handling ===")
    
    container = EnhancedDependencyContainer()
    
    # Try to resolve unregistered service
    try:
        service = container.resolve(IEmailService)
    except Exception as e:
        print(f"Expected error: {e}")
    
    # Create circular dependency
    class IServiceA(ABC):
        pass
    
    class IServiceB(ABC):
        pass
    
    class ServiceA(IServiceA):
        def __init__(self, service_b: IServiceB):
            self.service_b = service_b
    
    class ServiceB(IServiceB):
        def __init__(self, service_a: IServiceA):
            self.service_a = service_a
    
    container.register_transient(IServiceA, ServiceA)
    container.register_transient(IServiceB, ServiceB)
    
    try:
        service_a = container.resolve(IServiceA)
    except Exception as e:
        print(f"Circular dependency error: {e}")


async def run_all_examples():
    """Run all dependency injection examples."""
    print("MCP Video RAG Dependency Injection Examples")
    print("=" * 50)
    
    await example_basic_usage()
    await example_factory_pattern()
    await example_scoped_services()
    await example_testing_with_mocks()
    example_service_discovery()
    example_advanced_patterns()
    await example_error_handling()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    asyncio.run(run_all_examples()) 