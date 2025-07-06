"""
Local Storage Infrastructure for MCP Video RAG System.

This module provides comprehensive storage capabilities including:
- Local file storage with metadata management
- Storage backend abstraction for different providers
- File indexing and organization
- Storage monitoring and quota management
- Automated cleanup services
"""

# Import storage models
from .models import (
    StorageItem,
    StorageMetadata,
    StorageUsage,
    StorageConfiguration,
    CleanupPolicy,
    IndexEntry,
    StorageType,
    StorageStatus,
    CompressionType,
    CleanupAction,
    StorageStats,
)

# Import storage utilities
from .utils import (
    StorageUtils,
    FileHasher,
    StorageQuota,
    StorageMetrics,
    StorageException,
    StorageQuotaExceeded,
    StorageCorruption,
)

# Export all public components
__all__ = [
    # Storage models
    "StorageItem",
    "StorageMetadata",
    "StorageUsage",
    "StorageConfiguration",
    "CleanupPolicy",
    "IndexEntry",
    "StorageType",
    "StorageStatus",
    "CompressionType",
    "CleanupAction",
    "StorageStats",
    
    # Storage utilities
    "StorageUtils",
    "FileHasher",
    "StorageQuota",
    "StorageMetrics",
    "StorageException",
    "StorageQuotaExceeded",
    "StorageCorruption",
] 