"""
Storage Models for MCP Video RAG System.

This module defines data models for storage operations including storage items,
metadata, usage tracking, and configuration management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import uuid

from ..models.base import VideoMetadata


class StorageType(Enum):
    """Storage type enumeration."""
    LOCAL_FILE = "local_file"
    CLOUD_STORAGE = "cloud_storage"
    S3_BUCKET = "s3_bucket"
    AZURE_BLOB = "azure_blob"
    GOOGLE_CLOUD = "google_cloud"


class StorageStatus(Enum):
    """Storage item status enumeration."""
    PENDING = "pending"
    STORED = "stored"
    CORRUPTED = "corrupted"
    ARCHIVED = "archived"
    DELETED = "deleted"


class CompressionType(Enum):
    """Compression type enumeration."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    LZ4 = "lz4"


class CleanupAction(Enum):
    """Cleanup action enumeration."""
    DELETE = "delete"
    ARCHIVE = "archive"
    COMPRESS = "compress"
    MOVE = "move"


@dataclass
class StorageItem:
    """Represents a stored item in the storage system."""
    storage_id: str
    original_path: Path
    stored_path: Path
    storage_type: StorageType
    content_type: str
    file_size: int
    checksum: str
    checksum_algorithm: str = "sha256"
    compression: CompressionType = CompressionType.NONE
    status: StorageStatus = StorageStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.storage_id:
            self.storage_id = str(uuid.uuid4())
        
        # Ensure paths are Path objects
        if isinstance(self.original_path, str):
            self.original_path = Path(self.original_path)
        if isinstance(self.stored_path, str):
            self.stored_path = Path(self.stored_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "storage_id": self.storage_id,
            "original_path": str(self.original_path),
            "stored_path": str(self.stored_path),
            "storage_type": self.storage_type.value,
            "content_type": self.content_type,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm,
            "compression": self.compression.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "metadata": self.metadata,
            "tags": list(self.tags),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorageItem':
        """Create from dictionary representation."""
        return cls(
            storage_id=data["storage_id"],
            original_path=Path(data["original_path"]),
            stored_path=Path(data["stored_path"]),
            storage_type=StorageType(data["storage_type"]),
            content_type=data["content_type"],
            file_size=data["file_size"],
            checksum=data["checksum"],
            checksum_algorithm=data.get("checksum_algorithm", "sha256"),
            compression=CompressionType(data.get("compression", "none")),
            status=StorageStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            last_modified=datetime.fromisoformat(data["last_modified"]),
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", [])),
        )


@dataclass
class StorageMetadata:
    """Extended metadata for storage items."""
    storage_id: str
    video_metadata: Optional[VideoMetadata] = None
    thumbnail_path: Optional[Path] = None
    preview_path: Optional[Path] = None
    extracted_frames: List[Path] = field(default_factory=list)
    audio_path: Optional[Path] = None
    transcription_path: Optional[Path] = None
    processing_info: Dict[str, Any] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "storage_id": self.storage_id,
            "video_metadata": self.video_metadata.to_dict() if self.video_metadata else None,
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "preview_path": str(self.preview_path) if self.preview_path else None,
            "extracted_frames": [str(p) for p in self.extracted_frames],
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "transcription_path": str(self.transcription_path) if self.transcription_path else None,
            "processing_info": self.processing_info,
            "custom_metadata": self.custom_metadata,
        }


@dataclass
class StorageUsage:
    """Storage usage statistics."""
    total_files: int = 0
    total_size: int = 0
    used_space: int = 0
    available_space: int = 0
    files_by_type: Dict[str, int] = field(default_factory=dict)
    size_by_type: Dict[str, int] = field(default_factory=dict)
    oldest_file: Optional[datetime] = None
    newest_file: Optional[datetime] = None
    compression_ratio: float = 0.0
    
    def utilization_percentage(self) -> float:
        """Calculate storage utilization percentage."""
        if self.total_size == 0:
            return 0.0
        return (self.used_space / self.total_size) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_files": self.total_files,
            "total_size": self.total_size,
            "used_space": self.used_space,
            "available_space": self.available_space,
            "files_by_type": self.files_by_type,
            "size_by_type": self.size_by_type,
            "oldest_file": self.oldest_file.isoformat() if self.oldest_file else None,
            "newest_file": self.newest_file.isoformat() if self.newest_file else None,
            "compression_ratio": self.compression_ratio,
            "utilization_percentage": self.utilization_percentage(),
        }


@dataclass
class StorageConfiguration:
    """Storage system configuration."""
    base_path: Path
    max_storage_size: int  # in bytes
    max_files: int = 10000
    default_compression: CompressionType = CompressionType.NONE
    checksum_algorithm: str = "sha256"
    enable_deduplication: bool = True
    enable_compression: bool = False
    enable_encryption: bool = False
    backup_enabled: bool = False
    backup_path: Optional[Path] = None
    storage_backends: List[StorageType] = field(default_factory=lambda: [StorageType.LOCAL_FILE])
    temp_directory: Optional[Path] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)
        
        if self.backup_path and isinstance(self.backup_path, str):
            self.backup_path = Path(self.backup_path)
        
        if self.temp_directory and isinstance(self.temp_directory, str):
            self.temp_directory = Path(self.temp_directory)
        
        # Set default temp directory if not provided
        if not self.temp_directory:
            self.temp_directory = self.base_path / "temp"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "base_path": str(self.base_path),
            "max_storage_size": self.max_storage_size,
            "max_files": self.max_files,
            "default_compression": self.default_compression.value,
            "checksum_algorithm": self.checksum_algorithm,
            "enable_deduplication": self.enable_deduplication,
            "enable_compression": self.enable_compression,
            "enable_encryption": self.enable_encryption,
            "backup_enabled": self.backup_enabled,
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "storage_backends": [backend.value for backend in self.storage_backends],
            "temp_directory": str(self.temp_directory) if self.temp_directory else None,
        }


@dataclass
class CleanupPolicy:
    """Cleanup policy configuration."""
    enabled: bool = True
    max_age_days: int = 30
    max_size_gb: float = 10.0
    max_files: int = 1000
    cleanup_temp_files: bool = True
    temp_file_max_age_hours: int = 24
    cleanup_actions: List[CleanupAction] = field(default_factory=lambda: [CleanupAction.DELETE])
    preserve_tags: Set[str] = field(default_factory=set)
    exclude_patterns: List[str] = field(default_factory=list)
    schedule_cron: str = "0 2 * * *"  # Daily at 2 AM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "enabled": self.enabled,
            "max_age_days": self.max_age_days,
            "max_size_gb": self.max_size_gb,
            "max_files": self.max_files,
            "cleanup_temp_files": self.cleanup_temp_files,
            "temp_file_max_age_hours": self.temp_file_max_age_hours,
            "cleanup_actions": [action.value for action in self.cleanup_actions],
            "preserve_tags": list(self.preserve_tags),
            "exclude_patterns": self.exclude_patterns,
            "schedule_cron": self.schedule_cron,
        }


@dataclass
class IndexEntry:
    """Index entry for fast file lookups."""
    storage_id: str
    file_path: Path
    content_type: str
    file_size: int
    checksum: str
    created_at: datetime
    last_accessed: datetime
    tags: Set[str] = field(default_factory=set)
    metadata_keys: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "storage_id": self.storage_id,
            "file_path": str(self.file_path),
            "content_type": self.content_type,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "tags": list(self.tags),
            "metadata_keys": list(self.metadata_keys),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexEntry':
        """Create from dictionary representation."""
        return cls(
            storage_id=data["storage_id"],
            file_path=Path(data["file_path"]),
            content_type=data["content_type"],
            file_size=data["file_size"],
            checksum=data["checksum"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            tags=set(data.get("tags", [])),
            metadata_keys=set(data.get("metadata_keys", [])),
        )


@dataclass
class StorageStats:
    """Storage statistics for monitoring."""
    timestamp: datetime
    total_files: int
    total_size: int
    average_file_size: float
    storage_utilization: float
    operations_count: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_files": self.total_files,
            "total_size": self.total_size,
            "average_file_size": self.average_file_size,
            "storage_utilization": self.storage_utilization,
            "operations_count": self.operations_count,
            "error_count": self.error_count,
        } 