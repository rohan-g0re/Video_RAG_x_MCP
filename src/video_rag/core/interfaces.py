"""
Core interfaces and abstract base classes for the MCP Video RAG System.

This module defines the fundamental interfaces that all components must implement
to ensure consistency and enable dependency injection throughout the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Generator, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum

from ..models.base import (
    VideoMetadata, ProcessedVideo, SearchResult, ProcessingStatus,
    EmbeddingVector, TranscriptionSegment, VisualDescription
)


class ProcessingState(Enum):
    """Enumeration of possible processing states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IProcessor(ABC):
    """Base interface for all processing components."""
    
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> Any:
        """Process input data and return results."""
        pass
    
    @abstractmethod
    def get_status(self) -> ProcessingState:
        """Get current processing status."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources after processing."""
        pass


class IVideoProcessor(IProcessor):
    """Interface for video processing components."""
    
    @abstractmethod
    async def extract_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from video file."""
        pass
    
    @abstractmethod
    async def extract_frames(
        self, 
        video_path: Path, 
        interval: float = 1.0,
        quality: str = "medium"
    ) -> List[Path]:
        """Extract frames from video at specified interval."""
        pass
    
    @abstractmethod
    async def extract_audio(self, video_path: Path) -> Path:
        """Extract audio track from video."""
        pass
    
    @abstractmethod
    async def validate_video(self, video_path: Path) -> bool:
        """Validate video file format and integrity."""
        pass


class ITranscriptionService(IProcessor):
    """Interface for audio transcription services."""
    
    @abstractmethod
    async def transcribe(
        self, 
        audio_path: Path,
        language: Optional[str] = None
    ) -> List[TranscriptionSegment]:
        """Transcribe audio file to text with timestamps."""
        pass
    
    @abstractmethod
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        pass


class IVisualAnalyzer(IProcessor):
    """Interface for visual content analysis."""
    
    @abstractmethod
    async def analyze_frames(
        self, 
        frame_paths: List[Path],
        context: Optional[str] = None
    ) -> List[VisualDescription]:
        """Analyze visual content in frames."""
        pass
    
    @abstractmethod
    async def detect_scenes(self, video_path: Path) -> List[Dict[str, Any]]:
        """Detect scene changes in video."""
        pass


class IMCPBridge(ABC):
    """Interface for MCP (Model Context Protocol) bridge."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to MCP service."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close MCP connection."""
        pass
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text using MCP service."""
        pass
    
    @abstractmethod
    async def analyze_image(
        self, 
        image_path: Path, 
        prompt: str
    ) -> str:
        """Analyze image using vision model via MCP."""
        pass
    
    @abstractmethod
    async def get_embeddings(self, text: str) -> EmbeddingVector:
        """Generate embeddings for text via MCP."""
        pass


class IVectorStore(ABC):
    """Interface for vector storage and retrieval."""
    
    @abstractmethod
    async def add_embeddings(
        self, 
        embeddings: List[EmbeddingVector],
        metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """Add embeddings to the vector store."""
        pass
    
    @abstractmethod
    async def search_similar(
        self, 
        query_embedding: EmbeddingVector,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """Search for similar embeddings."""
        pass
    
    @abstractmethod
    async def delete_embeddings(self, ids: List[str]) -> bool:
        """Delete embeddings by IDs."""
        pass
    
    @abstractmethod
    async def update_metadata(
        self, 
        embedding_id: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for an embedding."""
        pass


class IStorageManager(ABC):
    """Interface for storage management."""
    
    @abstractmethod
    async def store_video(self, video_path: Path) -> str:
        """Store video file and return storage ID."""
        pass
    
    @abstractmethod
    async def retrieve_video(self, storage_id: str) -> Optional[Path]:
        """Retrieve video file by storage ID."""
        pass
    
    @abstractmethod
    async def delete_video(self, storage_id: str) -> bool:
        """Delete video file by storage ID."""
        pass
    
    @abstractmethod
    async def store_metadata(self, metadata: VideoMetadata) -> str:
        """Store video metadata and return ID."""
        pass
    
    @abstractmethod
    async def get_metadata(self, metadata_id: str) -> Optional[VideoMetadata]:
        """Retrieve metadata by ID."""
        pass
    
    @abstractmethod
    async def cleanup_temp_files(self, older_than: datetime) -> int:
        """Clean up temporary files older than specified date."""
        pass


class ISearchEngine(ABC):
    """Interface for search functionality."""
    
    @abstractmethod
    async def search(
        self, 
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """Perform search query."""
        pass
    
    @abstractmethod
    async def index_content(self, content: ProcessedVideo) -> bool:
        """Index processed video content."""
        pass
    
    @abstractmethod
    async def remove_from_index(self, video_id: str) -> bool:
        """Remove content from search index."""
        pass


class IRAGGenerator(ABC):
    """Interface for RAG (Retrieval-Augmented Generation) functionality."""
    
    @abstractmethod
    async def generate_answer(
        self, 
        question: str,
        context: List[SearchResult],
        video_id: Optional[str] = None
    ) -> str:
        """Generate answer based on retrieved context."""
        pass
    
    @abstractmethod
    async def extract_clips(
        self, 
        query: str,
        video_path: Path,
        context: List[SearchResult]
    ) -> List[Path]:
        """Extract video clips relevant to query."""
        pass


class IConfigurationManager(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass
    
    @abstractmethod
    def load_config(self, config_path: Path) -> None:
        """Load configuration from file."""
        pass
    
    @abstractmethod
    def save_config(self, config_path: Path) -> None:
        """Save configuration to file."""
        pass
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        pass


class ILoggingManager(ABC):
    """Interface for logging management."""
    
    @abstractmethod
    def get_logger(self, name: str) -> Any:
        """Get logger instance."""
        pass
    
    @abstractmethod
    def configure_logging(self, config: Dict[str, Any]) -> None:
        """Configure logging system."""
        pass
    
    @abstractmethod
    def set_log_level(self, level: str) -> None:
        """Set global log level."""
        pass


class IDependencyContainer(ABC):
    """Interface for dependency injection container."""
    
    @abstractmethod
    def register(
        self, 
        interface: type, 
        implementation: type,
        singleton: bool = False
    ) -> None:
        """Register implementation for interface."""
        pass
    
    @abstractmethod
    def resolve(self, interface: type) -> Any:
        """Resolve implementation for interface."""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure container with settings."""
        pass


# Type aliases for common patterns
ProcessorFactory = Callable[[], IProcessor]
ComponentRegistry = Dict[str, type]
ConfigDict = Dict[str, Any] 