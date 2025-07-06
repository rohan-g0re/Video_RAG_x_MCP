"""
Video Processing Module for Video RAG System.

This module provides comprehensive video processing capabilities including:
- Video metadata extraction (duration, resolution, codec, etc.)
- Frame extraction at specific timestamps or intervals
- Audio stream extraction for transcription
- Video format validation and conversion
- Core video processing coordination

The module integrates FFmpeg and OpenCV for robust video handling across
all major video formats (MP4, AVI, MOV, MKV, WebM, etc.).
"""

# Import video processing components
from .metadata import (
    VideoMetadataExtractor,
    VideoInfo,
    StreamInfo,
    CodecInfo,
    MetadataError,
)

# Import frame extraction utilities
from .frames import (
    FrameExtractor,
    FrameExtractionConfig,
    ExtractedFrame,
    FrameError,
)

# Import core video processor
from .processor import (
    VideoProcessor,
    ProcessingConfig,
    ProcessingResult,
    ProcessingError,
)

# Import audio extraction utilities
from .audio import (
    AudioExtractor,
    AudioInfo,
    AudioExtractionConfig,
    AudioError,
)

# Export all public components
__all__ = [
    # Metadata extraction
    "VideoMetadataExtractor",
    "VideoInfo",
    "StreamInfo", 
    "CodecInfo",
    "MetadataError",
    
    # Frame extraction
    "FrameExtractor",
    "FrameExtractionConfig",
    "ExtractedFrame",
    "FrameError",
    
    # Core processing
    "VideoProcessor",
    "ProcessingConfig",
    "ProcessingResult",
    "ProcessingError",
    
    # Audio extraction
    "AudioExtractor",
    "AudioInfo",
    "AudioExtractionConfig",
    "AudioError",
] 