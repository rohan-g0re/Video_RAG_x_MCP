"""
Video Metadata Extraction for Video RAG System.

This module provides comprehensive video metadata extraction using FFmpeg's ffprobe.
It extracts detailed information about video files including streams, codecs, 
duration, resolution, bitrates, and more.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ffmpeg


class MetadataError(Exception):
    """Base exception for metadata extraction errors."""
    pass


class VideoNotFoundError(MetadataError):
    """Exception raised when video file is not found."""
    pass


class FFmpegError(MetadataError):
    """Exception raised when FFmpeg/ffprobe fails."""
    pass


@dataclass
class CodecInfo:
    """Information about a codec."""
    name: str
    long_name: str
    type: str  # "video", "audio", "subtitle"
    profile: Optional[str] = None
    level: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "long_name": self.long_name,
            "type": self.type,
            "profile": self.profile,
            "level": self.level,
        }


@dataclass
class StreamInfo:
    """Information about a video/audio stream."""
    index: int
    codec: CodecInfo
    duration: Optional[float] = None
    bitrate: Optional[int] = None
    
    # Video-specific properties
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    aspect_ratio: Optional[str] = None
    pix_fmt: Optional[str] = None
    
    # Audio-specific properties
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    channel_layout: Optional[str] = None
    
    # Subtitle-specific properties
    language: Optional[str] = None
    title: Optional[str] = None
    
    def is_video(self) -> bool:
        """Check if this is a video stream."""
        return self.codec.type == "video"
    
    def is_audio(self) -> bool:
        """Check if this is an audio stream."""
        return self.codec.type == "audio"
    
    def is_subtitle(self) -> bool:
        """Check if this is a subtitle stream."""
        return self.codec.type == "subtitle"
    
    def get_resolution(self) -> Optional[str]:
        """Get resolution string (e.g., '1920x1080')."""
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "codec": self.codec.to_dict(),
            "duration": self.duration,
            "bitrate": self.bitrate,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "aspect_ratio": self.aspect_ratio,
            "pix_fmt": self.pix_fmt,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "channel_layout": self.channel_layout,
            "language": self.language,
            "title": self.title,
            "resolution": self.get_resolution(),
        }


@dataclass
class VideoInfo:
    """Comprehensive video file information."""
    file_path: Path
    file_size: int
    duration: float
    bitrate: int
    format_name: str
    format_long_name: str
    
    # Streams
    video_streams: List[StreamInfo]
    audio_streams: List[StreamInfo]
    subtitle_streams: List[StreamInfo]
    
    # Creation info
    creation_time: Optional[datetime] = None
    
    # Technical details
    container_format: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def primary_video_stream(self) -> Optional[StreamInfo]:
        """Get the primary video stream."""
        return self.video_streams[0] if self.video_streams else None
    
    @property
    def primary_audio_stream(self) -> Optional[StreamInfo]:
        """Get the primary audio stream."""
        return self.audio_streams[0] if self.audio_streams else None
    
    @property
    def has_video(self) -> bool:
        """Check if file has video streams."""
        return len(self.video_streams) > 0
    
    @property
    def has_audio(self) -> bool:
        """Check if file has audio streams."""
        return len(self.audio_streams) > 0
    
    @property
    def has_subtitles(self) -> bool:
        """Check if file has subtitle streams."""
        return len(self.subtitle_streams) > 0
    
    def get_duration_formatted(self) -> str:
        """Get formatted duration string (HH:MM:SS)."""
        td = timedelta(seconds=self.duration)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    def get_resolution(self) -> Optional[str]:
        """Get video resolution string."""
        if self.primary_video_stream:
            return self.primary_video_stream.get_resolution()
        return None
    
    def get_fps(self) -> Optional[float]:
        """Get video frame rate."""
        if self.primary_video_stream:
            return self.primary_video_stream.fps
        return None
    
    def is_supported_format(self) -> bool:
        """Check if video format is supported for processing."""
        supported_formats = {
            'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 
            'm4v', 'wmv', 'mpg', 'mpeg', '3gp', 'asf'
        }
        return self.format_name.lower() in supported_formats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "duration": self.duration,
            "duration_formatted": self.get_duration_formatted(),
            "bitrate": self.bitrate,
            "format_name": self.format_name,
            "format_long_name": self.format_long_name,
            "video_streams": [stream.to_dict() for stream in self.video_streams],
            "audio_streams": [stream.to_dict() for stream in self.audio_streams],
            "subtitle_streams": [stream.to_dict() for stream in self.subtitle_streams],
            "creation_time": self.creation_time.isoformat() if self.creation_time else None,
            "container_format": self.container_format,
            "metadata": self.metadata,
            "resolution": self.get_resolution(),
            "fps": self.get_fps(),
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "has_subtitles": self.has_subtitles,
            "is_supported": self.is_supported_format(),
        }


class VideoMetadataExtractor:
    """Extractor for video file metadata using FFmpeg."""
    
    def __init__(self, ffprobe_path: Optional[str] = None):
        """
        Initialize the metadata extractor.
        
        Args:
            ffprobe_path: Custom path to ffprobe executable
        """
        self.ffprobe_path = ffprobe_path or "ffprobe"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Verify ffprobe is available
        self._verify_ffprobe()
    
    def _verify_ffprobe(self) -> None:
        """Verify that ffprobe is available."""
        try:
            result = subprocess.run(
                [self.ffprobe_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise FFmpegError("ffprobe is not working correctly")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise FFmpegError(
                "ffprobe not found. Please install FFmpeg and ensure it's in your PATH."
            )
    
    def extract_metadata(self, video_path: Union[str, Path]) -> VideoInfo:
        """
        Extract comprehensive metadata from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoInfo object with comprehensive metadata
            
        Raises:
            VideoNotFoundError: If video file doesn't exist
            FFmpegError: If ffprobe fails to process the file
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise VideoNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Use ffmpeg-python to probe the file
            probe_data = ffmpeg.probe(str(video_path))
            
            # Extract format information
            format_info = probe_data.get("format", {})
            streams_info = probe_data.get("streams", [])
            
            # Parse streams
            video_streams = []
            audio_streams = []
            subtitle_streams = []
            
            for stream_data in streams_info:
                stream = self._parse_stream(stream_data)
                
                if stream.is_video():
                    video_streams.append(stream)
                elif stream.is_audio():
                    audio_streams.append(stream)
                elif stream.is_subtitle():
                    subtitle_streams.append(stream)
            
            # Extract creation time
            creation_time = None
            if "tags" in format_info and "creation_time" in format_info["tags"]:
                try:
                    creation_time = datetime.fromisoformat(
                        format_info["tags"]["creation_time"].replace("Z", "+00:00")
                    )
                except (ValueError, KeyError):
                    pass
            
            # Create VideoInfo object
            video_info = VideoInfo(
                file_path=video_path,
                file_size=int(format_info.get("size", 0)),
                duration=float(format_info.get("duration", 0)),
                bitrate=int(format_info.get("bit_rate", 0)),
                format_name=format_info.get("format_name", "unknown"),
                format_long_name=format_info.get("format_long_name", "unknown"),
                video_streams=video_streams,
                audio_streams=audio_streams,
                subtitle_streams=subtitle_streams,
                creation_time=creation_time,
                container_format=format_info.get("format_name"),
                metadata=format_info.get("tags", {}),
            )
            
            self.logger.info(f"Extracted metadata for {video_path}: {video_info.get_resolution()}, {video_info.get_duration_formatted()}")
            return video_info
            
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg error processing {video_path}: {e}"
            self.logger.error(error_msg)
            raise FFmpegError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error extracting metadata from {video_path}: {e}"
            self.logger.error(error_msg)
            raise MetadataError(error_msg)
    
    def _parse_stream(self, stream_data: Dict[str, Any]) -> StreamInfo:
        """Parse stream data from ffprobe output."""
        codec_type = stream_data.get("codec_type", "unknown")
        
        # Parse codec information
        codec = CodecInfo(
            name=stream_data.get("codec_name", "unknown"),
            long_name=stream_data.get("codec_long_name", "unknown"),
            type=codec_type,
            profile=stream_data.get("profile"),
            level=stream_data.get("level"),
        )
        
        # Parse duration
        duration = None
        if "duration" in stream_data:
            try:
                duration = float(stream_data["duration"])
            except (ValueError, TypeError):
                pass
        
        # Parse bitrate
        bitrate = None
        if "bit_rate" in stream_data:
            try:
                bitrate = int(stream_data["bit_rate"])
            except (ValueError, TypeError):
                pass
        
        # Parse video-specific properties
        width = stream_data.get("width")
        height = stream_data.get("height")
        
        fps = None
        if "r_frame_rate" in stream_data:
            try:
                fps_str = stream_data["r_frame_rate"]
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else None
                else:
                    fps = float(fps_str)
            except (ValueError, ZeroDivisionError):
                pass
        
        # Parse audio-specific properties
        sample_rate = None
        if "sample_rate" in stream_data:
            try:
                sample_rate = int(stream_data["sample_rate"])
            except (ValueError, TypeError):
                pass
        
        channels = stream_data.get("channels")
        channel_layout = stream_data.get("channel_layout")
        
        # Parse tags
        tags = stream_data.get("tags", {})
        language = tags.get("language")
        title = tags.get("title")
        
        return StreamInfo(
            index=stream_data.get("index", 0),
            codec=codec,
            duration=duration,
            bitrate=bitrate,
            width=width,
            height=height,
            fps=fps,
            aspect_ratio=stream_data.get("display_aspect_ratio"),
            pix_fmt=stream_data.get("pix_fmt"),
            sample_rate=sample_rate,
            channels=channels,
            channel_layout=channel_layout,
            language=language,
            title=title,
        )
    
    def get_basic_info(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic video information quickly.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with basic video information
        """
        try:
            info = self.extract_metadata(video_path)
            return {
                "duration": info.duration,
                "resolution": info.get_resolution(),
                "fps": info.get_fps(),
                "format": info.format_name,
                "has_audio": info.has_audio,
                "file_size": info.file_size,
            }
        except Exception as e:
            self.logger.error(f"Failed to get basic info for {video_path}: {e}")
            return {}
    
    def validate_video_file(self, video_path: Union[str, Path]) -> bool:
        """
        Validate if a file is a proper video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if file is a valid video, False otherwise
        """
        try:
            info = self.extract_metadata(video_path)
            return info.has_video and info.duration > 0
        except Exception:
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported video formats."""
        return [
            'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv',
            'm4v', 'wmv', 'mpg', 'mpeg', '3gp', 'asf',
            'ogv', 'ts', 'mts', 'm2ts', 'vob', 'rm',
            'rmvb', 'divx', 'xvid'
        ] 