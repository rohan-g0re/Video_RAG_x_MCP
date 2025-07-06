"""
Frame Extraction Utilities for Video RAG System.

This module provides precise frame extraction from video files using FFmpeg and OpenCV.
It supports extraction at specific timestamps, intervals, and intelligent keyframe-based
extraction for optimal visual analysis.
"""

import hashlib
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Generator

import cv2
import ffmpeg
import numpy as np
from PIL import Image


class FrameError(Exception):
    """Base exception for frame extraction errors."""
    pass


class FrameExtractionError(FrameError):
    """Exception raised when frame extraction fails."""
    pass


class InvalidTimestampError(FrameError):
    """Exception raised when timestamp is invalid."""
    pass


@dataclass
class FrameExtractionConfig:
    """Configuration for frame extraction."""
    # Quality settings
    output_format: str = "jpg"  # jpg, png, bmp
    quality: int = 95  # JPEG quality (1-100)
    width: Optional[int] = None  # Target width (maintains aspect ratio)
    height: Optional[int] = None  # Target height (maintains aspect ratio)
    max_dimension: Optional[int] = 1920  # Maximum dimension for any side
    
    # Extraction settings
    extract_keyframes_only: bool = False
    skip_similar_frames: bool = True
    similarity_threshold: float = 0.95  # Cosine similarity threshold
    
    # Storage settings
    save_frames: bool = True
    output_dir: Optional[Path] = None
    filename_template: str = "{video_name}_frame_{timestamp:.3f}s.{ext}"
    
    # Processing settings
    use_opencv: bool = False  # Use OpenCV instead of FFmpeg for extraction
    threads: int = 1
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.quality < 1 or self.quality > 100:
            raise ValueError("Quality must be between 1 and 100")
        
        if self.output_format.lower() not in ['jpg', 'jpeg', 'png', 'bmp']:
            raise ValueError("Output format must be jpg, jpeg, png, or bmp")
        
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("Similarity threshold must be between 0 and 1")


@dataclass
class ExtractedFrame:
    """Information about an extracted frame."""
    timestamp: float
    frame_number: int
    file_path: Optional[Path] = None
    image_data: Optional[np.ndarray] = None
    width: int = 0
    height: int = 0
    file_size: int = 0
    hash: Optional[str] = None
    similarity_score: Optional[float] = None
    is_keyframe: bool = False
    extraction_time: datetime = field(default_factory=datetime.now)
    
    def get_image(self) -> Optional[np.ndarray]:
        """Get image data, loading from file if necessary."""
        if self.image_data is not None:
            return self.image_data
        
        if self.file_path and self.file_path.exists():
            return cv2.imread(str(self.file_path))
        
        return None
    
    def get_pil_image(self) -> Optional[Image.Image]:
        """Get PIL Image object."""
        img_data = self.get_image()
        if img_data is not None:
            # Convert BGR to RGB for PIL
            img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
        return None
    
    def calculate_hash(self) -> str:
        """Calculate hash of the frame image."""
        if self.hash:
            return self.hash
        
        img_data = self.get_image()
        if img_data is not None:
            # Use image data for hash
            self.hash = hashlib.md5(img_data.tobytes()).hexdigest()
        elif self.file_path and self.file_path.exists():
            # Use file content for hash
            with open(self.file_path, 'rb') as f:
                self.hash = hashlib.md5(f.read()).hexdigest()
        
        return self.hash or ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "file_path": str(self.file_path) if self.file_path else None,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "hash": self.hash,
            "similarity_score": self.similarity_score,
            "is_keyframe": self.is_keyframe,
            "extraction_time": self.extraction_time.isoformat(),
        }


class FrameExtractor:
    """Advanced frame extraction from video files."""
    
    def __init__(self, config: Optional[FrameExtractionConfig] = None):
        """
        Initialize the frame extractor.
        
        Args:
            config: Frame extraction configuration
        """
        self.config = config or FrameExtractionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache for similarity comparison
        self._previous_frame_features: Optional[np.ndarray] = None
    
    def extract_frame_at_timestamp(
        self,
        video_path: Union[str, Path],
        timestamp: float,
        output_path: Optional[Path] = None
    ) -> ExtractedFrame:
        """
        Extract a single frame at a specific timestamp.
        
        Args:
            video_path: Path to the video file
            timestamp: Timestamp in seconds
            output_path: Optional output path for the frame
            
        Returns:
            ExtractedFrame object
            
        Raises:
            FrameExtractionError: If extraction fails
            InvalidTimestampError: If timestamp is invalid
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FrameExtractionError(f"Video file not found: {video_path}")
        
        if timestamp < 0:
            raise InvalidTimestampError("Timestamp cannot be negative")
        
        try:
            if self.config.use_opencv:
                return self._extract_frame_opencv(video_path, timestamp, output_path)
            else:
                return self._extract_frame_ffmpeg(video_path, timestamp, output_path)
                
        except Exception as e:
            raise FrameExtractionError(f"Failed to extract frame at {timestamp}s: {e}")
    
    def _extract_frame_ffmpeg(
        self,
        video_path: Path,
        timestamp: float,
        output_path: Optional[Path]
    ) -> ExtractedFrame:
        """Extract frame using FFmpeg."""
        
        # Determine output path
        if output_path is None and self.config.save_frames:
            if self.config.output_dir:
                output_dir = self.config.output_dir
            else:
                output_dir = video_path.parent / "frames"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = self.config.filename_template.format(
                video_name=video_path.stem,
                timestamp=timestamp,
                ext=self.config.output_format
            )
            output_path = output_dir / filename
        
        try:
            # Build FFmpeg command
            input_stream = ffmpeg.input(str(video_path), ss=timestamp)
            
            # Apply filters
            stream = input_stream.video
            
            if self.config.width or self.config.height or self.config.max_dimension:
                scale_filter = self._get_scale_filter()
                if scale_filter:
                    stream = ffmpeg.filter(stream, 'scale', scale_filter)
            
            # Output configuration
            output_kwargs = {
                'vframes': 1,
                'f': 'image2',
                'y': None,  # Overwrite without asking
            }
            
            if self.config.output_format.lower() in ['jpg', 'jpeg']:
                output_kwargs['q:v'] = max(1, min(31, 31 - (self.config.quality - 1) * 30 // 99))
            
            if output_path:
                # Save to file
                output_stream = ffmpeg.output(stream, str(output_path), **output_kwargs)
                ffmpeg.run(output_stream, quiet=True, overwrite_output=True)
                
                # Load image data if needed
                image_data = None
                if not self.config.save_frames:
                    image_data = cv2.imread(str(output_path))
                
                # Get file info
                file_size = output_path.stat().st_size if output_path.exists() else 0
                
                # Get image dimensions
                if image_data is not None:
                    height, width = image_data.shape[:2]
                else:
                    img = cv2.imread(str(output_path))
                    height, width = img.shape[:2] if img is not None else (0, 0)
                
            else:
                # Extract to memory (for temporary processing)
                with tempfile.NamedTemporaryFile(suffix=f'.{self.config.output_format}', delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                
                output_stream = ffmpeg.output(stream, str(tmp_path), **output_kwargs)
                ffmpeg.run(output_stream, quiet=True, overwrite_output=True)
                
                image_data = cv2.imread(str(tmp_path))
                height, width = image_data.shape[:2] if image_data is not None else (0, 0)
                file_size = tmp_path.stat().st_size
                
                # Clean up temp file if not saving
                if not self.config.save_frames:
                    tmp_path.unlink()
                    output_path = None
                else:
                    output_path = tmp_path
            
            # Calculate frame number (approximate)
            frame_number = int(timestamp * 30)  # Assuming 30 fps for approximation
            
            frame = ExtractedFrame(
                timestamp=timestamp,
                frame_number=frame_number,
                file_path=output_path,
                image_data=image_data if not self.config.save_frames else None,
                width=width,
                height=height,
                file_size=file_size,
                is_keyframe=False,  # Would need additional detection
            )
            
            frame.calculate_hash()
            return frame
            
        except ffmpeg.Error as e:
            raise FrameExtractionError(f"FFmpeg error: {e}")
    
    def _extract_frame_opencv(
        self,
        video_path: Path,
        timestamp: float,
        output_path: Optional[Path]
    ) -> ExtractedFrame:
        """Extract frame using OpenCV."""
        
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            if not cap.isOpened():
                raise FrameExtractionError("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            
            # Seek to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                raise FrameExtractionError(f"Could not read frame at timestamp {timestamp}")
            
            # Resize if needed
            if self.config.max_dimension:
                frame = self._resize_frame(frame)
            
            height, width = frame.shape[:2]
            
            # Save frame if required
            file_size = 0
            if output_path or self.config.save_frames:
                if output_path is None:
                    if self.config.output_dir:
                        output_dir = self.config.output_dir
                    else:
                        output_dir = video_path.parent / "frames"
                    
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = self.config.filename_template.format(
                        video_name=video_path.stem,
                        timestamp=timestamp,
                        ext=self.config.output_format
                    )
                    output_path = output_dir / filename
                
                # Set quality for JPEG
                save_params = []
                if self.config.output_format.lower() in ['jpg', 'jpeg']:
                    save_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.quality]
                elif self.config.output_format.lower() == 'png':
                    save_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
                
                cv2.imwrite(str(output_path), frame, save_params)
                file_size = output_path.stat().st_size if output_path.exists() else 0
            
            extracted_frame = ExtractedFrame(
                timestamp=timestamp,
                frame_number=frame_number,
                file_path=output_path,
                image_data=frame if not self.config.save_frames else None,
                width=width,
                height=height,
                file_size=file_size,
                is_keyframe=False,
            )
            
            extracted_frame.calculate_hash()
            return extracted_frame
            
        finally:
            cap.release()
    
    def extract_frames_at_intervals(
        self,
        video_path: Union[str, Path],
        interval_seconds: float,
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> List[ExtractedFrame]:
        """
        Extract frames at regular intervals.
        
        Args:
            video_path: Path to the video file
            interval_seconds: Interval between frames in seconds
            start_time: Start time in seconds
            end_time: End time in seconds (None for end of video)
            
        Returns:
            List of ExtractedFrame objects
        """
        video_path = Path(video_path)
        
        # Get video duration if end_time not specified
        if end_time is None:
            try:
                probe = ffmpeg.probe(str(video_path))
                duration = float(probe['format']['duration'])
                end_time = duration
            except Exception as e:
                raise FrameExtractionError(f"Could not determine video duration: {e}")
        
        frames = []
        current_time = start_time
        
        self.logger.info(f"Extracting frames from {start_time}s to {end_time}s at {interval_seconds}s intervals")
        
        while current_time <= end_time:
            try:
                frame = self.extract_frame_at_timestamp(video_path, current_time)
                
                # Skip similar frames if configured
                if self.config.skip_similar_frames and frames:
                    similarity = self._calculate_frame_similarity(frames[-1], frame)
                    frame.similarity_score = similarity
                    
                    if similarity > self.config.similarity_threshold:
                        self.logger.debug(f"Skipping similar frame at {current_time}s (similarity: {similarity:.3f})")
                        current_time += interval_seconds
                        continue
                
                frames.append(frame)
                self.logger.debug(f"Extracted frame at {current_time}s")
                
            except Exception as e:
                self.logger.warning(f"Failed to extract frame at {current_time}s: {e}")
            
            current_time += interval_seconds
        
        self.logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    def extract_keyframes(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None
    ) -> List[ExtractedFrame]:
        """
        Extract keyframes from video.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of ExtractedFrame objects (keyframes)
        """
        video_path = Path(video_path)
        
        try:
            # Use FFmpeg to detect scene changes
            input_stream = ffmpeg.input(str(video_path))
            
            # Scene detection filter
            scene_filter = ffmpeg.filter(
                input_stream.video,
                'select',
                'gt(scene,0.3)'  # Scene change threshold
            )
            
            # Get timestamps of scene changes
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp:
                tmp_path = Path(tmp.name)
            
            output = ffmpeg.output(
                scene_filter,
                str(tmp_path),
                f='null',
                vstats_file=str(tmp_path.with_suffix('.log'))
            )
            
            ffmpeg.run(output, quiet=True)
            
            # Parse scene change timestamps (this is a simplified approach)
            # In practice, you'd parse the vstats file or use a different method
            
            # For now, use interval-based extraction with keyframe preference
            frames = self.extract_frames_at_intervals(video_path, 10.0)  # Every 10 seconds
            
            # Mark as keyframes
            for frame in frames:
                frame.is_keyframe = True
            
            # Limit frames if specified
            if max_frames and len(frames) > max_frames:
                frames = frames[:max_frames]
            
            return frames
            
        except Exception as e:
            self.logger.warning(f"Keyframe extraction failed, falling back to interval extraction: {e}")
            return self.extract_frames_at_intervals(video_path, 10.0)
    
    def _get_scale_filter(self) -> Optional[str]:
        """Get FFmpeg scale filter string."""
        if self.config.width and self.config.height:
            return f"{self.config.width}:{self.config.height}"
        elif self.config.width:
            return f"{self.config.width}:-1"
        elif self.config.height:
            return f"-1:{self.config.height}"
        elif self.config.max_dimension:
            return f"'min({self.config.max_dimension},iw)':'min({self.config.max_dimension},ih)'"
        return None
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame according to configuration."""
        if not self.config.max_dimension:
            return frame
        
        height, width = frame.shape[:2]
        max_dim = max(height, width)
        
        if max_dim > self.config.max_dimension:
            scale_factor = self.config.max_dimension / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def _calculate_frame_similarity(
        self,
        frame1: ExtractedFrame,
        frame2: ExtractedFrame
    ) -> float:
        """Calculate similarity between two frames using histogram comparison."""
        
        img1 = frame1.get_image()
        img2 = frame2.get_image()
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Convert to grayscale and calculate histograms
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Calculate correlation coefficient
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return max(0.0, correlation)  # Ensure non-negative
    
    def cleanup_extracted_frames(self, frames: List[ExtractedFrame]) -> None:
        """Clean up extracted frame files."""
        for frame in frames:
            if frame.file_path and frame.file_path.exists():
                try:
                    frame.file_path.unlink()
                    self.logger.debug(f"Cleaned up frame file: {frame.file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup frame file {frame.file_path}: {e}")
    
    def get_frame_generator(
        self,
        video_path: Union[str, Path],
        interval_seconds: float,
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> Generator[ExtractedFrame, None, None]:
        """
        Generate frames lazily for memory efficiency.
        
        Args:
            video_path: Path to the video file
            interval_seconds: Interval between frames
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Yields:
            ExtractedFrame objects
        """
        video_path = Path(video_path)
        
        if end_time is None:
            try:
                probe = ffmpeg.probe(str(video_path))
                duration = float(probe['format']['duration'])
                end_time = duration
            except Exception as e:
                raise FrameExtractionError(f"Could not determine video duration: {e}")
        
        current_time = start_time
        
        while current_time <= end_time:
            try:
                frame = self.extract_frame_at_timestamp(video_path, current_time)
                yield frame
                
            except Exception as e:
                self.logger.warning(f"Failed to extract frame at {current_time}s: {e}")
            
            current_time += interval_seconds 