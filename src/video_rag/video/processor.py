"""
Core Video Processor for Video RAG System.

This module provides the main video processing coordination, combining metadata
extraction, frame extraction, and audio extraction into a unified interface.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from .metadata import VideoMetadataExtractor, VideoInfo, MetadataError
from .frames import FrameExtractor, FrameExtractionConfig, ExtractedFrame, FrameError
from .audio import AudioExtractor, AudioExtractionConfig, AudioInfo, AudioError


class ProcessingError(Exception):
    """Base exception for video processing errors."""
    pass


class VideoProcessingError(ProcessingError):
    """Exception raised during video processing."""
    pass


class UnsupportedFormatError(ProcessingError):
    """Exception raised for unsupported video formats."""
    pass


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    # Processing options
    extract_metadata: bool = True
    extract_frames: bool = True
    extract_audio: bool = True
    
    # Frame extraction settings
    frame_config: Optional[FrameExtractionConfig] = None
    frame_interval: float = 5.0  # seconds
    max_frames: Optional[int] = None
    extract_keyframes: bool = False
    
    # Audio extraction settings
    audio_config: Optional[AudioExtractionConfig] = None
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Output settings
    output_dir: Optional[Path] = None
    create_subdirs: bool = True
    
    # Quality validation
    validate_inputs: bool = True
    validate_outputs: bool = True
    
    # Callbacks
    progress_callback: Optional[Callable[[str, float], None]] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.frame_config is None:
            self.frame_config = FrameExtractionConfig()
        
        if self.audio_config is None:
            self.audio_config = AudioExtractionConfig()
        
        if self.max_workers < 1:
            self.max_workers = 1


@dataclass
class ProcessingResult:
    """Result of video processing."""
    video_path: Path
    success: bool
    
    # Processing results
    metadata: Optional[VideoInfo] = None
    frames: List[ExtractedFrame] = field(default_factory=list)
    audio: Optional[AudioInfo] = None
    
    # Processing info
    processing_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization."""
        if self.end_time is None:
            self.end_time = datetime.now()
            self.processing_time = (self.end_time - self.start_time).total_seconds()
    
    @property
    def has_metadata(self) -> bool:
        """Check if metadata was extracted."""
        return self.metadata is not None
    
    @property
    def has_frames(self) -> bool:
        """Check if frames were extracted."""
        return len(self.frames) > 0
    
    @property
    def has_audio(self) -> bool:
        """Check if audio was extracted."""
        return self.audio is not None
    
    @property
    def frame_count(self) -> int:
        """Get number of extracted frames."""
        return len(self.frames)
    
    def get_frame_timestamps(self) -> List[float]:
        """Get timestamps of all extracted frames."""
        return [frame.timestamp for frame in self.frames]
    
    def get_audio_duration(self) -> float:
        """Get audio duration."""
        return self.audio.duration if self.audio else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_path": str(self.video_path),
            "success": self.success,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "frames": [frame.to_dict() for frame in self.frames],
            "audio": self.audio.to_dict() if self.audio else None,
            "processing_time": self.processing_time,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
            "frame_count": self.frame_count,
            "audio_duration": self.get_audio_duration(),
        }


class VideoProcessor:
    """Main video processor coordinating all video processing tasks."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the video processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize processors
        self.metadata_extractor = VideoMetadataExtractor()
        self.frame_extractor = FrameExtractor(self.config.frame_config)
        self.audio_extractor = AudioExtractor(self.config.audio_config)
        
        # Statistics
        self.processed_videos = 0
        self.failed_videos = 0
        self.total_processing_time = 0.0
    
    def process_video(self, video_path: Union[str, Path]) -> ProcessingResult:
        """
        Process a single video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            ProcessingResult object
        """
        video_path = Path(video_path)
        start_time = datetime.now()
        
        result = ProcessingResult(
            video_path=video_path,
            success=False,
            start_time=start_time
        )
        
        self.logger.info(f"Starting processing: {video_path}")
        
        try:
            # Validate input
            if self.config.validate_inputs:
                self._validate_input_video(video_path)
            
            # Setup output directory
            output_dir = self._setup_output_directory(video_path)
            
            # Progress tracking
            total_steps = sum([
                self.config.extract_metadata,
                self.config.extract_frames,
                self.config.extract_audio
            ])
            current_step = 0
            
            # Extract metadata
            if self.config.extract_metadata:
                try:
                    self._report_progress("Extracting metadata", current_step / total_steps)
                    result.metadata = self.metadata_extractor.extract_metadata(video_path)
                    self.logger.info(f"Metadata extracted: {result.metadata.get_resolution()}, {result.metadata.get_duration_formatted()}")
                except Exception as e:
                    error_msg = f"Metadata extraction failed: {e}"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
                
                current_step += 1
            
            # Extract frames
            if self.config.extract_frames:
                try:
                    self._report_progress("Extracting frames", current_step / total_steps)
                    result.frames = self._extract_frames(video_path, output_dir, result.metadata)
                    self.logger.info(f"Extracted {len(result.frames)} frames")
                except Exception as e:
                    error_msg = f"Frame extraction failed: {e}"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
                
                current_step += 1
            
            # Extract audio
            if self.config.extract_audio:
                try:
                    self._report_progress("Extracting audio", current_step / total_steps)
                    result.audio = self._extract_audio(video_path, output_dir, result.metadata)
                    
                    if result.audio:
                        self.logger.info(f"Audio extracted: {result.audio.duration:.2f}s")
                        
                        # Validate audio quality
                        quality_report = self.audio_extractor.validate_audio_quality(result.audio)
                        if not quality_report['suitable_for_asr']:
                            result.warnings.extend(quality_report['warnings'])
                        result.stats['audio_quality'] = quality_report
                        
                except Exception as e:
                    error_msg = f"Audio extraction failed: {e}"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
                
                current_step += 1
            
            # Finalize result
            result.success = len(result.errors) == 0
            result.end_time = datetime.now()
            result.processing_time = (result.end_time - start_time).total_seconds()
            
            # Update statistics
            result.stats.update({
                'frame_count': len(result.frames),
                'audio_duration': result.get_audio_duration(),
                'processing_time': result.processing_time,
                'output_directory': str(output_dir),
            })
            
            # Report completion
            self._report_progress("Processing complete", 1.0)
            
            if result.success:
                self.processed_videos += 1
                self.logger.info(f"Processing completed successfully in {result.processing_time:.2f}s")
            else:
                self.failed_videos += 1
                self.logger.warning(f"Processing completed with errors: {result.errors}")
            
            self.total_processing_time += result.processing_time
            
            return result
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Processing failed: {e}")
            result.end_time = datetime.now()
            result.processing_time = (result.end_time - start_time).total_seconds()
            
            self.failed_videos += 1
            self.logger.error(f"Video processing failed: {e}")
            
            return result
    
    def process_video_batch(
        self,
        video_paths: List[Union[str, Path]],
        max_workers: Optional[int] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple videos in parallel.
        
        Args:
            video_paths: List of video file paths
            max_workers: Maximum number of worker threads
            
        Returns:
            List of ProcessingResult objects
        """
        if not self.config.parallel_processing or len(video_paths) == 1:
            # Sequential processing
            return [self.process_video(path) for path in video_paths]
        
        max_workers = max_workers or self.config.max_workers
        results = []
        
        self.logger.info(f"Processing {len(video_paths)} videos with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_path = {
                executor.submit(self.process_video, path): path 
                for path in video_paths
            }
            
            # Collect results
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Completed processing: {path}")
                except Exception as e:
                    error_result = ProcessingResult(
                        video_path=Path(path),
                        success=False,
                        errors=[f"Processing failed: {e}"]
                    )
                    results.append(error_result)
                    self.logger.error(f"Failed to process {path}: {e}")
        
        # Sort results by original order
        path_to_index = {str(Path(path)): i for i, path in enumerate(video_paths)}
        results.sort(key=lambda r: path_to_index.get(str(r.video_path), 999))
        
        self.logger.info(f"Batch processing complete: {len([r for r in results if r.success])}/{len(results)} successful")
        
        return results
    
    async def process_video_async(self, video_path: Union[str, Path]) -> ProcessingResult:
        """
        Process video asynchronously.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            ProcessingResult object
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_video, video_path)
    
    def _validate_input_video(self, video_path: Path) -> None:
        """Validate input video file."""
        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
        
        if not video_path.is_file():
            raise VideoProcessingError(f"Path is not a file: {video_path}")
        
        # Basic format validation
        supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.wmv'}
        if video_path.suffix.lower() not in supported_extensions:
            raise UnsupportedFormatError(f"Unsupported video format: {video_path.suffix}")
        
        # Check file size
        file_size = video_path.stat().st_size
        if file_size == 0:
            raise VideoProcessingError(f"Video file is empty: {video_path}")
        
        if file_size < 1024:  # Less than 1KB
            raise VideoProcessingError(f"Video file suspiciously small: {video_path}")
    
    def _setup_output_directory(self, video_path: Path) -> Path:
        """Setup output directory for processed files."""
        if self.config.output_dir:
            base_dir = self.config.output_dir
        else:
            base_dir = video_path.parent
        
        if self.config.create_subdirs:
            output_dir = base_dir / f"{video_path.stem}_processed"
        else:
            output_dir = base_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update extraction configs with output directory
        if self.config.frame_config:
            self.config.frame_config.output_dir = output_dir / "frames"
        
        if self.config.audio_config:
            self.config.audio_config.output_dir = output_dir / "audio"
        
        return output_dir
    
    def _extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        metadata: Optional[VideoInfo]
    ) -> List[ExtractedFrame]:
        """Extract frames from video."""
        
        if self.config.extract_keyframes:
            # Extract keyframes
            frames = self.frame_extractor.extract_keyframes(
                video_path,
                max_frames=self.config.max_frames
            )
        else:
            # Extract at intervals
            end_time = None
            if metadata and metadata.duration > 0:
                end_time = metadata.duration
            
            frames = self.frame_extractor.extract_frames_at_intervals(
                video_path,
                interval_seconds=self.config.frame_interval,
                end_time=end_time
            )
            
            # Limit frames if specified
            if self.config.max_frames and len(frames) > self.config.max_frames:
                # Select frames evenly distributed across video
                step = len(frames) / self.config.max_frames
                selected_indices = [int(i * step) for i in range(self.config.max_frames)]
                frames = [frames[i] for i in selected_indices]
        
        return frames
    
    def _extract_audio(
        self,
        video_path: Path,
        output_dir: Path,
        metadata: Optional[VideoInfo]
    ) -> Optional[AudioInfo]:
        """Extract audio from video."""
        
        # Check if video has audio
        if metadata and not metadata.has_audio:
            self.logger.warning(f"No audio streams found in {video_path}")
            return None
        
        try:
            return self.audio_extractor.extract_audio(video_path)
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            return None
    
    def _report_progress(self, message: str, progress: float) -> None:
        """Report processing progress."""
        if self.config.progress_callback:
            try:
                self.config.progress_callback(message, progress)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'processed_videos': self.processed_videos,
            'failed_videos': self.failed_videos,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': (
                self.total_processing_time / self.processed_videos 
                if self.processed_videos > 0 else 0
            ),
            'success_rate': (
                self.processed_videos / (self.processed_videos + self.failed_videos)
                if (self.processed_videos + self.failed_videos) > 0 else 0
            ),
        }
    
    def cleanup_processing_results(self, results: List[ProcessingResult]) -> None:
        """Clean up files from processing results."""
        for result in results:
            try:
                # Clean up frames
                if result.frames:
                    self.frame_extractor.cleanup_extracted_frames(result.frames)
                
                # Clean up audio
                if result.audio:
                    self.audio_extractor.cleanup_extracted_audio(result.audio)
                    
            except Exception as e:
                self.logger.warning(f"Cleanup failed for {result.video_path}: {e}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported video formats."""
        return self.metadata_extractor.get_supported_formats()
    
    def validate_video_file(self, video_path: Union[str, Path]) -> bool:
        """Validate if a file is a proper video file."""
        try:
            self._validate_input_video(Path(video_path))
            return self.metadata_extractor.validate_video_file(video_path)
        except Exception:
            return False 