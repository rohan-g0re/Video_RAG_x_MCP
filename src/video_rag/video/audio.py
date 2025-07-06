"""
Audio Extraction Utilities for Video RAG System.

This module provides audio extraction from video files for transcription purposes.
It supports various audio formats and preprocessing options for optimal ASR results.
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import ffmpeg
import numpy as np


class AudioError(Exception):
    """Base exception for audio extraction errors."""
    pass


class AudioExtractionError(AudioError):
    """Exception raised when audio extraction fails."""
    pass


class NoAudioStreamError(AudioError):
    """Exception raised when video has no audio streams."""
    pass


@dataclass
class AudioExtractionConfig:
    """Configuration for audio extraction."""
    # Output format
    output_format: str = "wav"  # wav, mp3, flac, aac
    sample_rate: int = 16000  # Hz (16kHz is optimal for most ASR models)
    channels: int = 1  # Mono for most ASR models
    bit_depth: int = 16  # bits per sample
    
    # Quality settings
    bitrate: Optional[str] = None  # e.g., "128k", "256k"
    codec: Optional[str] = None  # Force specific codec
    
    # Processing options
    normalize_audio: bool = True
    noise_reduction: bool = False
    high_pass_filter: float = 0  # Hz, 0 to disable
    low_pass_filter: float = 0  # Hz, 0 to disable
    
    # Segmentation
    segment_duration: Optional[float] = None  # seconds, None for no segmentation
    overlap_duration: float = 0.5  # seconds overlap between segments
    
    # Output settings
    output_dir: Optional[Path] = None
    filename_template: str = "{video_name}_audio.{ext}"
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        
        if self.channels not in [1, 2]:
            raise ValueError("Channels must be 1 (mono) or 2 (stereo)")
        
        if self.bit_depth not in [16, 24, 32]:
            raise ValueError("Bit depth must be 16, 24, or 32")
        
        if self.output_format.lower() not in ['wav', 'mp3', 'flac', 'aac', 'ogg']:
            raise ValueError("Output format must be wav, mp3, flac, aac, or ogg")


@dataclass
class AudioInfo:
    """Information about extracted audio."""
    file_path: Path
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int
    file_size: int
    format: str
    codec: str
    bitrate: Optional[int] = None
    
    # Segments (if audio was segmented)
    segments: List['AudioSegment'] = None
    
    def __post_init__(self):
        """Post-initialization."""
        if self.segments is None:
            self.segments = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "file_size": self.file_size,
            "format": self.format,
            "codec": self.codec,
            "bitrate": self.bitrate,
            "segments": [seg.to_dict() for seg in self.segments],
        }


@dataclass
class AudioSegment:
    """Information about an audio segment."""
    file_path: Path
    start_time: float
    end_time: float
    duration: float
    segment_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "segment_index": self.segment_index,
        }


class AudioExtractor:
    """Audio extraction from video files."""
    
    def __init__(self, config: Optional[AudioExtractionConfig] = None):
        """
        Initialize the audio extractor.
        
        Args:
            config: Audio extraction configuration
        """
        self.config = config or AudioExtractionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_audio(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Path] = None,
        stream_index: Optional[int] = None
    ) -> AudioInfo:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the video file
            output_path: Optional output path for the audio file
            stream_index: Specific audio stream index to extract
            
        Returns:
            AudioInfo object with extraction details
            
        Raises:
            AudioExtractionError: If extraction fails
            NoAudioStreamError: If video has no audio streams
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise AudioExtractionError(f"Video file not found: {video_path}")
        
        # Check for audio streams
        if not self._has_audio_stream(video_path):
            raise NoAudioStreamError(f"No audio streams found in: {video_path}")
        
        # Determine output path
        if output_path is None:
            if self.config.output_dir:
                output_dir = self.config.output_dir
            else:
                output_dir = video_path.parent
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = self.config.filename_template.format(
                video_name=video_path.stem,
                ext=self.config.output_format
            )
            output_path = output_dir / filename
        
        try:
            # Extract audio using FFmpeg
            input_stream = ffmpeg.input(str(video_path))
            
            # Select audio stream
            if stream_index is not None:
                audio_stream = input_stream[f'a:{stream_index}']
            else:
                audio_stream = input_stream.audio
            
            # Apply filters
            audio_stream = self._apply_audio_filters(audio_stream)
            
            # Output configuration
            output_kwargs = self._get_output_kwargs()
            
            # Execute extraction
            output_stream = ffmpeg.output(audio_stream, str(output_path), **output_kwargs)
            ffmpeg.run(output_stream, quiet=True, overwrite_output=True)
            
            # Get audio information
            audio_info = self._get_audio_info(output_path)
            
            # Segment audio if configured
            if self.config.segment_duration:
                segments = self._segment_audio(audio_info)
                audio_info.segments = segments
            
            self.logger.info(f"Extracted audio: {audio_info.duration:.2f}s, {audio_info.sample_rate}Hz, {audio_info.channels}ch")
            return audio_info
            
        except ffmpeg.Error as e:
            raise AudioExtractionError(f"FFmpeg error: {e}")
        except Exception as e:
            raise AudioExtractionError(f"Audio extraction failed: {e}")
    
    def _has_audio_stream(self, video_path: Path) -> bool:
        """Check if video file has audio streams."""
        try:
            probe = ffmpeg.probe(str(video_path))
            streams = probe.get('streams', [])
            return any(stream.get('codec_type') == 'audio' for stream in streams)
        except Exception:
            return False
    
    def _apply_audio_filters(self, audio_stream):
        """Apply audio processing filters."""
        
        # High-pass filter
        if self.config.high_pass_filter > 0:
            audio_stream = ffmpeg.filter(
                audio_stream, 
                'highpass', 
                frequency=self.config.high_pass_filter
            )
        
        # Low-pass filter
        if self.config.low_pass_filter > 0:
            audio_stream = ffmpeg.filter(
                audio_stream, 
                'lowpass', 
                frequency=self.config.low_pass_filter
            )
        
        # Noise reduction (simple noise gate)
        if self.config.noise_reduction:
            audio_stream = ffmpeg.filter(
                audio_stream,
                'afftdn',  # FFmpeg noise reduction
                nr=10,     # Noise reduction amount
                nf=-25     # Noise floor
            )
        
        # Normalize audio
        if self.config.normalize_audio:
            audio_stream = ffmpeg.filter(audio_stream, 'loudnorm')
        
        return audio_stream
    
    def _get_output_kwargs(self) -> Dict[str, Any]:
        """Get FFmpeg output arguments."""
        kwargs = {
            'ar': self.config.sample_rate,
            'ac': self.config.channels,
        }
        
        # Codec selection
        if self.config.codec:
            kwargs['acodec'] = self.config.codec
        elif self.config.output_format.lower() == 'wav':
            kwargs['acodec'] = 'pcm_s16le'
        elif self.config.output_format.lower() == 'mp3':
            kwargs['acodec'] = 'libmp3lame'
        elif self.config.output_format.lower() == 'flac':
            kwargs['acodec'] = 'flac'
        elif self.config.output_format.lower() == 'aac':
            kwargs['acodec'] = 'aac'
        
        # Bitrate
        if self.config.bitrate:
            kwargs['b:a'] = self.config.bitrate
        
        # Bit depth for PCM formats
        if self.config.output_format.lower() == 'wav':
            if self.config.bit_depth == 16:
                kwargs['acodec'] = 'pcm_s16le'
            elif self.config.bit_depth == 24:
                kwargs['acodec'] = 'pcm_s24le'
            elif self.config.bit_depth == 32:
                kwargs['acodec'] = 'pcm_s32le'
        
        return kwargs
    
    def _get_audio_info(self, audio_path: Path) -> AudioInfo:
        """Get information about the extracted audio file."""
        try:
            probe = ffmpeg.probe(str(audio_path))
            format_info = probe['format']
            audio_stream = next(
                stream for stream in probe['streams'] 
                if stream['codec_type'] == 'audio'
            )
            
            return AudioInfo(
                file_path=audio_path,
                duration=float(format_info.get('duration', 0)),
                sample_rate=int(audio_stream.get('sample_rate', 0)),
                channels=int(audio_stream.get('channels', 0)),
                bit_depth=self._get_bit_depth(audio_stream),
                file_size=int(format_info.get('size', 0)),
                format=format_info.get('format_name', 'unknown'),
                codec=audio_stream.get('codec_name', 'unknown'),
                bitrate=int(audio_stream.get('bit_rate', 0)) if audio_stream.get('bit_rate') else None,
            )
            
        except Exception as e:
            raise AudioExtractionError(f"Failed to get audio info: {e}")
    
    def _get_bit_depth(self, audio_stream: Dict[str, Any]) -> int:
        """Extract bit depth from audio stream info."""
        codec_name = audio_stream.get('codec_name', '')
        
        if 'pcm_s16' in codec_name:
            return 16
        elif 'pcm_s24' in codec_name:
            return 24
        elif 'pcm_s32' in codec_name:
            return 32
        elif 'flac' in codec_name:
            return int(audio_stream.get('bits_per_sample', 16))
        else:
            return 16  # Default assumption
    
    def _segment_audio(self, audio_info: AudioInfo) -> List[AudioSegment]:
        """Segment audio file into smaller chunks."""
        if not self.config.segment_duration:
            return []
        
        segments = []
        current_time = 0.0
        segment_index = 0
        
        while current_time < audio_info.duration:
            end_time = min(current_time + self.config.segment_duration, audio_info.duration)
            
            # Create segment filename
            segment_filename = f"{audio_info.file_path.stem}_segment_{segment_index:03d}{audio_info.file_path.suffix}"
            segment_path = audio_info.file_path.parent / segment_filename
            
            try:
                # Extract segment using FFmpeg
                input_stream = ffmpeg.input(str(audio_info.file_path), ss=current_time, t=end_time - current_time)
                output_stream = ffmpeg.output(input_stream, str(segment_path))
                ffmpeg.run(output_stream, quiet=True, overwrite_output=True)
                
                segment = AudioSegment(
                    file_path=segment_path,
                    start_time=current_time,
                    end_time=end_time,
                    duration=end_time - current_time,
                    segment_index=segment_index
                )
                
                segments.append(segment)
                
                self.logger.debug(f"Created audio segment {segment_index}: {current_time:.2f}s - {end_time:.2f}s")
                
            except Exception as e:
                self.logger.warning(f"Failed to create segment {segment_index}: {e}")
            
            # Move to next segment with overlap
            current_time += self.config.segment_duration - self.config.overlap_duration
            segment_index += 1
        
        self.logger.info(f"Created {len(segments)} audio segments")
        return segments
    
    def extract_audio_segment(
        self,
        video_path: Union[str, Path],
        start_time: float,
        duration: float,
        output_path: Optional[Path] = None
    ) -> AudioInfo:
        """
        Extract a specific audio segment from a video.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            duration: Duration in seconds
            output_path: Optional output path
            
        Returns:
            AudioInfo object for the segment
        """
        video_path = Path(video_path)
        
        if output_path is None:
            output_dir = video_path.parent
            filename = f"{video_path.stem}_segment_{start_time:.2f}s-{start_time + duration:.2f}s.{self.config.output_format}"
            output_path = output_dir / filename
        
        try:
            # Extract segment
            input_stream = ffmpeg.input(str(video_path), ss=start_time, t=duration)
            audio_stream = input_stream.audio
            
            # Apply filters
            audio_stream = self._apply_audio_filters(audio_stream)
            
            # Output
            output_kwargs = self._get_output_kwargs()
            output_stream = ffmpeg.output(audio_stream, str(output_path), **output_kwargs)
            ffmpeg.run(output_stream, quiet=True, overwrite_output=True)
            
            return self._get_audio_info(output_path)
            
        except Exception as e:
            raise AudioExtractionError(f"Failed to extract audio segment: {e}")
    
    def get_audio_stream_info(self, video_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Get information about all audio streams in a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of audio stream information dictionaries
        """
        try:
            probe = ffmpeg.probe(str(video_path))
            audio_streams = []
            
            for stream in probe.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_streams.append({
                        'index': stream.get('index'),
                        'codec': stream.get('codec_name'),
                        'sample_rate': stream.get('sample_rate'),
                        'channels': stream.get('channels'),
                        'channel_layout': stream.get('channel_layout'),
                        'duration': stream.get('duration'),
                        'bitrate': stream.get('bit_rate'),
                        'language': stream.get('tags', {}).get('language'),
                        'title': stream.get('tags', {}).get('title'),
                    })
            
            return audio_streams
            
        except Exception as e:
            self.logger.error(f"Failed to get audio stream info: {e}")
            return []
    
    def cleanup_extracted_audio(self, audio_info: AudioInfo) -> None:
        """Clean up extracted audio files."""
        try:
            # Remove main audio file
            if audio_info.file_path.exists():
                audio_info.file_path.unlink()
                self.logger.debug(f"Cleaned up audio file: {audio_info.file_path}")
            
            # Remove segments
            for segment in audio_info.segments:
                if segment.file_path.exists():
                    segment.file_path.unlink()
                    self.logger.debug(f"Cleaned up segment: {segment.file_path}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to cleanup audio files: {e}")
    
    def validate_audio_quality(self, audio_info: AudioInfo) -> Dict[str, Any]:
        """
        Validate audio quality for ASR processing.
        
        Args:
            audio_info: AudioInfo object
            
        Returns:
            Dictionary with quality assessment
        """
        quality_report = {
            'suitable_for_asr': True,
            'warnings': [],
            'recommendations': [],
        }
        
        # Check sample rate
        if audio_info.sample_rate < 8000:
            quality_report['suitable_for_asr'] = False
            quality_report['warnings'].append('Sample rate too low for good ASR results')
        elif audio_info.sample_rate < 16000:
            quality_report['warnings'].append('Sample rate below optimal for ASR (16kHz recommended)')
        
        # Check channels
        if audio_info.channels > 2:
            quality_report['warnings'].append('Multi-channel audio detected, consider converting to mono')
        
        # Check duration
        if audio_info.duration < 1.0:
            quality_report['warnings'].append('Audio segment very short, may affect ASR accuracy')
        elif audio_info.duration > 3600:  # 1 hour
            quality_report['recommendations'].append('Consider segmenting long audio for better processing')
        
        # Check file size (basic quality indicator)
        if audio_info.file_size < 1000:  # Less than 1KB
            quality_report['suitable_for_asr'] = False
            quality_report['warnings'].append('Audio file suspiciously small, may be corrupted')
        
        return quality_report 