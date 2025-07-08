#!/usr/bin/env python3
"""
Phase 1-A: Audio Extraction & Transcription

Extracts audio from video files and generates word-level transcripts using Whisper.
Outputs JSON with per-word timings as specified in the development plan.
"""

import os
import json
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging

import ffmpeg
import whisper
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioExtractor:
    """Handles audio extraction from video files using FFmpeg."""
    
    def __init__(self):
        self.temp_dir = None
    
    def extract_audio(self, video_path: str, output_path: str = None) -> str:
        """
        Extract audio from video file using FFmpeg.
        
        Args:
            video_path: Path to input video file
            output_path: Path to output audio file (optional)
            
        Returns:
            Path to extracted audio file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create temporary audio file if no output path provided
        if output_path is None:
            self.temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(self.temp_dir, "temp_audio.wav")
        
        try:
            logger.info(f"Extracting audio from {video_path}")
            
            # Extract audio using FFmpeg with specific format for Whisper
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(
                stream, 
                output_path,
                vn=None,  # No video
                acodec='pcm_s16le',  # 16-bit PCM encoding
                ar='16000',  # 16kHz sample rate (Whisper optimal)
                ac=1  # Mono channel
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            if not os.path.exists(output_path):
                raise RuntimeError("FFmpeg failed to create audio file")
                
            logger.info(f"Audio extracted to {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e}")
            raise RuntimeError(f"Failed to extract audio: {e}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)


class WhisperTranscriber:
    """Handles transcription using OpenAI Whisper with word-level timestamps."""
    
    def __init__(self, model_name: str = "medium"):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        logger.info("Whisper model loaded successfully")
    
    def transcribe_with_timestamps(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription and word-level timestamps
        """
        logger.info(f"Transcribing audio: {audio_path}")
        
        # Transcribe with word-level timestamps
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False
        )
        
        # Extract word-level timing information
        words_with_timestamps = []
        
        for segment in result.get('segments', []):
            for word_info in segment.get('words', []):
                words_with_timestamps.append({
                    'start': round(word_info.get('start', 0.0), 3),
                    'end': round(word_info.get('end', 0.0), 3),
                    'word': word_info.get('word', '').strip()
                })
        
        return {
            'text': result.get('text', ''),
            'language': result.get('language', 'unknown'),
            'words': words_with_timestamps
        }


class VideoTranscriptGenerator:
    """Main class that orchestrates audio extraction and transcription."""
    
    def __init__(self, whisper_model: str = "medium"):
        self.audio_extractor = AudioExtractor()
        self.transcriber = WhisperTranscriber(whisper_model)
    
    def process_video(self, video_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Process a video file: extract audio and generate transcript.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save transcript JSON (default: data/transcripts)
            
        Returns:
            Dictionary containing transcript data
        """
        if output_dir is None:
            output_dir = "data/transcripts"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video ID from filename
        video_id = Path(video_path).stem
        
        try:
            # Extract audio
            audio_path = self.audio_extractor.extract_audio(video_path)
            
            # Transcribe with timestamps
            transcript_data = self.transcriber.transcribe_with_timestamps(audio_path)
            
            # Format output according to spec
            output_data = {
                'video_id': video_id,
                'video_path': str(Path(video_path).absolute()),
                'language': transcript_data['language'],
                'full_text': transcript_data['text'],
                'words': transcript_data['words'],
                'word_count': len(transcript_data['words']),
                'duration_seconds': transcript_data['words'][-1]['end'] if transcript_data['words'] else 0.0
            }
            
            # Save to JSON file
            output_file = os.path.join(output_dir, f"{video_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transcript saved to {output_file}")
            logger.info(f"Processed {len(transcript_data['words'])} words in {output_data['duration_seconds']:.2f} seconds")
            
            return output_data
            
        finally:
            # Cleanup temporary files
            self.audio_extractor.cleanup()


def main():
    """CLI entry point for audio extraction and transcription."""
    parser = argparse.ArgumentParser(description="Extract audio and generate transcripts from video files")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output-dir", default="data/transcripts", 
                       help="Output directory for transcript JSON (default: data/transcripts)")
    parser.add_argument("--model", default="medium", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: medium)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process video
    generator = VideoTranscriptGenerator(whisper_model=args.model)
    result = generator.process_video(args.video_path, args.output_dir)
    
    print(f"✓ Successfully processed video: {args.video_path}")
    print(f"✓ Transcript saved with {result['word_count']} words")
    print(f"✓ Duration: {result['duration_seconds']:.2f} seconds")


if __name__ == "__main__":
    main() 