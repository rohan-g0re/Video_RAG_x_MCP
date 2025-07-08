#!/usr/bin/env python3
"""
Phase 1-B: 10-Second Segmentation & Normalization

Converts word-level transcripts into 10-second segments with normalized text.
Handles silent periods and outputs exactly ⌈duration/10⌉ segments.
"""

import os
import json
import math
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextNormalizer:
    """Handles text normalization and cleaning."""
    
    def __init__(self):
        # Pattern to match punctuation and special characters
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text according to specification:
        - Convert to lowercase
        - Strip punctuation  
        - Collapse whitespace
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text string
        """
        if not text or not text.strip():
            return ""
        
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove punctuation
        normalized = self.punctuation_pattern.sub(' ', normalized)
        
        # Collapse multiple whitespace into single spaces
        normalized = self.whitespace_pattern.sub(' ', normalized)
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized


class TranscriptSegmenter:
    """Segments word-level transcripts into 10-second buckets."""
    
    def __init__(self, segment_duration: float = 10.0):
        self.segment_duration = segment_duration
        self.normalizer = TextNormalizer()
    
    def segment_transcript(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Segment transcript into 10-second buckets.
        
        Args:
            transcript_data: Input transcript with word-level timestamps
            
        Returns:
            Segmented transcript data
        """
        words = transcript_data.get('words', [])
        video_id = transcript_data.get('video_id', 'unknown')
        duration = transcript_data.get('duration_seconds', 0.0)
        
        if not words:
            logger.warning("No words found in transcript")
            return self._create_empty_segments(video_id, duration)
        
        # Calculate number of segments needed
        num_segments = math.ceil(duration / self.segment_duration)
        if num_segments == 0:
            num_segments = 1
            
        logger.info(f"Creating {num_segments} segments for {duration:.2f}s video")
        
        segments = []
        
        for i in range(num_segments):
            start_time = i * self.segment_duration
            end_time = min((i + 1) * self.segment_duration, duration)
            
            # Find words that belong to this segment
            segment_words = self._get_words_in_timeframe(words, start_time, end_time)
            
            # Combine words into text and normalize
            if segment_words:
                raw_text = ' '.join(word['word'] for word in segment_words)
                normalized_text = self.normalizer.normalize_text(raw_text)
            else:
                normalized_text = ""
            
            # Check if this is a silent segment (≤2 words)
            if len(segment_words) <= 2:
                normalized_text = ""
                logger.debug(f"Silent segment {i}: {len(segment_words)} words")
            
            segment = {
                'start': round(start_time, 1),
                'end': round(end_time, 1),
                'text': normalized_text,
                'word_count': len(segment_words),
                'raw_words': segment_words  # Keep for debugging/validation
            }
            
            segments.append(segment)
        
        # Create output data structure
        output_data = {
            'video_id': video_id,
            'total_duration': duration,
            'segment_duration': self.segment_duration,
            'total_segments': len(segments),
            'segments': segments
        }
        
        # Validation
        self._validate_segments(output_data, duration)
        
        return output_data
    
    def _get_words_in_timeframe(self, words: List[Dict], start: float, end: float) -> List[Dict]:
        """
        Get words that fall within a specific timeframe.
        
        Words are included if their midpoint falls within the segment.
        """
        segment_words = []
        
        for word in words:
            word_start = word.get('start', 0.0)
            word_end = word.get('end', 0.0)
            word_midpoint = (word_start + word_end) / 2
            
            # Include word if its midpoint is in this segment
            if start <= word_midpoint < end:
                segment_words.append(word)
        
        return segment_words
    
    def _create_empty_segments(self, video_id: str, duration: float) -> Dict[str, Any]:
        """Create empty segments for videos with no words."""
        num_segments = max(1, math.ceil(duration / self.segment_duration))
        segments = []
        
        for i in range(num_segments):
            start_time = i * self.segment_duration
            end_time = min((i + 1) * self.segment_duration, duration)
            
            segments.append({
                'start': round(start_time, 1),
                'end': round(end_time, 1),
                'text': "",
                'word_count': 0,
                'raw_words': []
            })
        
        return {
            'video_id': video_id,
            'total_duration': duration,
            'segment_duration': self.segment_duration,
            'total_segments': len(segments),
            'segments': segments
        }
    
    def _validate_segments(self, output_data: Dict[str, Any], expected_duration: float):
        """Validate segment output meets requirements."""
        segments = output_data['segments']
        
        # Check segment count
        expected_segments = math.ceil(expected_duration / self.segment_duration)
        actual_segments = len(segments)
        
        if actual_segments != expected_segments:
            logger.warning(f"Segment count mismatch: expected {expected_segments}, got {actual_segments}")
        
        # Check segment timing
        for i, segment in enumerate(segments):
            expected_start = i * self.segment_duration
            expected_end = min((i + 1) * self.segment_duration, expected_duration)
            
            if abs(segment['start'] - expected_start) > 0.1:
                logger.warning(f"Segment {i} start time mismatch: expected {expected_start}, got {segment['start']}")
            
            if abs(segment['end'] - expected_end) > 0.1:
                logger.warning(f"Segment {i} end time mismatch: expected {expected_end}, got {segment['end']}")
        
        logger.info(f"✓ Validation passed: {actual_segments} segments, {expected_duration:.2f}s duration")


def process_transcript_file(input_file: str, output_file: str = None, segment_duration: float = 10.0) -> Dict[str, Any]:
    """
    Process a transcript JSON file and create segmented output.
    
    Args:
        input_file: Path to input transcript JSON
        output_file: Path to output segmented JSON (optional)
        segment_duration: Duration of each segment in seconds
        
    Returns:
        Segmented transcript data
    """
    # Load input transcript
    with open(input_file, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Create segmenter and process
    segmenter = TranscriptSegmenter(segment_duration)
    segmented_data = segmenter.segment_transcript(transcript_data)
    
    # Determine output file path
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_segmented.json"
    
    # Save segmented data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(segmented_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Segmented transcript saved to {output_file}")
    logger.info(f"Created {segmented_data['total_segments']} segments from {segmented_data['total_duration']:.2f}s video")
    
    return segmented_data


def main():
    """CLI entry point for transcript segmentation."""
    parser = argparse.ArgumentParser(description="Segment word-level transcripts into 10-second buckets")
    parser.add_argument("input_file", help="Path to input transcript JSON file")
    parser.add_argument("--output-file", "-o", help="Path to output segmented JSON file")
    parser.add_argument("--segment-duration", "-d", type=float, default=10.0,
                       help="Duration of each segment in seconds (default: 10.0)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process transcript
    result = process_transcript_file(
        args.input_file, 
        args.output_file, 
        args.segment_duration
    )
    
    print(f"✓ Successfully segmented transcript: {args.input_file}")
    print(f"✓ Created {result['total_segments']} segments")
    print(f"✓ Duration: {result['total_duration']:.2f} seconds")
    
    # Print segment summary
    silent_segments = sum(1 for seg in result['segments'] if not seg['text'])
    print(f"✓ Silent segments: {silent_segments}/{result['total_segments']}")


if __name__ == "__main__":
    main() 