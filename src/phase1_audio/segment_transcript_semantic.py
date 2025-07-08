#!/usr/bin/env python3
"""
Phase 1-B Enhanced: Semantic-Aware Transcript Segmentation

Improved chunking approach that:
- Uses adaptive segment lengths (5-15 seconds) based on content
- Preserves punctuation and capitalization for readable captions  
- Respects sentence boundaries and natural speech pauses
- Creates rich metadata for better retrieval
"""

import os
import json
import math
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticTextProcessor:
    """Handles smart text processing while preserving readability."""
    
    def __init__(self):
        # Patterns for sentence detection
        self.sentence_endings = re.compile(r'[.!?]+')
        self.quote_endings = re.compile(r'[.!?]+["\']')
        
    def clean_text_lightly(self, text: str) -> str:
        """
        Light text cleaning that preserves readability.
        Only removes extra whitespace, keeps punctuation and capitalization.
        """
        if not text or not text.strip():
            return ""
        
        # Collapse multiple whitespace but keep structure
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common transcription issues
        cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)  # Remove space before punctuation
        cleaned = re.sub(r'([.!?])\s*([a-z])', lambda m: f"{m.group(1)} {m.group(2).upper()}", cleaned)  # Capitalize after sentence
        
        return cleaned
    
    def is_sentence_boundary(self, word: str, next_word_start: float, current_word_end: float) -> bool:
        """Check if this word represents a sentence boundary."""
        # Check for sentence-ending punctuation
        has_sentence_end = bool(self.sentence_endings.search(word))
        
        # Check for significant pause (>0.5s indicates sentence boundary)
        has_pause = next_word_start - current_word_end > 0.5
        
        return has_sentence_end or has_pause
    
    def detect_topic_change(self, words: List[str], window_size: int = 5) -> bool:
        """Simple topic change detection based on vocabulary shift."""
        if len(words) < window_size * 2:
            return False
            
        # Compare word overlap between first and last parts of window
        first_half = set(words[:window_size])
        second_half = set(words[-window_size:])
        
        overlap = len(first_half.intersection(second_half))
        total_unique = len(first_half.union(second_half))
        
        # Low overlap suggests topic change
        return (overlap / total_unique) < 0.3 if total_unique > 0 else False


class SemanticTranscriptSegmenter:
    """Creates semantic-aware segments with natural boundaries."""
    
    def __init__(self, min_duration: float = 5.0, max_duration: float = 15.0, target_duration: float = 10.0):
        self.min_duration = min_duration
        self.max_duration = max_duration  
        self.target_duration = target_duration
        self.text_processor = SemanticTextProcessor()
        
    def segment_transcript(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create semantic-aware segments with natural boundaries.
        """
        words = transcript_data.get('words', [])
        video_id = transcript_data.get('video_id', 'unknown')
        duration = transcript_data.get('duration_seconds', 0.0)
        
        if not words:
            logger.warning("No words found in transcript")
            return self._create_empty_segments(video_id, duration)
        
        logger.info(f"Creating semantic segments for {duration:.2f}s video with {len(words)} words")
        
        segments = self._create_semantic_segments(words, duration)
        
        # Create output data structure
        output_data = {
            'video_id': video_id,
            'total_duration': duration,
            'segmentation_method': 'semantic_adaptive',
            'segment_config': {
                'min_duration': self.min_duration,
                'max_duration': self.max_duration,
                'target_duration': self.target_duration
            },
            'total_segments': len(segments),
            'segments': segments
        }
        
        # Enhanced validation
        self._validate_semantic_segments(output_data, duration)
        
        return output_data
    
    def _create_semantic_segments(self, words: List[Dict], total_duration: float) -> List[Dict]:
        """Create segments using semantic boundaries."""
        segments = []
        current_segment_words = []
        segment_start_time = 0.0
        
        for i, word in enumerate(words):
            current_segment_words.append(word)
            
            # Calculate current segment duration
            current_duration = word['end'] - segment_start_time
            
            # Check for natural break conditions
            is_min_duration_met = current_duration >= self.min_duration
            is_max_duration_exceeded = current_duration >= self.max_duration
            is_target_duration_reached = current_duration >= self.target_duration
            
            # Check for natural boundaries
            is_sentence_end = False
            has_long_pause = False
            
            if i < len(words) - 1:  # Not the last word
                next_word = words[i + 1]
                is_sentence_end = self.text_processor.is_sentence_boundary(
                    word['word'], next_word['start'], word['end']
                )
                has_long_pause = next_word['start'] - word['end'] > 0.7
            
            # Topic change detection (simplified)
            has_topic_change = False
            if len(current_segment_words) > 10:
                recent_words = [w['word'] for w in current_segment_words[-10:]]
                has_topic_change = self.text_processor.detect_topic_change(recent_words)
            
            # Decision logic for segmentation
            should_segment = False
            
            if is_max_duration_exceeded:
                should_segment = True
                reason = "max_duration_exceeded"
            elif is_min_duration_met and (is_sentence_end or has_long_pause):
                should_segment = True  
                reason = "natural_boundary"
            elif is_target_duration_reached and (is_sentence_end or has_topic_change):
                should_segment = True
                reason = "target_duration_with_boundary"
            elif i == len(words) - 1:  # Last word
                should_segment = True
                reason = "end_of_transcript"
            
            if should_segment:
                segment = self._create_segment_from_words(
                    current_segment_words, 
                    segment_start_time,
                    reason
                )
                segments.append(segment)
                
                # Reset for next segment
                current_segment_words = []
                segment_start_time = word['end']
        
        return segments
    
    def _create_segment_from_words(self, words: List[Dict], start_time: float, reason: str) -> Dict[str, Any]:
        """Create a segment object from a list of words."""
        if not words:
            return {
                'start': start_time,
                'end': start_time,
                'caption': "",
                'word_count': 0,
                'metadata': {
                    'duration': 0.0,
                    'segmentation_reason': reason,
                    'sentence_count': 0,
                    'has_pause': False
                }
            }
        
        # Build readable caption (preserve original formatting)
        raw_text = ' '.join(word['word'] for word in words)
        caption = self.text_processor.clean_text_lightly(raw_text)
        
        # Calculate timing
        actual_start = words[0]['start']
        actual_end = words[-1]['end']
        duration = actual_end - actual_start
        
        # Analyze content
        sentence_count = len(self.text_processor.sentence_endings.findall(caption))
        if sentence_count == 0 and caption:  # If no clear sentences, count as 1
            sentence_count = 1
            
        # Check for significant pauses within segment
        has_internal_pause = False
        for i in range(len(words) - 1):
            if words[i + 1]['start'] - words[i]['end'] > 0.5:
                has_internal_pause = True
                break
        
        # Detect content type
        content_type = self._classify_content(caption)
        
        return {
            'start': round(actual_start, 2),
            'end': round(actual_end, 2),
            'caption': caption,
            'word_count': len(words),
            'metadata': {
                'duration': round(duration, 2),
                'segmentation_reason': reason,
                'sentence_count': sentence_count,
                'has_pause': has_internal_pause,
                'content_type': content_type,
                'timing_formatted': f"{self._format_timestamp(actual_start)} - {self._format_timestamp(actual_end)}"
            },
            'raw_words': words  # Keep for compatibility and debugging
        }
    
    def _classify_content(self, text: str) -> str:
        """Simple content classification."""
        if not text:
            return "silence"
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['?', 'who', 'what', 'where', 'when', 'why', 'how']):
            return "question"
        elif any(word in text_lower for word in ['!', 'yeah', 'yes', 'no', 'oh', 'ah']):
            return "exclamation"
        elif any(word in text_lower for word in ['i think', 'i believe', 'maybe', 'probably']):
            return "opinion"
        elif len(text.split()) < 3:
            return "short_response"
        else:
            return "statement"
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS.mm"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"
    
    def _create_empty_segments(self, video_id: str, duration: float) -> Dict[str, Any]:
        """Create empty segments for videos with no words."""
        return {
            'video_id': video_id,
            'total_duration': duration,
            'segmentation_method': 'semantic_adaptive',
            'total_segments': 0,
            'segments': []
        }
    
    def _validate_semantic_segments(self, output_data: Dict[str, Any], expected_duration: float):
        """Validate semantic segmentation results."""
        segments = output_data['segments']
        
        if not segments:
            logger.warning("No segments created")
            return
        
        # Check coverage
        total_covered = segments[-1]['end'] - segments[0]['start']
        coverage = total_covered / expected_duration
        
        # Check duration distribution
        durations = [seg['metadata']['duration'] for seg in segments]
        avg_duration = sum(durations) / len(durations)
        
        # Check for gaps
        gaps = []
        for i in range(len(segments) - 1):
            gap = segments[i + 1]['start'] - segments[i]['end']
            if gap > 0.1:  # 100ms gap tolerance
                gaps.append(gap)
        
        logger.info(f"✓ Semantic segmentation validation:")
        logger.info(f"  • Segments created: {len(segments)}")
        logger.info(f"  • Coverage: {coverage:.1%} of video")
        logger.info(f"  • Average duration: {avg_duration:.1f}s")
        logger.info(f"  • Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
        logger.info(f"  • Gaps detected: {len(gaps)}")
        
        # Quality metrics
        readable_segments = sum(1 for seg in segments if seg['caption'] and len(seg['caption']) > 10)
        logger.info(f"  • Readable segments: {readable_segments}/{len(segments)}")


def process_transcript_file_semantic(input_file: str, output_file: str = None, 
                                   min_duration: float = 5.0, max_duration: float = 15.0) -> Dict[str, Any]:
    """
    Process a transcript JSON file with semantic segmentation.
    """
    # Load input transcript
    with open(input_file, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Create semantic segmenter
    segmenter = SemanticTranscriptSegmenter(min_duration, max_duration)
    segmented_data = segmenter.segment_transcript(transcript_data)
    
    # Determine output file path
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_semantic_segmented.json"
    
    # Save segmented data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(segmented_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Semantic segments saved to {output_file}")
    
    return segmented_data


def main():
    """CLI entry point for semantic transcript segmentation."""
    parser = argparse.ArgumentParser(description="Create semantic-aware transcript segments")
    parser.add_argument("input_file", help="Path to input transcript JSON file")
    parser.add_argument("--output-file", "-o", help="Path to output segmented JSON file")
    parser.add_argument("--min-duration", "-min", type=float, default=5.0,
                       help="Minimum segment duration in seconds (default: 5.0)")
    parser.add_argument("--max-duration", "-max", type=float, default=15.0,
                       help="Maximum segment duration in seconds (default: 15.0)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process transcript
    result = process_transcript_file_semantic(
        args.input_file,
        args.output_file,
        args.min_duration,
        args.max_duration
    )
    
    print(f"✓ Successfully created semantic segments: {args.input_file}")
    print(f"✓ Segments created: {result['total_segments']}")
    print(f"✓ Duration: {result['total_duration']:.2f} seconds")
    print(f"✓ Method: {result['segmentation_method']}")


if __name__ == "__main__":
    main() 