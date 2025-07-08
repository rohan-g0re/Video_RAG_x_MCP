#!/usr/bin/env python3
"""
Phase 1-B Enhanced: Semantic-Aware Transcript Segmentation

Simplified chunking approach that:
- Uses adaptive segment lengths (5-15 seconds) based on content
- Preserves punctuation and capitalization for readable captions  
- Respects sentence boundaries only (no pause or topic detection)
- Adds 1-second overlap between segments for better context
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
        # Patterns for sentence detection (punctuation only)
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
    
    def is_sentence_boundary(self, word: str) -> bool:
        """Check if this word represents a sentence boundary (punctuation only)."""
        # Check for sentence-ending punctuation only
        return bool(self.sentence_endings.search(word))


class SemanticTranscriptSegmenter:
    """Creates semantic-aware segments with natural boundaries and overlap."""
    
    def __init__(self, min_duration: float = 7.0, max_duration: float = 15.0, 
                 target_duration: float = 10.0, overlap_duration: float = 1.0):
        self.min_duration = min_duration
        self.max_duration = max_duration  
        self.target_duration = target_duration
        self.overlap_duration = overlap_duration
        self.text_processor = SemanticTextProcessor()
        
    def segment_transcript(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create semantic-aware segments with natural boundaries and overlap.
        """
        words = transcript_data.get('words', [])
        video_id = transcript_data.get('video_id', 'unknown')
        duration = transcript_data.get('duration_seconds', 0.0)
        
        if not words:
            logger.warning("No words found in transcript")
            return self._create_empty_segments(video_id, duration)
        
        logger.info(f"Creating semantic segments for {duration:.2f}s video with {len(words)} words")
        
        # Create base segments without overlap
        base_segments = self._create_semantic_segments(words, duration)
        
        # Add overlap to segments
        overlapped_segments = self._add_overlap_to_segments(base_segments, words, duration)
        
        # Create output data structure
        output_data = {
            'video_id': video_id,
            'total_duration': duration,
            'segmentation_method': 'semantic_adaptive_overlap',
            'segment_config': {
                'min_duration': self.min_duration,
                'max_duration': self.max_duration,
                'target_duration': self.target_duration,
                'overlap_duration': self.overlap_duration
            },
            'total_segments': len(overlapped_segments),
            'segments': overlapped_segments
        }
        
        # Enhanced validation
        self._validate_semantic_segments(output_data, duration)
        
        return output_data
    
    def _create_semantic_segments(self, words: List[Dict], total_duration: float) -> List[Dict]:
        """Create segments using simplified semantic boundaries (punctuation only)."""
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
            
            # Check for sentence boundary (punctuation only)
            is_sentence_end = self.text_processor.is_sentence_boundary(word['word'])
            
            # Simplified decision logic
            should_segment = False
            
            if is_max_duration_exceeded:
                should_segment = True
            elif is_min_duration_met and is_sentence_end:
                should_segment = True  
            elif is_target_duration_reached and is_sentence_end:
                should_segment = True
            elif i == len(words) - 1:  # Last word
                should_segment = True
            
            if should_segment:
                segment = self._create_segment_from_words(
                    current_segment_words, 
                    segment_start_time
                )
                segments.append(segment)
                
                # Reset for next segment
                current_segment_words = []
                segment_start_time = word['end']
        
        return segments
    
    def _add_overlap_to_segments(self, segments: List[Dict], words: List[Dict], 
                                total_duration: float) -> List[Dict]:
        """Add overlap to segments by extending start/end times."""
        if not segments:
            return segments
        
        overlapped_segments = []
        
        for i, segment in enumerate(segments):
            # Calculate overlapped timing
            original_start = segment['start']
            original_end = segment['end']
            
            # Extend backward by overlap_duration (but not before 0)
            overlap_start = max(0.0, original_start - self.overlap_duration)
            
            # Extend forward by overlap_duration (but not beyond video end)
            overlap_end = min(total_duration, original_end + self.overlap_duration)
            
            # Find words that fall within the overlapped time range
            overlapped_words = []
            for word in words:
                if (word['start'] >= overlap_start and word['end'] <= overlap_end):
                    overlapped_words.append(word)
            
            # Create overlapped segment
            if overlapped_words:
                overlapped_segment = self._create_segment_from_words(
                    overlapped_words,
                    overlap_start,
                    force_timing=(overlap_start, overlap_end)
                )
                # Add overlap metadata
                overlapped_segment['metadata']['original_start'] = original_start
                overlapped_segment['metadata']['original_end'] = original_end
                overlapped_segment['metadata']['overlap_added'] = self.overlap_duration
            else:
                # Fallback to original segment if no words found in overlap range
                overlapped_segment = segment
                overlapped_segment['metadata']['overlap_added'] = 0.0
            
            overlapped_segments.append(overlapped_segment)
        
        return overlapped_segments
    
    def _create_segment_from_words(self, words: List[Dict], start_time: float, 
                                  force_timing: Tuple[float, float] = None) -> Dict[str, Any]:
        """Create a segment object from a list of words."""
        if not words:
            return {
                'start': start_time,
                'end': start_time,
                'caption': "",
                'word_count': 0,
                'metadata': {
                    'duration': 0.0,
                    'sentence_count': 0
                }
            }
        
        # Build readable caption (preserve original formatting)
        raw_text = ' '.join(word['word'] for word in words)
        caption = self.text_processor.clean_text_lightly(raw_text)
        
        # Calculate timing
        if force_timing:
            actual_start, actual_end = force_timing
        else:
            actual_start = words[0]['start']
            actual_end = words[-1]['end']
        
        duration = actual_end - actual_start
        
        # Analyze content
        sentence_count = len(self.text_processor.sentence_endings.findall(caption))
        if sentence_count == 0 and caption:  # If no clear sentences, count as 1
            sentence_count = 1
        
        # Detect content type
        content_type = self._classify_content(caption)
        
        return {
            'start': round(actual_start, 2),
            'end': round(actual_end, 2),
            'caption': caption,
            'word_count': len(words),
            'metadata': {
                'duration': round(duration, 2),
                'sentence_count': sentence_count,
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
            'segmentation_method': 'semantic_adaptive_overlap',
            'total_segments': 0,
            'segments': []
        }
    
    def _validate_semantic_segments(self, output_data: Dict[str, Any], expected_duration: float):
        """Validate semantic segmentation results."""
        segments = output_data['segments']
        
        if not segments:
            logger.warning("No segments created")
            return
        
        # Check duration distribution
        durations = [seg['metadata']['duration'] for seg in segments]
        avg_duration = sum(durations) / len(durations)
        
        # Check for overlaps
        overlaps = []
        for i in range(len(segments) - 1):
            if segments[i]['end'] > segments[i + 1]['start']:
                overlap = segments[i]['end'] - segments[i + 1]['start']
                overlaps.append(overlap)
        
        logger.info(f"✓ Semantic segmentation validation:")
        logger.info(f"  • Segments created: {len(segments)}")
        logger.info(f"  • Average duration: {avg_duration:.1f}s")
        logger.info(f"  • Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
        logger.info(f"  • Overlaps detected: {len(overlaps)} (expected with 1s overlap)")
        if overlaps:
            logger.info(f"  • Average overlap: {sum(overlaps)/len(overlaps):.1f}s")
        
        # Quality metrics
        readable_segments = sum(1 for seg in segments if seg['caption'] and len(seg['caption']) > 10)
        logger.info(f"  • Readable segments: {readable_segments}/{len(segments)}")


def process_transcript_file_semantic(input_file: str, output_file: str = None, 
                                   min_duration: float = 7.0, max_duration: float = 15.0,
                                   overlap_duration: float = 1.0) -> Dict[str, Any]:
    """
    Process a transcript JSON file with simplified semantic segmentation and overlap.
    """
    # Load input transcript
    with open(input_file, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Create semantic segmenter
    segmenter = SemanticTranscriptSegmenter(min_duration, max_duration, overlap_duration=overlap_duration)
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
    """CLI entry point for simplified semantic transcript segmentation."""
    parser = argparse.ArgumentParser(description="Create simplified semantic-aware transcript segments with overlap")
    parser.add_argument("input_file", help="Path to input transcript JSON file")
    parser.add_argument("--output-file", "-o", help="Path to output segmented JSON file")
    parser.add_argument("--min-duration", "-min", type=float, default=5.0,
                       help="Minimum segment duration in seconds (default: 5.0)")
    parser.add_argument("--max-duration", "-max", type=float, default=15.0,
                       help="Maximum segment duration in seconds (default: 15.0)")
    parser.add_argument("--overlap-duration", "-overlap", type=float, default=1.0,
                       help="Overlap duration in seconds (default: 1.0)")
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
        args.max_duration,
        args.overlap_duration
    )
    
    print(f"✓ Successfully created semantic segments: {args.input_file}")
    print(f"✓ Segments created: {result['total_segments']}")
    print(f"✓ Duration: {result['total_duration']:.2f} seconds")
    print(f"✓ Method: {result['segmentation_method']}")
    print(f"✓ Overlap: {args.overlap_duration}s")


if __name__ == "__main__":
    main() 