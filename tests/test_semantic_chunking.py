#!/usr/bin/env python3
"""
Test Script: Semantic Chunking Comparison

Tests the new semantic-aware chunking against the original approach
using the test_video.mp4 transcript data.
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_audio.segment_transcript import process_transcript_file
from phase1_audio.segment_transcript_semantic import process_transcript_file_semantic

def compare_chunking_approaches():
    """Compare original vs semantic chunking approaches."""
    
    print("=" * 80)
    print("🔬 CHUNKING APPROACH COMPARISON - TEST VIDEO")
    print("=" * 80)
    
    transcript_file = "data/transcripts/test_video.json"
    
    if not Path(transcript_file).exists():
        print(f"❌ Transcript file not found: {transcript_file}")
        print("Please run the audio extraction first!")
        return False
    
    # Load original transcript data
    with open(transcript_file, 'r') as f:
        original_data = json.load(f)
    
    print(f"📹 Video: {original_data['video_id']}")
    print(f"⏱️  Duration: {original_data['duration_seconds']:.1f}s")
    print(f"📝 Total words: {original_data['word_count']}")
    print()
    
    # Test 1: Original Fixed 10-second Chunking
    print("🔹 APPROACH 1: ORIGINAL FIXED 10-SECOND CHUNKING")
    print("-" * 60)
    
    start_time = time.time()
    original_segments = process_transcript_file(transcript_file, "data/transcripts/test_video_original.json")
    original_time = time.time() - start_time
    
    print(f"✓ Processing time: {original_time:.3f}s")
    print(f"✓ Segments created: {original_segments['total_segments']}")
    print(f"✓ Method: Fixed 10-second windows")
    print()
    
    # Show original segments
    print("📋 Original Segments:")
    for i, seg in enumerate(original_segments['segments'][:3]):  # Show first 3
        print(f"  {i+1}. [{seg['start']:.1f}s-{seg['end']:.1f}s] ({seg['word_count']} words)")
        print(f"     \"{seg['text'][:80]}...\"")
    if len(original_segments['segments']) > 3:
        print(f"     ... and {len(original_segments['segments'])-3} more segments")
    print()
    
    # Test 2: New Semantic-Aware Chunking
    print("🔹 APPROACH 2: NEW SEMANTIC-AWARE CHUNKING")
    print("-" * 60)
    
    start_time = time.time()
    semantic_segments = process_transcript_file_semantic(
        transcript_file, 
        "data/transcripts/test_video_semantic.json",
        min_duration=5.0,
        max_duration=15.0
    )
    semantic_time = time.time() - start_time
    
    print(f"✓ Processing time: {semantic_time:.3f}s")
    print(f"✓ Segments created: {semantic_segments['total_segments']}")
    print(f"✓ Method: {semantic_segments['segmentation_method']}")
    print()
    
    # Show semantic segments with enhanced details
    print("📋 Semantic Segments:")
    for i, seg in enumerate(semantic_segments['segments']):
        duration = seg['metadata']['duration']
        reason = seg['metadata']['segmentation_reason']
        content_type = seg['metadata']['content_type']
        timing = seg['metadata']['timing_formatted']
        
        print(f"  {i+1}. [{timing}] ({duration:.1f}s, {seg['word_count']} words)")
        print(f"     Caption: \"{seg['caption']}\"")
        print(f"     Type: {content_type} | Reason: {reason}")
        print()
    
    # Detailed Comparison
    print("=" * 80)
    print("📊 DETAILED COMPARISON")
    print("=" * 80)
    
    # Calculate metrics
    orig_durations = [seg['end'] - seg['start'] for seg in original_segments['segments']]
    sem_durations = [seg['metadata']['duration'] for seg in semantic_segments['segments']]
    
    # Text quality comparison
    orig_readable = sum(1 for seg in original_segments['segments'] 
                       if seg['text'] and len(seg['text']) > 20)
    sem_readable = sum(1 for seg in semantic_segments['segments'] 
                      if seg['caption'] and len(seg['caption']) > 20)
    
    # Content type analysis
    content_types = {}
    for seg in semantic_segments['segments']:
        ct = seg['metadata']['content_type']
        content_types[ct] = content_types.get(ct, 0) + 1
    
    print("📏 SEGMENT METRICS:")
    print(f"  Original approach:")
    print(f"    • Segments: {len(original_segments['segments'])}")
    print(f"    • Avg duration: {sum(orig_durations)/len(orig_durations):.1f}s")
    print(f"    • Duration range: {min(orig_durations):.1f}s - {max(orig_durations):.1f}s")
    print(f"    • Readable segments: {orig_readable}/{len(original_segments['segments'])}")
    print()
    print(f"  Semantic approach:")
    print(f"    • Segments: {len(semantic_segments['segments'])}")
    print(f"    • Avg duration: {sum(sem_durations)/len(sem_durations):.1f}s")
    print(f"    • Duration range: {min(sem_durations):.1f}s - {max(sem_durations):.1f}s")
    print(f"    • Readable segments: {sem_readable}/{len(semantic_segments['segments'])}")
    print()
    
    print("🎭 CONTENT TYPE DISTRIBUTION:")
    for content_type, count in content_types.items():
        percentage = (count / len(semantic_segments['segments'])) * 100
        print(f"    • {content_type}: {count} segments ({percentage:.1f}%)")
    print()
    
    print("🔍 TEXT QUALITY COMPARISON:")
    
    # Show side-by-side comparison of first few segments
    max_compare = min(3, len(original_segments['segments']), len(semantic_segments['segments']))
    
    for i in range(max_compare):
        orig_seg = original_segments['segments'][i]
        sem_seg = semantic_segments['segments'][i] if i < len(semantic_segments['segments']) else None
        
        print(f"\n  Segment {i+1} Comparison:")
        print(f"    Original: \"{orig_seg['text'][:60]}...\"")
        if sem_seg:
            print(f"    Semantic: \"{sem_seg['caption'][:60]}...\"")
            print(f"    Timing:   {sem_seg['metadata']['timing_formatted']} ({sem_seg['metadata']['duration']:.1f}s)")
        print()
    
    print("=" * 80)
    print("🏆 SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    # Analyze improvements
    improvements = []
    if sem_readable > orig_readable:
        improvements.append(f"✅ Better readability: {sem_readable} vs {orig_readable} readable segments")
    
    if len(semantic_segments['segments']) != len(original_segments['segments']):
        improvements.append(f"📊 Adaptive segmentation: {len(semantic_segments['segments'])} vs {len(original_segments['segments'])} segments")
    
    # Check for better natural boundaries
    natural_boundaries = sum(1 for seg in semantic_segments['segments'] 
                           if seg['metadata']['segmentation_reason'] in ['natural_boundary', 'target_duration_with_boundary'])
    if natural_boundaries > 0:
        improvements.append(f"🎯 Natural boundaries: {natural_boundaries}/{len(semantic_segments['segments'])} segments")
    
    print("🎉 SEMANTIC CHUNKING IMPROVEMENTS:")
    for improvement in improvements:
        print(f"  {improvement}")
    
    if not improvements:
        print("  📝 Results are similar - may need parameter tuning")
    
    print()
    print("💡 KEY BENEFITS OF SEMANTIC APPROACH:")
    print("  ✅ Preserves punctuation and capitalization")
    print("  ✅ Respects sentence boundaries")  
    print("  ✅ Adaptive segment lengths (5-15s)")
    print("  ✅ Rich metadata for better retrieval")
    print("  ✅ Content type classification")
    print("  ✅ Human-readable captions")
    
    return True

def main():
    """Run the chunking comparison test."""
    
    success = compare_chunking_approaches()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("  1. Review the semantic segments above")
        print("  2. Check if captions are more readable")
        print("  3. Verify natural timing boundaries")
        print("  4. Test with embedding generation")
        print("\n✅ Semantic chunking test completed successfully!")
    else:
        print("\n❌ Semantic chunking test failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 