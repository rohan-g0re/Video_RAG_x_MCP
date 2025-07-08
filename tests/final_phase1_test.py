#!/usr/bin/env python3
"""
Final Phase 1 Test with Real Video

Tests the complete Phase 1 audio processing pipeline using test_video.mp4:
1-A: Audio extraction and Whisper transcription
1-B: 10-second segmentation and normalization  
1-C: CLIP text embedding with performance validation
"""

import sys
import time
import json
import math
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_audio.extract_transcribe import VideoTranscriptGenerator
from phase1_audio.segment_transcript import process_transcript_file
from phase1_audio.embed_text import process_segmented_file

def run_complete_phase1_pipeline(video_path: str):
    """Run the complete Phase 1 pipeline on a real video file."""
    
    print("=" * 70)
    print("🎬 FINAL PHASE 1 TEST WITH REAL VIDEO")
    print("=" * 70)
    print(f"Testing with: {video_path}")
    print()
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        # Phase 1-A: Audio Extraction & Transcription
        print("🎵 Phase 1-A: Audio Extraction & Whisper Transcription")
        print("-" * 50)
        
        transcript_start = time.time()
        generator = VideoTranscriptGenerator(whisper_model="base")  # Use base for faster testing
        transcript_data = generator.process_video(video_path, "data/transcripts")
        transcript_time = time.time() - transcript_start
        
        print(f"✓ Transcription completed in {transcript_time:.1f}s")
        print(f"✓ Video ID: {transcript_data['video_id']}")
        print(f"✓ Duration: {transcript_data['duration_seconds']:.1f}s")
        print(f"✓ Words extracted: {transcript_data['word_count']}")
        print(f"✓ Language detected: {transcript_data['language']}")
        
        # Validate timing coverage
        words = transcript_data['words']
        valid_timing = sum(1 for w in words if 'start' in w and 'end' in w and 
                          isinstance(w['start'], (int, float)) and isinstance(w['end'], (int, float)))
        coverage = valid_timing / len(words) if words else 0
        
        if coverage >= 0.95:
            print(f"✓ Timing coverage: {coverage:.1%} ≥ 95% requirement")
        else:
            print(f"❌ Timing coverage: {coverage:.1%} < 95% requirement")
            return False
        
        print()
        
        # Phase 1-B: Segmentation & Normalization
        print("📊 Phase 1-B: Transcript Segmentation & Normalization")
        print("-" * 50)
        
        video_id = transcript_data['video_id']
        transcript_file = f"data/transcripts/{video_id}.json"
        
        segment_start = time.time()
        segmented_data = process_transcript_file(transcript_file)
        segment_time = time.time() - segment_start
        
        print(f"✓ Segmentation completed in {segment_time:.2f}s")
        
        # Validate segment count
        duration = transcript_data['duration_seconds']
        expected_segments = math.ceil(duration / 10.0)
        actual_segments = segmented_data['total_segments']
        
        if actual_segments == expected_segments:
            print(f"✓ Segment count: {actual_segments} = ⌈{duration:.1f}/10⌉")
        else:
            print(f"❌ Segment count: {actual_segments} ≠ ⌈{duration:.1f}/10⌉ = {expected_segments}")
            return False
        
        # Show segment statistics
        text_segments = sum(1 for seg in segmented_data['segments'] if seg['text'])
        silent_segments = actual_segments - text_segments
        
        print(f"✓ Text segments: {text_segments}")
        print(f"✓ Silent segments: {silent_segments}")
        print(f"✓ Average words per segment: {sum(seg['word_count'] for seg in segmented_data['segments']) / actual_segments:.1f}")
        
        print()
        
        # Phase 1-C: Text Embedding
        print("🧠 Phase 1-C: CLIP Text Embedding")
        print("-" * 50)
        
        segmented_file = f"data/transcripts/{video_id}_segmented.json"
        embeddings_file = f"data/transcripts/{video_id}_embeddings.json"
        
        embed_start = time.time()
        embed_results = process_segmented_file(
            segmented_file, 
            embeddings_file, 
            clip_model="ViT-B-32", 
            batch_size=32
        )
        embed_time = time.time() - embed_start
        
        print(f"✓ Embedding completed in {embed_time:.2f}s")
        
        if embed_results['status'] == 'success':
            print(f"✓ Segments processed: {embed_results['segments_processed']}")
            print(f"✓ Embedding dimension: {embed_results['embedding_dimension']}")
            print(f"✓ Performance time: {embed_results['processing_time_seconds']:.2f}s")
            print(f"✓ Performance target: ≤{embed_results['performance_target_seconds']:.2f}s")
            
            if embed_results['performance_ok']:
                print("✓ Performance requirement: PASSED")
            else:
                print("❌ Performance requirement: FAILED")
                return False
                
        else:
            print(f"❌ Embedding failed: {embed_results.get('message', 'Unknown error')}")
            return False
        
        print()
        
        # Final Summary
        total_time = transcript_time + segment_time + embed_time
        
        print("=" * 70)
        print("🏆 FINAL PHASE 1 TEST RESULTS")
        print("=" * 70)
        print(f"✓ Total pipeline time: {total_time:.1f}s")
        print(f"✓ Video processed: {transcript_data['duration_seconds']:.1f}s duration")
        print(f"✓ Words transcribed: {transcript_data['word_count']}")
        print(f"✓ Segments created: {actual_segments}")
        print(f"✓ Embeddings generated: {embed_results['embeddings_generated']}")
        print()
        print("🎉 ALL PHASE 1 DELIVERABLES VALIDATED WITH REAL VIDEO!")
        print()
        print("Phase 1 Acceptance Criteria - ALL PASSED:")
        print("• extract_transcribe.py: ≥95% timing field coverage ✓")
        print("• segment_transcript.py: Exact ⌈duration/10⌉ segments ✓") 
        print("• embed_text.py: Performance ≤(N/8)s target ✓")
        print()
        print("🚀 Phase 1 is COMPLETE and ready for Phase 2!")
        
        return True
        
    except Exception as e:
        print(f"💥 Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run final Phase 1 test with real video."""
    video_path = "test_video.mp4"
    
    success = run_complete_phase1_pipeline(video_path)
    
    if success:
        print("\n✅ FINAL TEST PASSED - Phase 1 implementation is complete!")
    else:
        print("\n❌ FINAL TEST FAILED - Please review the errors above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 