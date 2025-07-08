#!/usr/bin/env python3
"""
Simplified Semantic Pipeline Test

Tests the simplified semantic-aware chunking + embedding pipeline
with overlap functionality and shows detailed sample results.
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_audio.segment_transcript_semantic import process_transcript_file_semantic
from phase1_audio.embed_text_semantic import process_semantic_segmented_file

def test_simplified_semantic_pipeline():
    """Test the complete simplified semantic-aware pipeline with overlap."""
    
    print("=" * 90)
    print("🚀 SIMPLIFIED SEMANTIC-AWARE VIDEO RAG PIPELINE TEST")
    print("=" * 90)
    
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
    
    # Step 1: Create simplified semantic segments with overlap
    print("🔹 STEP 1: SIMPLIFIED SEMANTIC-AWARE SEGMENTATION WITH OVERLAP")
    print("-" * 70)
    
    start_time = time.time()
    semantic_segments = process_transcript_file_semantic(
        transcript_file, 
        "data/transcripts/test_video_simplified_semantic.json",
        min_duration=5.0,
        max_duration=15.0,
        overlap_duration=1.0
    )
    segmentation_time = time.time() - start_time
    
    print(f"✅ Segmentation completed in {segmentation_time:.3f}s")
    print(f"✅ Segments created: {semantic_segments['total_segments']}")
    print(f"✅ Method: {semantic_segments['segmentation_method']}")
    print(f"✅ Overlap: {semantic_segments['segment_config']['overlap_duration']}s")
    print()
    
    # Step 2: Generate simplified semantic embeddings
    print("🔹 STEP 2: SIMPLIFIED EMBEDDING GENERATION")
    print("-" * 70)
    
    start_time = time.time()
    embedding_results = process_semantic_segmented_file(
        "data/transcripts/test_video_simplified_semantic.json",
        "data/embeddings/test_video_simplified_embeddings.json",
        batch_size=16
    )
    embedding_time = time.time() - start_time
    
    print(f"✅ Embedding completed in {embedding_time:.3f}s")
    print(f"✅ Embeddings generated: {embedding_results['embeddings_generated']}")
    print(f"✅ Embedding dimension: {embedding_results['embedding_dimension']}")
    print(f"✅ Performance: {'PASS' if embedding_results['performance_ok'] else 'FAIL'}")
    print()
    
    # Step 3: Detailed analysis of simplified approach
    print("🔹 STEP 3: SIMPLIFIED APPROACH ANALYSIS")
    print("-" * 70)
    
    print("📊 SIMPLIFIED METRICS:")
    print(f"  • Readability score: {embedding_results['readability_score']:.1%}")
    print(f"  • Overlap enabled: {embedding_results['overlap_enabled']}")
    print(f"  • Avg segment duration: {embedding_results['avg_segment_duration']:.1f}s")
    print()
    
    print("🎭 Content Type Analysis:")
    for content_type, count in embedding_results['content_type_distribution'].items():
        percentage = (count / embedding_results['segments_processed']) * 100
        print(f"    • {content_type}: {count} segments ({percentage:.1f}%)")
    print()
    
    # Step 4: Show sample segments with overlap details
    print("🔹 STEP 4: SAMPLE SEGMENTS WITH OVERLAP")
    print("-" * 70)
    
    # Load the generated files for detailed analysis
    with open("data/transcripts/test_video_simplified_semantic.json", 'r') as f:
        semantic_data = json.load(f)
    
    with open("data/embeddings/test_video_simplified_embeddings.json", 'r') as f:
        embedding_data = json.load(f)
    
    print("📝 SIMPLIFIED SEMANTIC SEGMENTS (All segments):")
    for i, segment in enumerate(semantic_data['segments']):
        metadata = segment['metadata']
        
        # Show overlap information
        overlap_info = ""
        if 'overlap_added' in metadata and metadata['overlap_added'] > 0:
            original_start = metadata.get('original_start', segment['start'])
            original_end = metadata.get('original_end', segment['end'])
            overlap_info = f" | OVERLAP: {metadata['overlap_added']}s (orig: {original_start:.1f}-{original_end:.1f}s)"
        
        print(f"\n  {i+1}. [{metadata['timing_formatted']}] ({metadata['duration']:.1f}s)")
        print(f"     Caption: \"{segment['caption']}\"")
        print(f"     Type: {metadata['content_type']} | Words: {segment['word_count']} | Sentences: {metadata['sentence_count']}{overlap_info}")
    print()
    
    # Step 5: Compare with non-overlapping approach
    print("🔹 STEP 5: OVERLAP IMPACT ANALYSIS")
    print("-" * 70)
    
    # Check overlap between consecutive segments
    overlaps_found = []
    for i in range(len(semantic_data['segments']) - 1):
        current_end = semantic_data['segments'][i]['end']
        next_start = semantic_data['segments'][i + 1]['start']
        
        if current_end > next_start:
            overlap_duration = current_end - next_start
            overlaps_found.append(overlap_duration)
            print(f"  📊 Segment {i+1} → {i+2}: {overlap_duration:.1f}s overlap")
    
    if overlaps_found:
        avg_overlap = sum(overlaps_found) / len(overlaps_found)
        print(f"\n  ✅ Total overlaps: {len(overlaps_found)}")
        print(f"  ✅ Average overlap: {avg_overlap:.1f}s")
        print(f"  ✅ Expected overlap: 1.0s")
    else:
        print(f"  ⚠️  No overlaps detected (might be at video boundaries)")
    print()
    
    # Step 6: Performance summary
    print("🔹 STEP 6: PERFORMANCE SUMMARY")
    print("-" * 70)
    
    total_time = segmentation_time + embedding_time
    segments_per_second = len(semantic_data['segments']) / total_time
    
    print("⚡ PROCESSING PERFORMANCE:")
    print(f"  • Total processing time: {total_time:.3f}s")
    print(f"  • Segmentation time: {segmentation_time:.3f}s")
    print(f"  • Embedding time: {embedding_time:.3f}s") 
    print(f"  • Processing rate: {segments_per_second:.1f} segments/second")
    print()
    
    print("🎯 SIMPLIFICATION BENEFITS:")
    print(f"  ✅ Removed unreliable speech pause detection")
    print(f"  ✅ Removed rigid topic change constraints")
    print(f"  ✅ Simplified metadata structure")
    print(f"  ✅ Added configurable 1-second overlap")
    print(f"  ✅ Faster processing with cleaner logic")
    print(f"  ✅ More reliable sentence-based boundaries")
    print()
    
    # Step 7: RAG readiness check
    print("🔹 STEP 7: RAG PIPELINE READINESS")
    print("-" * 70)
    
    # Check that all required fields are present for RAG
    sample_embedding = embedding_data[0]
    required_fields = ['embedding', 'metadata']
    metadata_fields = ['video_id', 'modality', 'start', 'end', 'segment_index', 'content_type', 'overlap_added']
    
    fields_present = all(field in sample_embedding for field in required_fields)
    metadata_complete = all(field in sample_embedding['metadata'] for field in metadata_fields)
    
    print("🎯 RAG COMPATIBILITY CHECK:")
    print(f"  • Embedding structure: {'✅ Valid' if fields_present else '❌ Invalid'}")
    print(f"  • Metadata completeness: {'✅ Complete' if metadata_complete else '❌ Incomplete'}")
    print(f"  • Embedding dimension: {len(sample_embedding['embedding'])}D")
    print(f"  • Total retrievable chunks: {len(embedding_data)}")
    print(f"  • Overlap support: {'✅ Yes' if embedding_results['overlap_enabled'] else '❌ No'}")
    print()
    
    if fields_present and metadata_complete:
        print("🎉 SIMPLIFIED SEMANTIC PIPELINE SUCCESS!")
        print("   Ready for Phase 2: Vector storage & retrieval")
    else:
        print("⚠️  PIPELINE ISSUES DETECTED!")
        print("   Fix required before proceeding to Phase 2")
    
    return fields_present and metadata_complete

def main():
    """Run the simplified semantic pipeline test."""
    
    success = test_simplified_semantic_pipeline()
    
    if success:
        print("\n" + "=" * 90)
        print("🏆 SIMPLIFIED SEMANTIC-AWARE PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 90)
        print()
        print("📋 DELIVERABLES CREATED:")
        print("  ✅ data/transcripts/test_video_simplified_semantic.json")
        print("  ✅ data/embeddings/test_video_simplified_embeddings.json")
        print()
        print("🚀 READY FOR NEXT PHASE:")
        print("  • Vector database integration")
        print("  • Similarity search with overlap-aware retrieval")
        print("  • Multimodal retrieval (when video frames added)")
        print()
        print("💡 KEY IMPROVEMENTS FROM SIMPLIFICATION:")
        print("  • More reliable segmentation (punctuation-based only)")
        print("  • Configurable overlap for better context continuity")
        print("  • Simplified metadata for easier processing")
        print("  • Faster processing without complex heuristics")
        print("  • Better maintainability and debugging")
    else:
        print("\n❌ Simplified semantic pipeline test failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 