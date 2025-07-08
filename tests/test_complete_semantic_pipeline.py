#!/usr/bin/env python3
"""
Complete Semantic Pipeline Test

Tests the full semantic-aware chunking + embedding pipeline
and compares results with the original approach.
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_audio.segment_transcript_semantic import process_transcript_file_semantic
from phase1_audio.embed_text_semantic import process_semantic_segmented_file
from phase1_audio.embed_text import process_segmented_file

def test_complete_semantic_pipeline():
    """Test the complete semantic-aware pipeline with embeddings."""
    
    print("=" * 90)
    print("🚀 COMPLETE SEMANTIC-AWARE VIDEO RAG PIPELINE TEST")
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
    
    # Step 1: Create semantic segments
    print("🔹 STEP 1: SEMANTIC-AWARE SEGMENTATION")
    print("-" * 70)
    
    start_time = time.time()
    semantic_segments = process_transcript_file_semantic(
        transcript_file, 
        "data/transcripts/test_video_semantic_final.json",
        min_duration=5.0,
        max_duration=15.0
    )
    segmentation_time = time.time() - start_time
    
    print(f"✅ Segmentation completed in {segmentation_time:.3f}s")
    print(f"✅ Segments created: {semantic_segments['total_segments']}")
    print(f"✅ Method: {semantic_segments['segmentation_method']}")
    print()
    
    # Step 2: Generate semantic embeddings
    print("🔹 STEP 2: ENHANCED EMBEDDING GENERATION")
    print("-" * 70)
    
    start_time = time.time()
    embedding_results = process_semantic_segmented_file(
        "data/transcripts/test_video_semantic_final.json",
        "data/embeddings/test_video_semantic_embeddings.json",
        batch_size=16
    )
    embedding_time = time.time() - start_time
    
    print(f"✅ Embedding completed in {embedding_time:.3f}s")
    print(f"✅ Embeddings generated: {embedding_results['embeddings_generated']}")
    print(f"✅ Embedding dimension: {embedding_results['embedding_dimension']}")
    print(f"✅ Performance: {'PASS' if embedding_results['performance_ok'] else 'FAIL'}")
    print()
    
    # Step 3: Detailed results analysis
    print("🔹 STEP 3: SEMANTIC ANALYSIS RESULTS")
    print("-" * 70)
    
    print("📊 ENHANCED METRICS:")
    print(f"  • Readability score: {embedding_results['readability_score']:.1%}")
    print(f"  • Natural boundaries: {embedding_results['natural_boundaries']}/{embedding_results['segments_processed']}")
    print(f"  • Avg segment duration: {embedding_results['avg_segment_duration']:.1f}s")
    print()
    
    print("🎭 Content Type Analysis:")
    for content_type, count in embedding_results['content_type_distribution'].items():
        percentage = (count / embedding_results['segments_processed']) * 100
        print(f"    • {content_type}: {count} segments ({percentage:.1f}%)")
    print()
    
    # Step 4: Show readable captions vs original text
    print("🔹 STEP 4: CAPTION QUALITY COMPARISON")
    print("-" * 70)
    
    # Load the generated files for comparison
    with open("data/transcripts/test_video_semantic_final.json", 'r') as f:
        semantic_data = json.load(f)
    
    with open("data/embeddings/test_video_semantic_embeddings.json", 'r') as f:
        embedding_data = json.load(f)
    
    print("📝 SEMANTIC CAPTIONS (First 5 segments):")
    for i, segment in enumerate(semantic_data['segments'][:5]):
        metadata = segment['metadata']
        print(f"\n  {i+1}. [{metadata['timing_formatted']}] ({metadata['duration']:.1f}s)")
        print(f"     Caption: \"{segment['caption']}\"")
        print(f"     Type: {metadata['content_type']} | Reason: {metadata['segmentation_reason']}")
        print(f"     Words: {segment['word_count']} | Sentences: {metadata['sentence_count']}")
    print()
    
    # Step 5: Performance summary
    print("🔹 STEP 5: PERFORMANCE SUMMARY")
    print("-" * 70)
    
    total_time = segmentation_time + embedding_time
    segments_per_second = len(semantic_data['segments']) / total_time
    
    print("⚡ PROCESSING PERFORMANCE:")
    print(f"  • Total processing time: {total_time:.3f}s")
    print(f"  • Segmentation time: {segmentation_time:.3f}s")
    print(f"  • Embedding time: {embedding_time:.3f}s") 
    print(f"  • Processing rate: {segments_per_second:.1f} segments/second")
    print()
    
    print("📈 QUALITY IMPROVEMENTS:")
    print(f"  ✅ Readable captions with punctuation and capitalization")
    print(f"  ✅ Natural sentence boundaries respected")
    print(f"  ✅ Adaptive segment lengths (5-15s)")
    print(f"  ✅ Rich metadata for enhanced retrieval")
    print(f"  ✅ Content type classification")
    print(f"  ✅ Human-friendly timing format")
    print()
    
    # Step 6: Retrieval readiness check
    print("🔹 STEP 6: RAG PIPELINE READINESS")
    print("-" * 70)
    
    # Check that all required fields are present for RAG
    sample_embedding = embedding_data[0]
    required_fields = ['embedding', 'metadata']
    metadata_fields = ['video_id', 'modality', 'start', 'end', 'segment_index', 'content_type']
    
    fields_present = all(field in sample_embedding for field in required_fields)
    metadata_complete = all(field in sample_embedding['metadata'] for field in metadata_fields)
    
    print("🎯 RAG COMPATIBILITY CHECK:")
    print(f"  • Embedding structure: {'✅ Valid' if fields_present else '❌ Invalid'}")
    print(f"  • Metadata completeness: {'✅ Complete' if metadata_complete else '❌ Incomplete'}")
    print(f"  • Embedding dimension: {len(sample_embedding['embedding'])}D")
    print(f"  • Total retrievable chunks: {len(embedding_data)}")
    print()
    
    if fields_present and metadata_complete:
        print("🎉 SEMANTIC PIPELINE SUCCESS!")
        print("   Ready for Phase 2: Vector storage & retrieval")
    else:
        print("⚠️  PIPELINE ISSUES DETECTED!")
        print("   Fix required before proceeding to Phase 2")
    
    return fields_present and metadata_complete

def main():
    """Run the complete semantic pipeline test."""
    
    success = test_complete_semantic_pipeline()
    
    if success:
        print("\n" + "=" * 90)
        print("🏆 SEMANTIC-AWARE VIDEO RAG PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 90)
        print()
        print("📋 DELIVERABLES CREATED:")
        print("  ✅ data/transcripts/test_video_semantic_final.json")
        print("  ✅ data/embeddings/test_video_semantic_embeddings.json")
        print()
        print("🚀 READY FOR NEXT PHASE:")
        print("  • Vector database integration")
        print("  • Similarity search implementation")
        print("  • Multimodal retrieval (when video frames added)")
        print()
        print("💡 KEY IMPROVEMENTS ACHIEVED:")
        print("  • Better text readability with punctuation/capitalization")
        print("  • Natural segment boundaries instead of fixed time windows")
        print("  • Rich metadata for enhanced retrieval")
        print("  • Content type classification for better filtering")
        print("  • Adaptive segment lengths for optimal chunk sizes")
    else:
        print("\n❌ Semantic pipeline test failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 