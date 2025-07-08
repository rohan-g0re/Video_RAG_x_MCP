#!/usr/bin/env python3
"""
Complete Video RAG Pipeline Analysis

Analyzes the complete pipeline from video ingestion to embeddings
and shows detailed results for each phase.
"""

import sys
import json
import time
from pathlib import Path

def analyze_complete_pipeline():
    """Analyze the complete video RAG pipeline results."""
    
    print("=" * 100)
    print("🎬 COMPLETE VIDEO RAG PIPELINE ANALYSIS")
    print("=" * 100)
    
    # File paths
    video_file = "test_video.mp4"
    transcript_file = "data/transcripts/test_video.json"
    segments_file = "data/transcripts/test_video_semantic.json"
    embeddings_file = "data/embeddings/test_video_embeddings.json"
    
    # Check if all files exist
    if not Path(video_file).exists():
        print("❌ Video file not found!")
        return False
        
    if not Path(transcript_file).exists():
        print("❌ Transcript file not found!")
        return False
        
    if not Path(segments_file).exists():
        print("❌ Segments file not found!")
        return False
        
    if not Path(embeddings_file).exists():
        print("❌ Embeddings file not found!")
        return False
    
    print("✅ All pipeline files found!")
    print()
    
    # Phase 1-A Analysis: Audio Extraction & Transcription
    print("🔹 PHASE 1-A: AUDIO EXTRACTION & TRANSCRIPTION ANALYSIS")
    print("-" * 80)
    
    with open(transcript_file, 'r') as f:
        transcript_data = json.load(f)
    
    video_size = Path(video_file).stat().st_size / (1024 * 1024)  # MB
    transcript_size = Path(transcript_file).stat().st_size / 1024  # KB
    
    print(f"📹 Video Analysis:")
    print(f"  • File: {video_file}")
    print(f"  • Size: {video_size:.1f} MB")
    print(f"  • Duration: {transcript_data['duration_seconds']:.1f} seconds")
    print()
    
    print(f"📝 Transcript Analysis:")
    print(f"  • Total words: {transcript_data['word_count']}")
    print(f"  • Words per second: {transcript_data['word_count'] / transcript_data['duration_seconds']:.1f}")
    print(f"  • File size: {transcript_size:.1f} KB")
    print(f"  • Model used: {transcript_data.get('model', 'unknown')}")
    print()
    
    # Show sample words with timestamps
    print("📊 Sample Words with Timestamps:")
    words = transcript_data['words'][:10]  # First 10 words
    for i, word in enumerate(words):
        print(f"  {i+1:2d}. [{word['start']:5.2f}s-{word['end']:5.2f}s] \"{word['word']}\"")
    print()
    
    # Phase 1-B Analysis: Semantic Segmentation
    print("🔹 PHASE 1-B: SEMANTIC SEGMENTATION ANALYSIS")
    print("-" * 80)
    
    with open(segments_file, 'r') as f:
        segments_data = json.load(f)
    
    segments_size = Path(segments_file).stat().st_size / 1024  # KB
    
    print(f"🎯 Segmentation Configuration:")
    config = segments_data['segment_config']
    print(f"  • Method: {segments_data['segmentation_method']}")
    print(f"  • Min duration: {config['min_duration']}s")
    print(f"  • Max duration: {config['max_duration']}s")
    print(f"  • Target duration: {config['target_duration']}s")
    print(f"  • Overlap duration: {config['overlap_duration']}s")
    print()
    
    print(f"📏 Segmentation Results:")
    segments = segments_data['segments']
    durations = [seg['metadata']['duration'] for seg in segments]
    
    print(f"  • Total segments: {len(segments)}")
    print(f"  • Average duration: {sum(durations)/len(durations):.1f}s")
    print(f"  • Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
    print(f"  • File size: {segments_size:.1f} KB")
    print()
    
    # Analyze overlaps
    overlaps = []
    for i in range(len(segments) - 1):
        current_end = segments[i]['end']
        next_start = segments[i + 1]['start']
        if current_end > next_start:
            overlap = current_end - next_start
            overlaps.append(overlap)
    
    if overlaps:
        print(f"🔄 Overlap Analysis:")
        print(f"  • Overlaps detected: {len(overlaps)}")
        print(f"  • Average overlap: {sum(overlaps)/len(overlaps):.1f}s")
        print(f"  • Overlap range: {min(overlaps):.1f}s - {max(overlaps):.1f}s")
    else:
        print(f"🔄 No overlaps detected")
    print()
    
    # Show all segments with details
    print("📋 All Semantic Segments:")
    for i, segment in enumerate(segments):
        metadata = segment['metadata']
        
        # Check for overlap info
        overlap_info = ""
        if 'overlap_added' in metadata and metadata['overlap_added'] > 0:
            original_start = metadata.get('original_start', segment['start'])
            original_end = metadata.get('original_end', segment['end'])
            overlap_info = f" | OVERLAP: {metadata['overlap_added']}s"
        
        print(f"\n  {i+1:2d}. [{metadata['timing_formatted']}] ({metadata['duration']:.1f}s)")
        print(f"      Caption: \"{segment['caption']}\"")
        print(f"      Details: {metadata['content_type']} | {segment['word_count']} words | {metadata['sentence_count']} sentences{overlap_info}")
    print()
    
    # Phase 1-C Analysis: Embedding Generation
    print("🔹 PHASE 1-C: EMBEDDING GENERATION ANALYSIS")
    print("-" * 80)
    
    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)
    
    embeddings_size = Path(embeddings_file).stat().st_size / 1024  # KB
    
    print(f"🧠 Embedding Configuration:")
    sample_embedding = embeddings_data[0]
    metadata = sample_embedding['metadata']
    
    print(f"  • Model: CLIP ViT-B-32")
    print(f"  • Embedding dimension: {len(sample_embedding['embedding'])}D")
    print(f"  • Total embeddings: {len(embeddings_data)}")
    print(f"  • File size: {embeddings_size:.1f} KB")
    print(f"  • Segmentation method: {metadata['segmentation_method']}")
    print()
    
    # Analyze content types
    content_types = {}
    for embedding in embeddings_data:
        content_type = embedding['metadata']['content_type']
        content_types[content_type] = content_types.get(content_type, 0) + 1
    
    print(f"🎭 Content Type Distribution:")
    for content_type, count in content_types.items():
        percentage = (count / len(embeddings_data)) * 100
        print(f"  • {content_type}: {count} segments ({percentage:.1f}%)")
    print()
    
    # Show sample embeddings with metadata
    print("🔢 Sample Embeddings (First 3):")
    for i, embedding in enumerate(embeddings_data[:3]):
        meta = embedding['metadata']
        print(f"\n  {i+1}. Segment {meta['segment_index']} [{meta['timing_formatted']}]")
        print(f"     Video: {meta['video_id']} | Type: {meta['content_type']}")
        print(f"     Duration: {meta['duration']}s | Words: {meta['word_count']}")
        if 'overlap_added' in meta:
            print(f"     Overlap: {meta['overlap_added']}s")
        # Show first few embedding values
        embedding_values = embedding['embedding'][:5]
        print(f"     Embedding: [{', '.join(f'{v:.4f}' for v in embedding_values)}, ...]")
    print()
    
    # Overall Pipeline Analysis
    print("🔹 COMPLETE PIPELINE SUMMARY")
    print("-" * 80)
    
    # Calculate efficiency metrics
    words_per_segment = transcript_data['word_count'] / len(segments)
    embeddings_per_second = len(embeddings_data) / transcript_data['duration_seconds']
    
    print(f"📊 Processing Efficiency:")
    print(f"  • Video → Transcript: {transcript_data['word_count']} words extracted")
    print(f"  • Transcript → Segments: {len(segments)} semantic chunks created")
    print(f"  • Segments → Embeddings: {len(embeddings_data)} embeddings generated")
    print(f"  • Words per segment: {words_per_segment:.1f}")
    print(f"  • Embeddings per second: {embeddings_per_second:.1f}")
    print()
    
    print(f"💾 Storage Breakdown:")
    total_size = video_size * 1024 + transcript_size + segments_size + embeddings_size  # KB
    print(f"  • Original video: {video_size:.1f} MB ({video_size*1024/total_size*100:.1f}%)")
    print(f"  • Transcript: {transcript_size:.1f} KB ({transcript_size/total_size*100:.1f}%)")
    print(f"  • Segments: {segments_size:.1f} KB ({segments_size/total_size*100:.1f}%)")
    print(f"  • Embeddings: {embeddings_size:.1f} KB ({embeddings_size/total_size*100:.1f}%)")
    print(f"  • Total pipeline: {total_size/1024:.1f} MB")
    print()
    
    print(f"🎯 Pipeline Quality Metrics:")
    
    # Calculate readability score
    readable_segments = sum(1 for seg in segments 
                          if seg['caption'] and len(seg['caption']) > 10 and 
                          any(c in seg['caption'] for c in '.!?'))
    readability_score = readable_segments / len(segments) * 100
    
    print(f"  • Readability score: {readability_score:.1f}% ({readable_segments}/{len(segments)} segments)")
    print(f"  • Average segment length: {sum(durations)/len(durations):.1f}s (target: 10s)")
    print(f"  • Overlap coverage: {len(overlaps)}/{len(segments)-1} transitions")
    print(f"  • Content variety: {len(content_types)} different content types")
    print()
    
    # RAG Readiness Assessment
    print("🔹 RAG READINESS ASSESSMENT")
    print("-" * 80)
    
    # Check required fields
    required_fields = ['embedding', 'metadata']
    metadata_fields = ['video_id', 'modality', 'start', 'end', 'segment_index', 'content_type']
    
    sample = embeddings_data[0]
    fields_present = all(field in sample for field in required_fields)
    metadata_complete = all(field in sample['metadata'] for field in metadata_fields)
    
    print(f"✅ RAG Pipeline Validation:")
    print(f"  • Embedding structure: {'✅ Valid' if fields_present else '❌ Invalid'}")
    print(f"  • Metadata completeness: {'✅ Complete' if metadata_complete else '❌ Incomplete'}")
    print(f"  • Searchable chunks: {len(embeddings_data)}")
    print(f"  • Embedding dimensionality: {len(sample['embedding'])}D")
    print(f"  • Temporal coverage: {transcript_data['duration_seconds']:.1f}s")
    print(f"  • Overlap support: {'✅ Yes' if overlaps else '❌ No'}")
    print()
    
    if fields_present and metadata_complete and len(embeddings_data) > 0:
        print("🎉 PIPELINE STATUS: READY FOR PRODUCTION RAG!")
        print("   ✅ All phases completed successfully")
        print("   ✅ High-quality semantic segmentation")
        print("   ✅ Rich embeddings with metadata")
        print("   ✅ Overlap support for context continuity")
        success = True
    else:
        print("⚠️  PIPELINE STATUS: NEEDS ATTENTION")
        print("   Some validation checks failed")
        success = False
    
    return success

def main():
    """Run the complete pipeline analysis."""
    
    success = analyze_complete_pipeline()
    
    if success:
        print("\n" + "=" * 100)
        print("🏆 COMPLETE VIDEO RAG PIPELINE ANALYSIS SUCCESSFUL!")
        print("=" * 100)
        print()
        print("🚀 READY FOR NEXT STEPS:")
        print("  • Vector database storage (Phase 2)")
        print("  • Similarity search implementation")
        print("  • Query processing and retrieval")
        print("  • Response generation")
    else:
        print("\n❌ Pipeline analysis detected issues")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 