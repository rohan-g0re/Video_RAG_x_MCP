#!/usr/bin/env python3
"""
Lean Video RAG Pipeline Driver

Executes Phases 1-3, then provides interactive CLI for queries.
Results saved to JSON with full metadata for verification.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Phase imports
from phase1_audio.extract_transcribe import VideoTranscriptGenerator
from phase1_audio.segment_transcript_semantic import process_transcript_file_semantic
from phase1_audio.embed_text_semantic import process_semantic_segmented_file
from phase2_visual.sample_frames import FrameSampler
from phase2_visual.embed_frames import FrameEmbedder

# Phase 3, 4, and 5 imports
try:
    from src.phase3_db.client import VectorStoreClient
    from src.phase3_db.ingest import BatchIngestor
    from phase4_retriever import search_videos, Retriever
    from phase5_generation import QAService
    COMPONENTS_AVAILABLE = True
    PHASE5_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required components not available: {e}")
    COMPONENTS_AVAILABLE = False
    PHASE5_AVAILABLE = False

import logging
# Suppress INFO logs for cleaner terminal output during LLM debugging
logging.basicConfig(level=logging.ERROR)  # Only show errors
logging.getLogger("phase1_audio").setLevel(logging.ERROR)
logging.getLogger("src.phase3_db").setLevel(logging.ERROR)
logging.getLogger("phase4_retriever").setLevel(logging.ERROR)
logging.getLogger("src.phase5_generation").setLevel(logging.ERROR)


class LeanVideoRAGDriver:
    """Lean Video RAG pipeline with interactive querying - supports multiple videos."""
    
    def __init__(self, videos_dir: str = "videos"):
        self.videos_dir = Path(videos_dir)
        self.data_dir = Path("data")
        
        # Ensure directories exist
        for subdir in ["transcripts", "embeddings", "frames"]:
            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Find all video files in the videos directory
        self.video_files = []
        if self.videos_dir.exists():
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            for ext in video_extensions:
                self.video_files.extend(self.videos_dir.glob(f"*{ext}"))
        
        if not self.video_files:
            raise ValueError(f"No video files found in {self.videos_dir}")
        
        print(f"🎬 Found {len(self.video_files)} video(s) to process:")
        for video in self.video_files:
            print(f"   - {video.name} ({video.stat().st_size / (1024*1024):.1f} MB)")
    
    def run_phase1_for_video(self, video_path: Path) -> bool:
        """Phase 1: Audio → Transcription → Segmentation → Embedding for a single video"""
        video_id = video_path.stem
        print(f"\n🎵 Phase 1: Audio Processing for {video_id}...")
        
        try:
            # Transcription
            generator = VideoTranscriptGenerator(whisper_model="base")
            transcript_data = generator.process_video(str(video_path), str(self.data_dir / "transcripts"))
            transcript_file = self.data_dir / "transcripts" / f"{video_id}.json"
            
            # Semantic segmentation
            semantic_file = self.data_dir / "transcripts" / f"{video_id}_semantic.json"
            semantic_data = process_transcript_file_semantic(
                str(transcript_file), str(semantic_file),
                min_duration=5.0, max_duration=15.0, overlap_duration=1.0
            )
            
            # Text embedding
            embeddings_file = self.data_dir / "embeddings" / f"{video_id}_embeddings.json"
            embedding_results = process_semantic_segmented_file(
                str(semantic_file), str(embeddings_file), batch_size=16
            )
            
            print(f"✅ Audio ({video_id}): {semantic_data['total_segments']} segments, {embedding_results['embeddings_generated']} embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Phase 1 failed for {video_id}: {e}")
            return False
    
    def run_phase2_for_video(self, video_path: Path) -> bool:
        """Phase 2: Frame Extraction → Embedding for a single video"""
        video_id = video_path.stem
        print(f"🖼️  Phase 2: Frame Processing for {video_id}...")
        
        try:
            # Frame sampling
            frame_sampler = FrameSampler(frames_dir=str(self.data_dir / "frames"), interval=10)
            frame_metadata = frame_sampler.sample_frames(str(video_path), video_id)
            
            # Frame embedding
            frame_embedder = FrameEmbedder(model_name="ViT-B-32", pretrained="openai")
            frame_embeddings = frame_embedder.embed_frames(frame_metadata, batch_size=32)
            
            # Save embeddings
            embeddings_list = [embedding.to_dict() for embedding in frame_embeddings]
            frame_embeddings_file = self.data_dir / "embeddings" / f"{video_id}_frame_embeddings.json"
            with open(frame_embeddings_file, 'w') as f:
                json.dump(embeddings_list, f, indent=2)
            
            print(f"✅ Frames ({video_id}): {len(frame_metadata)} extracted, {len(frame_embeddings)} embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Phase 2 failed for {video_id}: {e}")
            return False
    
    def run_phase3_for_video(self, video_path: Path, clear_existing: bool = False) -> bool:
        """Phase 3: ChromaDB Ingestion for a single video"""
        video_id = video_path.stem
        print(f"🗃️  Phase 3: Database Ingestion for {video_id}...")
        
        try:
            # Initialize ChromaDB
            vector_client = VectorStoreClient(persist_directory="data/chroma")
            ingestor = BatchIngestor(vector_client=vector_client, batch_size=20)
            
            # Clear existing data only for the first video or if explicitly requested
            if clear_existing:
                print(f"🗑️  Clearing existing data for {video_id}...")
                vector_client.delete_by_video_id(video_id)
            
            # Ingest embeddings
            phase1_file = self.data_dir / "embeddings" / f"{video_id}_embeddings.json"
            phase2_file = self.data_dir / "embeddings" / f"{video_id}_frame_embeddings.json"
            
            phase1_result = ingestor.ingest_phase1_embeddings(str(phase1_file))
            phase2_result = ingestor.ingest_phase2_embeddings(str(phase2_file))
            
            collection_info = vector_client.get_collection_info()
            
            print(f"✅ Database ({video_id}): {phase1_result.segments_processed} audio + {phase2_result.segments_processed} frames")
            print(f"📊 Total in collection: {collection_info['count']} vectors")
            return phase1_result.success and phase2_result.success
            
        except Exception as e:
            print(f"❌ Phase 3 failed for {video_id}: {e}")
            return False
    
    def run_phases_for_all_videos(self) -> bool:
        """Run phases 1-3 for all videos in the videos directory"""
        print(f"🚀 Starting multi-video processing pipeline...")
        print(f"📁 Processing {len(self.video_files)} video(s)")
        
        total_processed = 0
        total_failed = 0
        
        for i, video_path in enumerate(self.video_files):
            video_id = video_path.stem
            print(f"\n{'='*60}")
            print(f"🎬 Processing Video {i+1}/{len(self.video_files)}: {video_id}")
            print(f"{'='*60}")
            
            # Run phases 1-3 for this video
            phases_success = []
            
            # Phase 1: Audio processing
            phases_success.append(self.run_phase1_for_video(video_path))
            
            # Phase 2: Frame processing
            phases_success.append(self.run_phase2_for_video(video_path))
            
            # Phase 3: Database ingestion (clear existing data only for first video)
            clear_existing = (i == 0)  # Only clear for first video
            phases_success.append(self.run_phase3_for_video(video_path, clear_existing))
            
            # Check if all phases succeeded for this video
            if all(phases_success):
                print(f"✅ Video {video_id} processed successfully")
                total_processed += 1
            else:
                print(f"❌ Video {video_id} failed in one or more phases")
                total_failed += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 MULTI-VIDEO PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully processed: {total_processed} videos")
        print(f"❌ Failed: {total_failed} videos")
        print(f"📊 Total videos: {len(self.video_files)}")
        
        # Get final database stats
        if total_processed > 0:
            try:
                vector_client = VectorStoreClient(persist_directory="data/chroma")
                collection_info = vector_client.get_collection_info()
                video_list = vector_client.list_videos()
                
                print(f"\n🗃️  FINAL DATABASE STATUS:")
                print(f"   Total vectors: {collection_info['count']}")
                print(f"   Videos in database: {len(video_list)}")
                for video_id in video_list:
                    print(f"     - {video_id}")
                    
            except Exception as e:
                print(f"⚠️  Could not get final database stats: {e}")
        
        return total_processed > 0 and total_failed == 0
    
    def save_results_to_json(self, query: str, documents: List, search_time: float) -> str:
        """Save search results with metadata AND actual content for LLM usage."""
        # Extract all unique video IDs from documents
        video_ids = list(set(doc.metadata.get('video_id', 'unknown') for doc in documents))
        
        results_data = {
            "query": query,
            "search_time_seconds": search_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_ids": video_ids,  # Changed to support multiple videos
            "total_results": len(documents),
            "results": []
        }
        
        for i, doc in enumerate(documents):
            result_data = {
                "rank": i + 1,
                "content": doc.page_content,  # ✅ ACTUAL transcript text or frame description
                "metadata": dict(doc.metadata),
                "timing": doc.get_timing_info(),
                "modality": doc.metadata.get('modality'),
                "is_audio": doc.is_audio_segment(),
                "is_frame": doc.is_frame_segment(),
                # ✅ ADDED: Full content details for LLM usage
                "llm_content": {
                    "full_text": doc.page_content if doc.is_audio_segment() else None,
                    "frame_path": doc.metadata.get('path') if doc.is_frame_segment() else None,
                    "frame_description": f"Visual frame from {doc.get_timing_info()}" if doc.is_frame_segment() else None,
                    "citation_format": f"[{doc.get_timing_info()}] {doc.metadata.get('modality')} from {doc.metadata.get('video_id')}",
                    "content_type": doc.metadata.get('content_type', 'unknown'),
                    "word_count": doc.metadata.get('word_count'),
                    "duration_seconds": doc.metadata.get('duration')
                },
                # ✅ ADDED: Complete context for LLM processing
                "context_for_llm": {
                    "video_id": doc.metadata.get('video_id'),
                    "segment_timing": doc.get_timing_info(), 
                    "start_time": doc.metadata.get('start'),
                    "end_time": doc.metadata.get('end'),
                    "modality_type": doc.metadata.get('modality'),
                    "has_actual_content": bool(doc.page_content and len(doc.page_content.strip()) > 0),
                    "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                }
            }
            results_data["results"].append(result_data)
        
        # ✅ ADDED: Summary for LLM usage with content analysis
        audio_segments = [doc for doc in documents if doc.is_audio_segment()]
        frame_segments = [doc for doc in documents if doc.is_frame_segment()]
        
        # Group by video for multi-video analysis
        video_breakdown = {}
        for doc in documents:
            video_id = doc.metadata.get('video_id', 'unknown')
            if video_id not in video_breakdown:
                video_breakdown[video_id] = {"audio": 0, "frames": 0}
            if doc.is_audio_segment():
                video_breakdown[video_id]["audio"] += 1
            else:
                video_breakdown[video_id]["frames"] += 1
        
        results_data["llm_ready_summary"] = {
            "total_audio_segments": len(audio_segments),
            "total_frame_segments": len(frame_segments),
            "total_words_available": sum(doc.metadata.get('word_count', 0) for doc in audio_segments),
            "total_content_duration": sum(doc.metadata.get('duration', 0) for doc in documents),
            "content_available": True,
            "citations_included": True,
            "ready_for_llm_processing": True,
            "multi_video_analysis": True,  # New field for multi-video support
            "video_breakdown": video_breakdown,  # Breakdown by video
            "content_summary": {
                "has_transcript_text": len(audio_segments) > 0,
                "has_visual_frames": len(frame_segments) > 0,
                "longest_audio_segment": max((doc.metadata.get('word_count', 0) for doc in audio_segments), default=0),
                "time_range": f"{min(doc.metadata.get('start', 0) for doc in documents):.1f}s - {max(doc.metadata.get('end', 0) for doc in documents):.1f}s" if documents else "0s - 0s",
                "videos_included": len(video_ids)
            }
        }
        
        # ✅ ADDED: LLM instruction template
        results_data["llm_instructions"] = {
            "usage": "Use the 'llm_content' field for each result to access full transcript text or frame information",
            "citations": "Use 'citation_format' for proper source attribution in responses",
            "multi_video_note": "Results may span multiple videos - check video_id in metadata for proper attribution",
            "content_types": [
                "full_text: Complete transcript content for audio segments",
                "frame_path: File path to visual frame image",
                "frame_description: Descriptive text for visual content"
            ],
            "example_usage": "To answer questions, combine multiple audio segments' full_text and reference frame_path for visual elements"
        }
        
        # Save to file
        safe_query = "".join(c if c.isalnum() or c in " -_" else "" for c in query)[:50]
        filename = f"search_results_{safe_query.replace(' ', '_')}_{int(time.time())}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return str(filepath)
    
    def interactive_query_loop(self):
        """Interactive CLI for user queries with Phase 5 LLM generation."""
        print("\n" + "="*60)
        print("🔍 INTERACTIVE QUERY MODE (with LLM Generation)")
        print("="*60)
        print("Enter natural language queries about your video content.")
        print("Commands: 'quit', 'exit', 'help', 'stats', 'llm' (toggle LLM mode)")
        print("="*60)
        
        if not COMPONENTS_AVAILABLE:
            print("❌ Phase 4 components not available for querying")
            return
        
        # Initialize retriever
        try:
            retriever = Retriever()
            stats = retriever.get_stats()
            print(f"📊 Database: {stats.total_documents} documents ({stats.audio_documents} audio, {stats.frame_documents} frames)")
        except Exception as e:
            print(f"❌ Cannot initialize retriever: {e}")
            return
        
        # Initialize QA Service for LLM generation
        qa_service = None
        llm_mode = True  # Default to LLM mode enabled
        
        if PHASE5_AVAILABLE:
            try:
                qa_service = QAService(retriever=retriever)
                print(f"🤖 LLM Generation: ENABLED (ChatGroq Llama-3.1-8b-instant)")
                print()
            except Exception as e:
                print(f"⚠️  LLM Generation unavailable: {e}")
                print(f"📋 Falling back to retrieval-only mode")
                llm_mode = False
                print()
        else:
            print(f"⚠️  Phase 5 not available - retrieval-only mode")
            llm_mode = False
            print()
        
        while True:
            try:
                mode_indicator = "🤖" if (llm_mode and qa_service) else "📋"
                query = input(f"{mode_indicator} Query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("  • Commands: quit, exit, stats, llm")
                    print("  • LLM mode: Generate AI answers with citations")
                    print("  • Retrieval mode: Show raw video segments only")
                    continue
                
                if query.lower() == 'llm':
                    if qa_service:
                        llm_mode = not llm_mode
                        status = "ENABLED" if llm_mode else "DISABLED"
                        print(f"🤖 LLM Generation: {status}")
                    else:
                        print("⚠️  LLM Generation not available")
                    print()
                    continue
                
                if query.lower() == 'stats':
                    stats = retriever.get_stats()
                    print(f"\n📊 STATISTICS:")
                    print(f"   Total documents: {stats.total_documents}")
                    print(f"   Audio segments: {stats.audio_documents}")
                    print(f"   Frame segments: {stats.frame_documents}")
                    print(f"   Videos: {stats.total_videos}")
                    print(f"   LLM Mode: {'ENABLED' if llm_mode and qa_service else 'DISABLED'}")
                    print()
                    continue
                
                # Execute search with optional LLM generation
                start_time = time.time()
                
                if llm_mode and qa_service:
                    # Phase 5: LLM Generation Mode
                    try:
                        from phase5_generation.qa_service import QARequest
                        import asyncio
                        
                        # Create QA request
                        qa_request = QARequest(question=query, k=5)
                        
                        # Generate LLM response
                        print(f"\n🤖 Generating AI response for: '{query}'")
                        response = asyncio.run(qa_service.process_question(qa_request))
                        
                        total_time = time.time() - start_time
                        
                        # Display LLM answer
                        print(f"\n✅ AI Answer (generated in {total_time:.3f}s):")
                        print("=" * 60)
                        print(response.answer)
                        print("=" * 60)
                        
                        # Display sources
                        if response.sources:
                            print(f"\n📚 Sources ({len(response.sources)} segments):")
                            for i, source in enumerate(response.sources, 1):
                                timing = f"{source.start:.1f}s-{source.end:.1f}s"
                                print(f"   {i}. [{source.video_id}: {timing}] {source.modality.upper()}")
                                if source.content_preview:
                                    preview = source.content_preview[:60] + "..." if len(source.content_preview) > 60 else source.content_preview
                                    print(f"      💬 {preview}")
                        
                        # Performance metrics
                        print(f"\n⏱️  Performance:")
                        print(f"   Retrieval: {response.retrieval_time_seconds:.3f}s")
                        print(f"   Generation: {response.generation_time_seconds:.3f}s")
                        print(f"   Total: {response.processing_time_seconds:.3f}s")
                        
                    except Exception as e:
                        print(f"❌ LLM Generation failed: {e}")
                        print("📋 Falling back to retrieval mode...")
                        llm_mode = False
                
                if not llm_mode or not qa_service:
                    # Phase 4: Retrieval-Only Mode  
                    documents = search_videos(query, k=5)
                    search_time = time.time() - start_time
                    
                    if documents:
                        print(f"\n✅ Found {len(documents)} results in {search_time:.3f}s:")
                        
                        for i, doc in enumerate(documents, 1):
                            timing = doc.get_timing_info()
                            modality = doc.metadata.get('modality', 'unknown')
                            
                            print(f"\n   {i}. [{timing}] {modality.upper()}")
                            
                            if doc.is_audio_segment():
                                content = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                                print(f"      📝 {content}")
                            else:
                                print(f"      🖼️  {doc.page_content}")
                            
                            # Show key metadata
                            if 'word_count' in doc.metadata:
                                print(f"      📊 Words: {doc.metadata['word_count']}")
                        
                        # Save to JSON
                        json_file = self.save_results_to_json(query, documents, search_time)
                        print(f"\n💾 Results saved to: {json_file}")
                        
                    else:
                        print(f"⚠️  No results found in {search_time:.3f}s")
            
                print()
            
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
    
    def run(self) -> bool:
        """Run complete pipeline followed by interactive queries."""
        if not COMPONENTS_AVAILABLE:
            print("❌ Required components not available")
            return False
        
        # Run multi-video processing for all videos found
        start_time = time.time()
        
        # Process all videos through phases 1-3
        success = self.run_phases_for_all_videos()
        
        if not success:
            print("❌ Multi-video processing failed")
            return False
        
        total_time = time.time() - start_time
        print(f"\n🎉 Multi-video pipeline completed in {total_time:.1f}s")
        print("✅ All video content indexed and ready for querying")
        
        # Start interactive querying
        self.interactive_query_loop()
        
        return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lean Video RAG Pipeline")
    parser.add_argument("--video", default="videos", help="Video file or directory to process")
    
    args = parser.parse_args()
    
    driver = LeanVideoRAGDriver(videos_dir=args.video)
    success = driver.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 