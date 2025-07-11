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
    """Lean Video RAG pipeline with interactive querying."""
    
    def __init__(self, video_path: str = "test_video.mp4"):
        self.video_path = Path(video_path)
        self.video_id = self.video_path.stem
        self.data_dir = Path("data")
        
        # Ensure directories exist
        for subdir in ["transcripts", "embeddings", "frames"]:
            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 Processing: {self.video_path}")
    
    def run_phase1(self) -> bool:
        """Phase 1: Audio → Transcription → Segmentation → Embedding"""
        print("\n🎵 Phase 1: Audio Processing...")
        
        try:
            # Transcription
            generator = VideoTranscriptGenerator(whisper_model="base")
            transcript_data = generator.process_video(str(self.video_path), str(self.data_dir / "transcripts"))
            transcript_file = self.data_dir / "transcripts" / f"{self.video_id}.json"
            
            # Semantic segmentation
            semantic_file = self.data_dir / "transcripts" / f"{self.video_id}_semantic.json"
            semantic_data = process_transcript_file_semantic(
                str(transcript_file), str(semantic_file),
                min_duration=5.0, max_duration=15.0, overlap_duration=1.0
            )
            
            # Text embedding
            embeddings_file = self.data_dir / "embeddings" / f"{self.video_id}_embeddings.json"
            embedding_results = process_semantic_segmented_file(
                str(semantic_file), str(embeddings_file), batch_size=16
            )
            
            print(f"✅ Audio: {semantic_data['total_segments']} segments, {embedding_results['embeddings_generated']} embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Phase 1 failed: {e}")
            return False
    
    def run_phase2(self) -> bool:
        """Phase 2: Frame Extraction → Embedding"""
        print("🖼️  Phase 2: Frame Processing...")
        
        try:
            # Frame sampling
            frame_sampler = FrameSampler(frames_dir=str(self.data_dir / "frames"), interval=10)
            frame_metadata = frame_sampler.sample_frames(str(self.video_path), self.video_id)
            
            # Frame embedding
            frame_embedder = FrameEmbedder(model_name="ViT-B-32", pretrained="openai")
            frame_embeddings = frame_embedder.embed_frames(frame_metadata, batch_size=32)
            
            # Save embeddings
            embeddings_list = [embedding.to_dict() for embedding in frame_embeddings]
            frame_embeddings_file = self.data_dir / "embeddings" / f"{self.video_id}_frame_embeddings.json"
            with open(frame_embeddings_file, 'w') as f:
                json.dump(embeddings_list, f, indent=2)
            
            print(f"✅ Frames: {len(frame_metadata)} extracted, {len(frame_embeddings)} embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Phase 2 failed: {e}")
            return False
    
    def run_phase3(self) -> bool:
        """Phase 3: ChromaDB Ingestion"""
        print("🗃️  Phase 3: Database Ingestion...")
        
        try:
            # Initialize ChromaDB
            vector_client = VectorStoreClient(persist_directory="data/chroma")
            ingestor = BatchIngestor(vector_client=vector_client, batch_size=20)
            
            # Clear existing data
            vector_client.delete_by_video_id(self.video_id)
            
            # Ingest embeddings
            phase1_file = self.data_dir / "embeddings" / f"{self.video_id}_embeddings.json"
            phase2_file = self.data_dir / "embeddings" / f"{self.video_id}_frame_embeddings.json"
            
            phase1_result = ingestor.ingest_phase1_embeddings(str(phase1_file))
            phase2_result = ingestor.ingest_phase2_embeddings(str(phase2_file))
            
            collection_info = vector_client.get_collection_info()
            
            print(f"✅ Database: {phase1_result.segments_processed} audio + {phase2_result.segments_processed} frames = {collection_info['count']} total")
            return phase1_result.success and phase2_result.success
            
        except Exception as e:
            print(f"❌ Phase 3 failed: {e}")
            return False
    
    def save_results_to_json(self, query: str, documents: List, search_time: float) -> str:
        """Save search results with metadata AND actual content for LLM usage."""
        results_data = {
            "query": query,
            "search_time_seconds": search_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_id": self.video_id,
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
        
        results_data["llm_ready_summary"] = {
            "total_audio_segments": len(audio_segments),
            "total_frame_segments": len(frame_segments),
            "total_words_available": sum(doc.metadata.get('word_count', 0) for doc in audio_segments),
            "total_content_duration": sum(doc.metadata.get('duration', 0) for doc in documents),
            "content_available": True,
            "citations_included": True,
            "ready_for_llm_processing": True,
            "content_summary": {
                "has_transcript_text": len(audio_segments) > 0,
                "has_visual_frames": len(frame_segments) > 0,
                "longest_audio_segment": max((doc.metadata.get('word_count', 0) for doc in audio_segments), default=0),
                "time_range": f"{min(doc.metadata.get('start', 0) for doc in documents):.1f}s - {max(doc.metadata.get('end', 0) for doc in documents):.1f}s" if documents else "0s - 0s"
            }
        }
        
        # ✅ ADDED: LLM instruction template
        results_data["llm_instructions"] = {
            "usage": "Use the 'llm_content' field for each result to access full transcript text or frame information",
            "citations": "Use 'citation_format' for proper source attribution in responses",
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
        
        if not self.video_path.exists():
            print(f"❌ Video file not found: {self.video_path}")
            return False
        
        print(f"📁 Video size: {self.video_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Execute phases
        start_time = time.time()
        
        phases = [
            ("Phase 1", self.run_phase1),
            ("Phase 2", self.run_phase2),
            ("Phase 3", self.run_phase3)
        ]
        
        for phase_name, phase_func in phases:
            if not phase_func():
                print(f"❌ {phase_name} failed - stopping pipeline")
                return False
        
        total_time = time.time() - start_time
        print(f"\n🎉 Pipeline completed in {total_time:.1f}s")
        print("✅ Video content indexed and ready for querying")
        
        # Start interactive querying
        self.interactive_query_loop()
        
        return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lean Video RAG Pipeline")
    parser.add_argument("--video", default="test_video.mp4", help="Video file to process")
    
    args = parser.parse_args()
    
    driver = LeanVideoRAGDriver(video_path=args.video)
    success = driver.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 