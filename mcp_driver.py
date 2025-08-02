#!/usr/bin/env python3
"""
Video RAG Pipeline MCP Driver

Modified version of driver.py that runs Phases 1-4 only (no Phase 5 LLM generation).
The retrieved chunks will be provided to Claude Desktop as the MCP client.
Results saved to JSON with full metadata for MCP tool responses.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Phase 1 and 2 imports
from phase1_audio.extract_transcribe import VideoTranscriptGenerator
from phase1_audio.segment_transcript_semantic import SemanticTranscriptSegmenter
from phase1_audio.embed_text_semantic import SemanticEmbeddingProcessor
from phase2_visual.processor import FrameSampler, FrameEmbedder

# Phase 3 and 4 imports
try:
    from src.phase3_db.client import VectorStoreClient
    from src.phase3_db.ingest import BatchIngestor
    from phase4_retriever import search_videos, Retriever
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Required components not available: {e}")
    COMPONENTS_AVAILABLE = False

import logging
# Suppress INFO logs for cleaner terminal output
logging.basicConfig(level=logging.ERROR)  # Only show errors
logging.getLogger("phase1_audio").setLevel(logging.ERROR)
logging.getLogger("src.phase3_db").setLevel(logging.ERROR)
logging.getLogger("phase4_retriever").setLevel(logging.ERROR)


class VideoRAGMCPDriver:
    """Video RAG pipeline for MCP - supports multiple videos through phases 1-4 only."""
    
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
        
        print(f"VIDEO: Found {len(self.video_files)} video(s) to process:")
        for video in self.video_files:
            print(f"   - {video.name} ({video.stat().st_size / (1024*1024):.1f} MB)")
    
    def run_phase1_for_video(self, video_path: Path) -> bool:
        """Phase 1: Audio → Transcription → Segmentation → Embedding (optimized, no intermediate files)"""
        video_id = video_path.stem
        print(f"\nAUDIO: Phase 1: Audio Processing for {video_id}...")
        
        try:
            # Step 1: Transcription (in-memory only)
            generator = VideoTranscriptGenerator(whisper_model="base")
            transcript_data = generator.process_video(str(video_path), save_file=False)
            
            # Step 2: Semantic segmentation (in-memory only)
            segmenter = SemanticTranscriptSegmenter(
                min_duration=5.0, max_duration=15.0, overlap_duration=1.0
            )
            semantic_data = segmenter.segment_transcript(transcript_data)
            
            # Step 3: Text embedding (save only final file needed for Phase 3)
            embeddings_file = self.data_dir / "embeddings" / f"{video_id}_embeddings.json"
            processor = SemanticEmbeddingProcessor(clip_model="ViT-B-32", batch_size=16)
            embedding_results = processor.process_semantic_segments(
                semantic_data, str(embeddings_file)
            )
            
            print(f"SUCCESS: Audio ({video_id}): {semantic_data['total_segments']} segments, {embedding_results['embeddings_generated']} embeddings")
            return True
            
        except Exception as e:
            print(f"ERROR: Phase 1 failed for {video_id}: {e}")
            return False
    
    def run_phase2_for_video(self, video_path: Path) -> bool:
        """Phase 2: Frame Extraction → Embedding for a single video"""
        video_id = video_path.stem
        print(f"FRAMES: Phase 2: Frame Processing for {video_id}...")
        
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
            
            print(f"SUCCESS: Frames ({video_id}): {len(frame_metadata)} extracted, {len(frame_embeddings)} embeddings")
            return True
            
        except Exception as e:
            print(f"ERROR: Phase 2 failed for {video_id}: {e}")
            return False
    
    def run_phase3_for_video(self, video_path: Path, clear_existing: bool = False) -> bool:
        """Phase 3: ChromaDB Ingestion for a single video"""
        video_id = video_path.stem
        print(f"DATABASE: Phase 3: Database Ingestion for {video_id}...")
        
        try:
            # Initialize ChromaDB
            vector_client = VectorStoreClient(persist_directory="data/chroma")
            ingestor = BatchIngestor(vector_client=vector_client, batch_size=20)
            
            # Use combined ingestion method that automatically clears existing data
            phase1_file = self.data_dir / "embeddings" / f"{video_id}_embeddings.json"
            phase2_file = self.data_dir / "embeddings" / f"{video_id}_frame_embeddings.json"
            
            # Combined ingestion automatically clears existing data for this video_id
            combined_result = ingestor.ingest_combined_video(video_id, str(phase1_file), str(phase2_file))
            
            # Extract individual results for logging (using actual stats from combined result)
            if hasattr(ingestor, 'stats'):
                audio_segments = ingestor.stats.get('audio_segments', 0)
                visual_segments = ingestor.stats.get('visual_segments', 0)
            else:
                # Fallback estimation
                audio_segments = combined_result.segments_processed * 2 // 3  # Rough estimate
                visual_segments = combined_result.segments_processed // 3
            
            phase1_result = type('Result', (), {
                'segments_processed': audio_segments,
                'success': combined_result.success
            })()
            phase2_result = type('Result', (), {
                'segments_processed': visual_segments,
                'success': combined_result.success
            })()
            
            collection_info = vector_client.get_collection_info()
            
            print(f"SUCCESS: Database ({video_id}): {phase1_result.segments_processed} audio + {phase2_result.segments_processed} frames")
            print(f"STATS: Total in collection: {collection_info['count']} vectors")
            return phase1_result.success and phase2_result.success
            
        except Exception as e:
            print(f"ERROR: Phase 3 failed for {video_id}: {e}")
            return False
    
    def run_phases_for_all_videos(self) -> bool:
        """Run phases 1-3 for all videos in the videos directory"""
        print(f"PROCESSING: Starting multi-video processing pipeline...")
        print(f"FILES: Processing {len(self.video_files)} video(s)")
        
        total_processed = 0
        total_failed = 0
        
        for i, video_path in enumerate(self.video_files):
            video_id = video_path.stem
            print(f"\n{'='*60}")
            print(f"VIDEO: Processing Video {i+1}/{len(self.video_files)}: {video_id}")
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
                print(f"SUCCESS: Video {video_id} processed successfully")
                total_processed += 1
            else:
                print(f"ERROR: Video {video_id} failed in one or more phases")
                total_failed += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"STATS: MULTI-VIDEO PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"SUCCESS: Successfully processed: {total_processed} videos")
        print(f"ERROR: Failed: {total_failed} videos")
        print(f"STATS: Total videos: {len(self.video_files)}")
        
        # Get final database stats
        if total_processed > 0:
            try:
                vector_client = VectorStoreClient(persist_directory="data/chroma")
                collection_info = vector_client.get_collection_info()
                video_list = vector_client.list_videos()
                
                print(f"\nDATABASE: FINAL DATABASE STATUS:")
                print(f"   Total vectors: {collection_info['count']}")
                print(f"   Videos in database: {len(video_list)}")
                for video_id in video_list:
                    print(f"     - {video_id}")
                    
            except Exception as e:
                print(f"WARNING: Could not get final database stats: {e}")
        
        return total_processed > 0 and total_failed == 0
    
    def save_results_to_json(self, query: str, documents: List, search_time: float) -> str:
        """Save search results with metadata AND actual content for MCP client usage."""
        # Extract all unique video IDs from documents
        video_ids = list(set(doc.metadata.get('video_id', 'unknown') for doc in documents))
        
        results_data = {
            "query": query,
            "search_time_seconds": search_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_ids": video_ids,  # Support multiple videos
            "total_results": len(documents),
            "results": []
        }
        
        for i, doc in enumerate(documents):
            result_data = {
                "rank": i + 1,
                "content": doc.page_content,  # ACTUAL transcript text or frame description
                "metadata": dict(doc.metadata),
                "timing": doc.get_timing_info(),
                "modality": doc.metadata.get('modality'),
                "is_audio": doc.is_audio_segment(),
                "is_frame": doc.is_frame_segment(),
                # ADDED: Full content details for MCP client usage
                "mcp_content": {
                    "full_text": doc.page_content if doc.is_audio_segment() else None,
                    "frame_path": doc.metadata.get('path') if doc.is_frame_segment() else None,
                    "frame_description": f"Visual frame from {doc.get_timing_info()}" if doc.is_frame_segment() else None,
                    "citation_format": f"[{doc.get_timing_info()}] {doc.metadata.get('modality')} from {doc.metadata.get('video_id')}",
                    "content_type": doc.metadata.get('content_type', 'unknown'),
                    "word_count": doc.metadata.get('word_count'),
                    "duration_seconds": doc.metadata.get('duration')
                },
                # ADDED: Complete context for MCP client processing
                "context_for_mcp": {
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
        
        # ADDED: Summary for MCP client usage with content analysis
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
        
        results_data["mcp_ready_summary"] = {
            "total_audio_segments": len(audio_segments),
            "total_frame_segments": len(frame_segments),
            "total_words_available": sum(doc.metadata.get('word_count', 0) for doc in audio_segments),
            "total_content_duration": sum(doc.metadata.get('duration', 0) for doc in documents),
            "content_available": True,
            "citations_included": True,
            "ready_for_mcp_processing": True,
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
        
        # ADDED: MCP instruction template
        results_data["mcp_instructions"] = {
            "usage": "Use the 'mcp_content' field for each result to access full transcript text or frame information",
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
    
    def search_videos(self, query: str, k: int = 5) -> tuple[List, str]:
        """Phase 4: Search videos and return results with metadata file path."""
        if not COMPONENTS_AVAILABLE:
            raise ValueError("Phase 4 components not available for querying")
        
        try:
            # Execute search
            start_time = time.time()
            documents = search_videos(query, k=k)
            search_time = time.time() - start_time
            
            if documents:
                print(f"SUCCESS: Found {len(documents)} results in {search_time:.3f}s")
                
                # Save results to JSON with MCP-ready format
                json_file = self.save_results_to_json(query, documents, search_time)
                print(f"SAVED: Results saved to: {json_file}")
                
                return documents, json_file
            else:
                print(f"WARNING: No results found in {search_time:.3f}s")
                return [], ""
                
        except Exception as e:
            print(f"ERROR: Search failed: {e}")
            raise
    
    def intelligent_search(self, query: str, k: int = 5) -> tuple[List, str, Dict[str, Any]]:
        """
        Intelligent search that automatically handles video processing if needed.
        
        Returns:
            - documents: List of search results
            - json_file: Path to detailed results file
            - status: Dict with success status and metadata
        """
        if not COMPONENTS_AVAILABLE:
            return [], "", {
                "success": False,
                "error": "Phase 4 components not available for querying",
                "processed_videos": False
            }
        
        status = {
            "success": False,
            "processed_videos": False,
            "videos_found": len(self.video_files),
            "error": None
        }
        
        try:
            # Check if videos are already processed by attempting a quick search
            videos_processed = False
            try:
                test_documents, _ = self.search_videos("test", k=1)
                videos_processed = len(test_documents) > 0
            except Exception:
                # If search fails, videos likely not processed
                videos_processed = False
            
            # Process videos if needed
            if not videos_processed:
                print("PROCESSING: Videos not processed yet - running pipeline...")
                success = self.process_all_videos()
                if not success:
                    status["error"] = "Failed to process videos"
                    return [], "", status
                
                status["processed_videos"] = True
                print("SUCCESS: Video processing completed")
            
            # Execute the actual search
            documents, json_file = self.search_videos(query, k=k)
            
            status["success"] = True
            status["total_results"] = len(documents)
            
            return documents, json_file, status
            
        except Exception as e:
            status["error"] = str(e)
            print(f"ERROR: Intelligent search failed: {e}")
            return [], "", status
    
    def format_mcp_response(self, query: str, documents: List, json_file: str, status: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format search results for MCP client consumption.
        
        Converts raw documents into a structured response with metadata and summaries.
        """
        if not status["success"]:
            return {
                "success": False,
                "query": query,
                "error": status.get("error", "Unknown error"),
                "results": []
            }
        
        if not documents:
            return {
                "success": True,
                "query": query,
                "total_results": 0,
                "results": [],
                "message": "No matching content found"
            }
        
        # Format results as list of content chunks with metadata
        results = []
        for i, doc in enumerate(documents):
            # Get the actual content and metadata
            result = {
                "rank": i + 1,
                "content": doc.page_content,
                "video_id": doc.metadata.get('video_id', 'unknown'),
                "timing": doc.get_timing_info(),
                "start_time": doc.metadata.get('start', 0),
                "end_time": doc.metadata.get('end', 0),
                "modality": doc.metadata.get('modality', 'unknown'),
                "is_audio": doc.is_audio_segment(),
                "is_visual": doc.is_frame_segment(),
                "citation": f"[{doc.get_timing_info()}] {doc.metadata.get('modality')} from {doc.metadata.get('video_id')}"
            }
            
            # Add specific fields based on content type
            if doc.is_audio_segment():
                result["word_count"] = doc.metadata.get('word_count', 0)
                result["duration"] = doc.metadata.get('duration', 0)
            elif doc.is_frame_segment():
                result["frame_path"] = doc.metadata.get('path')
            
            results.append(result)
        
        # Create summary stats
        audio_segments = [r for r in results if r["is_audio"]]
        visual_segments = [r for r in results if r["is_visual"]]
        video_ids = list(set(r["video_id"] for r in results))
        
        response = {
            "success": True,
            "query": query,
            "total_results": len(results),
            "results": results,  # List of content chunks with metadata
            "summary": {
                "audio_segments": len(audio_segments),
                "visual_segments": len(visual_segments),
                "videos_included": len(video_ids),
                "video_ids": video_ids,
                "time_range": f"{min(r['start_time'] for r in results):.1f}s - {max(r['end_time'] for r in results):.1f}s" if results else "0s - 0s",
                "total_words": sum(r.get("word_count", 0) for r in audio_segments)
            },
            "detailed_results_file": json_file  # Full MCP-ready JSON file path
        }
        
        return response
    
    def search_and_format_for_mcp(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Complete search workflow for MCP: intelligent search + formatted response.
        
        This is the single method MCP server should call.
        """
        documents, json_file, status = self.intelligent_search(query, k=k)
        return self.format_mcp_response(query, documents, json_file, status)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not COMPONENTS_AVAILABLE:
            return {"error": "Components not available"}
        
        try:
            retriever = Retriever()
            stats = retriever.get_stats()
            return {
                "total_documents": stats.total_documents,
                "audio_documents": stats.audio_documents,
                "frame_documents": stats.frame_documents,
                "total_videos": stats.total_videos
            }
        except Exception as e:
            return {"error": str(e)}
    
    def process_all_videos(self) -> bool:
        """Process all videos through phases 1-3 and prepare for retrieval."""
        if not COMPONENTS_AVAILABLE:
            print("ERROR: Required components not available")
            return False
        
        # Run multi-video processing for all videos found
        start_time = time.time()
        
        # Process all videos through phases 1-3
        success = self.run_phases_for_all_videos()
        
        if not success:
            print("ERROR: Multi-video processing failed")
            return False
        
        total_time = time.time() - start_time
        print(f"\nCOMPLETE: Multi-video pipeline completed in {total_time:.1f}s")
        print("SUCCESS: All video content indexed and ready for MCP queries")
        
        return True
    
    def test_query(self, query: str, k: int = 5) -> bool:
        """Simple query test to verify retrieval works correctly."""
        if not COMPONENTS_AVAILABLE:
            print("ERROR: Required components not available for querying")
            return False
        
        print(f"\nSEARCH: Testing Query: '{query}'")
        print(f"STATS: Retrieving top {k} results...")
        
        try:
            # Execute search
            start_time = time.time()
            documents, json_file = self.search_videos(query, k=k)
            search_time = time.time() - start_time
            
            if not documents:
                print("WARNING: No results found")
                return False
            
            print(f"\nSUCCESS: Found {len(documents)} results in {search_time:.3f}s:")
            print(f"{'='*80}")
            
            for i, doc in enumerate(documents, 1):
                print(f"\n#{i} [{doc.get_timing_info()}] {doc.metadata.get('video_id', 'unknown')} ({doc.metadata.get('modality', 'unknown')})")
                
                if doc.is_audio_segment():
                    content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    print(f"   Text: {content_preview}")
                    print(f"   Words: {doc.metadata.get('word_count', 0)}, Duration: {doc.metadata.get('duration', 0):.1f}s")
                elif doc.is_frame_segment():
                    print(f"   Frame: {doc.page_content}")
                    print(f"   Path: {doc.metadata.get('path', 'N/A')}")
            
            print(f"\n{'='*80}")
            print(f"SAVED: Detailed results saved to: {json_file}")
            return True
            
        except Exception as e:
            print(f"ERROR: Query test failed: {e}")
            return False


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video RAG Pipeline MCP Driver")
    parser.add_argument("--video", default="videos", help="Video file or directory to process")
    parser.add_argument("--query", help="Test query to search for (optional)")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return (default: 5)")
    
    args = parser.parse_args()
    
    driver = VideoRAGMCPDriver(videos_dir=args.video)
    
    # Process videos if not already done
    success = driver.process_all_videos()
    if not success:
        return 1
    
    # Test query if provided
    if args.query:
        print(f"\n{'='*80}")
        print(f"SEARCH: QUERY TEST MODE")
        print(f"{'='*80}")
        query_success = driver.test_query(args.query, args.k)
        if not query_success:
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())