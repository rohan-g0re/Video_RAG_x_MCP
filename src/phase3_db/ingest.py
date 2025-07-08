"""
Batch Ingestion Interface for Video RAG Embeddings

Handles ingestion of embeddings from Phase 1 (audio) and Phase 2 (visual)
into the ChromaDB vector store with batch processing and error handling.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import time

from .client import VectorStoreClient
from .models import VideoSegment, BatchIngestRequest, BatchIngestResponse

logger = logging.getLogger(__name__)


class BatchIngestor:
    """
    Handles batch ingestion of video segment embeddings from Phase 1 and Phase 2 outputs.
    
    Supports both file-based and direct data ingestion with comprehensive error handling
    and validation.
    """
    
    def __init__(self, vector_client: Optional[VectorStoreClient] = None, batch_size: int = 20):
        """
        Initialize batch ingestor.
        
        Args:
            vector_client: Optional VectorStoreClient instance (creates default if None)
            batch_size: Size of batches for ChromaDB operations
        """
        self.vector_client = vector_client or VectorStoreClient()
        self.batch_size = batch_size
        self.stats = {
            "total_processed": 0,
            "total_errors": 0,
            "audio_segments": 0,
            "visual_segments": 0,
            "processing_time": 0.0
        }
        
        logger.info(f"BatchIngestor initialized with batch_size={batch_size}")
    
    def ingest_phase1_embeddings(self, embeddings_file: Union[str, Path]) -> BatchIngestResponse:
        """
        Ingest embeddings from Phase 1 (audio processing) output file.
        
        Args:
            embeddings_file: Path to Phase 1 embeddings JSON file
            
        Returns:
            BatchIngestResponse with operation results
        """
        embeddings_file = Path(embeddings_file)
        
        if not embeddings_file.exists():
            return BatchIngestResponse(
                success=False,
                segments_processed=0,
                segments_failed=0,
                error_messages=[f"Embeddings file not found: {embeddings_file}"],
                collection_name=self.vector_client.collection_name
            )
        
        logger.info(f"Ingesting Phase 1 embeddings from: {embeddings_file}")
        
        try:
            # Load embeddings data
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            # Convert to VideoSegment objects
            segments = []
            errors = []
            
            for i, embedding_item in enumerate(embeddings_data):
                try:
                    segment = VideoSegment.from_phase1_output(embedding_item)
                    segments.append(segment)
                except Exception as e:
                    error_msg = f"Failed to parse Phase 1 segment {i}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Batch ingest the segments
            return self._batch_ingest_segments(segments, errors, "phase1")
            
        except Exception as e:
            error_msg = f"Failed to load Phase 1 embeddings file: {e}"
            logger.error(error_msg)
            return BatchIngestResponse(
                success=False,
                segments_processed=0,
                segments_failed=0,
                error_messages=[error_msg],
                collection_name=self.vector_client.collection_name
            )
    
    def ingest_phase2_embeddings(self, frame_embeddings: Union[str, Path, List[Dict]]) -> BatchIngestResponse:
        """
        Ingest embeddings from Phase 2 (visual processing) output.
        
        Args:
            frame_embeddings: Path to embeddings file or list of FrameEmbedding dictionaries
            
        Returns:
            BatchIngestResponse with operation results
        """
        logger.info("Ingesting Phase 2 (visual) embeddings")
        
        # Handle both file path and direct data input
        if isinstance(frame_embeddings, (str, Path)):
            embeddings_file = Path(frame_embeddings)
            
            if not embeddings_file.exists():
                return BatchIngestResponse(
                    success=False,
                    segments_processed=0,
                    segments_failed=0,
                    error_messages=[f"Frame embeddings file not found: {embeddings_file}"],
                    collection_name=self.vector_client.collection_name
                )
            
            try:
                with open(embeddings_file, 'r') as f:
                    embeddings_data = json.load(f)
            except Exception as e:
                return BatchIngestResponse(
                    success=False,
                    segments_processed=0,
                    segments_failed=0,
                    error_messages=[f"Failed to load frame embeddings file: {e}"],
                    collection_name=self.vector_client.collection_name
                )
        else:
            embeddings_data = frame_embeddings
        
        # Convert to VideoSegment objects
        segments = []
        errors = []
        
        for i, frame_data in enumerate(embeddings_data):
            try:
                segment = VideoSegment.from_phase2_output(frame_data)
                segments.append(segment)
            except Exception as e:
                error_msg = f"Failed to parse Phase 2 segment {i}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        return self._batch_ingest_segments(segments, errors, "phase2")
    
    def ingest_combined_video(self, video_id: str, phase1_file: Union[str, Path], 
                             phase2_data: Union[str, Path, List[Dict]]) -> BatchIngestResponse:
        """
        Ingest both audio and visual embeddings for a complete video.
        
        Args:
            video_id: Unique video identifier
            phase1_file: Path to Phase 1 audio embeddings file
            phase2_data: Phase 2 visual embeddings (file path or data)
            
        Returns:
            BatchIngestResponse with combined operation results
        """
        logger.info(f"Ingesting combined embeddings for video: {video_id}")
        
        start_time = time.time()
        
        # Clear any existing data for this video first
        delete_result = self.vector_client.delete_by_video_id(video_id)
        if delete_result.get("segments_deleted", 0) > 0:
            logger.info(f"Removed {delete_result['segments_deleted']} existing segments for video: {video_id}")
        
        all_segments = []
        all_errors = []
        
        # Ingest Phase 1 (audio) embeddings
        phase1_response = self.ingest_phase1_embeddings(phase1_file)
        
        if phase1_response.success:
            logger.info(f"Phase 1: Successfully processed {phase1_response.segments_processed} audio segments")
        else:
            logger.error(f"Phase 1 ingestion failed: {phase1_response.error_messages}")
            all_errors.extend(phase1_response.error_messages)
        
        # Ingest Phase 2 (visual) embeddings  
        phase2_response = self.ingest_phase2_embeddings(phase2_data)
        
        if phase2_response.success:
            logger.info(f"Phase 2: Successfully processed {phase2_response.segments_processed} visual segments")
        else:
            logger.error(f"Phase 2 ingestion failed: {phase2_response.error_messages}")
            all_errors.extend(phase2_response.error_messages)
        
        # Update stats
        self.stats["processing_time"] = time.time() - start_time
        self.stats["audio_segments"] = phase1_response.segments_processed
        self.stats["visual_segments"] = phase2_response.segments_processed
        self.stats["total_processed"] = phase1_response.segments_processed + phase2_response.segments_processed
        self.stats["total_errors"] = phase1_response.segments_failed + phase2_response.segments_failed
        
        # Get final collection info
        collection_info = self.vector_client.get_collection_info()
        
        combined_success = phase1_response.success and phase2_response.success
        
        return BatchIngestResponse(
            success=combined_success,
            segments_processed=self.stats["total_processed"],
            segments_failed=self.stats["total_errors"],
            error_messages=all_errors,
            collection_name=self.vector_client.collection_name,
            total_segments_in_collection=collection_info["count"]
        )
    
    def _batch_ingest_segments(self, segments: List[VideoSegment], 
                              errors: List[str], source: str) -> BatchIngestResponse:
        """
        Internal method to batch ingest segments with error handling.
        
        Args:
            segments: List of VideoSegment objects to ingest
            errors: List of parsing errors from data conversion
            source: Source identifier (phase1, phase2, etc.)
            
        Returns:
            BatchIngestResponse with detailed results
        """
        if not segments:
            return BatchIngestResponse(
                success=len(errors) == 0,
                segments_processed=0,
                segments_failed=len(errors),
                error_messages=errors,
                collection_name=self.vector_client.collection_name
            )
        
        logger.info(f"Batch ingesting {len(segments)} segments from {source}")
        
        total_processed = 0
        total_failed = len(errors)  # Start with parsing errors
        all_errors = errors.copy()
        
        # Process segments in batches
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            logger.info(f"Processing batch {batch_num} ({len(batch)} segments)")
            
            try:
                result = self.vector_client.add_segments(batch)
                
                if result["success"]:
                    total_processed += result["segments_added"]
                    logger.info(f"Batch {batch_num}: Added {result['segments_added']} segments")
                else:
                    total_failed += len(batch)
                    error_msg = f"Batch {batch_num} failed: {result.get('message', 'Unknown error')}"
                    all_errors.append(error_msg)
                    logger.error(error_msg)
                    
            except Exception as e:
                total_failed += len(batch)
                error_msg = f"Batch {batch_num} exception: {e}"
                all_errors.append(error_msg)
                logger.error(error_msg)
        
        # Get final collection info
        collection_info = self.vector_client.get_collection_info()
        
        success = total_failed == 0
        
        logger.info(f"Batch ingestion complete: {total_processed} processed, {total_failed} failed")
        
        return BatchIngestResponse(
            success=success,
            segments_processed=total_processed,
            segments_failed=total_failed,
            error_messages=all_errors,
            collection_name=self.vector_client.collection_name,
            total_segments_in_collection=collection_info["count"]
        )
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            **self.stats,
            "collection_info": self.vector_client.get_collection_info()
        }
    
    def reset_stats(self) -> None:
        """Reset ingestion statistics."""
        self.stats = {
            "total_processed": 0,
            "total_errors": 0,
            "audio_segments": 0,
            "visual_segments": 0,
            "processing_time": 0.0
        }


def ingest_from_files(phase1_embeddings_file: str, phase2_embeddings_file: str = None, 
                     video_id: str = None) -> BatchIngestResponse:
    """
    Convenience function to ingest embeddings from Phase 1 and optionally Phase 2 files.
    
    Args:
        phase1_embeddings_file: Path to Phase 1 embeddings JSON file
        phase2_embeddings_file: Optional path to Phase 2 embeddings file
        video_id: Optional video ID for combined ingestion
        
    Returns:
        BatchIngestResponse with operation results
    """
    ingestor = BatchIngestor()
    
    if phase2_embeddings_file and video_id:
        # Combined ingestion
        return ingestor.ingest_combined_video(video_id, phase1_embeddings_file, phase2_embeddings_file)
    else:
        # Phase 1 only
        return ingestor.ingest_phase1_embeddings(phase1_embeddings_file) 