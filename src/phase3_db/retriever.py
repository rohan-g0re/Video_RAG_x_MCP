"""
Vector Retrieval Interface for Video RAG

Handles query embedding and similarity search with metadata filtering
and result ranking for the Video RAG pipeline.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
import open_clip
import numpy as np

from .client import VectorStoreClient
from .models import QueryRequest, QueryResponse, QueryResult, VideoSegment, EmbeddingMetadata

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Handles query processing and similarity search for video segments.
    
    Uses the same CLIP model as Phase 1 and Phase 2 to ensure embedding compatibility.
    """
    
    def __init__(self, vector_client: Optional[VectorStoreClient] = None, 
                 clip_model: str = "ViT-B-32", device: str = None):
        """
        Initialize the vector retriever.
        
        Args:
            vector_client: Optional VectorStoreClient instance
            clip_model: CLIP model architecture to use for query embedding
            device: Device for model inference (auto-detect if None)
        """
        self.vector_client = vector_client or VectorStoreClient()
        self.clip_model = clip_model
        
        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize CLIP model for query embedding
        self._load_clip_model()
        
        logger.info(f"VectorRetriever initialized with {clip_model} on {self.device}")
    
    def _load_clip_model(self) -> None:
        """Load CLIP model for query text embedding."""
        logger.info(f"Loading CLIP model: {self.clip_model}")
        
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.clip_model, pretrained="openai"
            )
            self.tokenizer = open_clip.get_tokenizer(self.clip_model)
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            with torch.no_grad():
                dummy_text = self.tokenizer(["test"])
                dummy_embedding = self.model.encode_text(dummy_text)
                self.embedding_dim = dummy_embedding.shape[1]
            
            logger.info(f"CLIP model loaded: {self.embedding_dim}D embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"CLIP model initialization failed: {e}")
    
    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Embed a query text using CLIP text encoder.
        
        Args:
            query_text: Natural language query
            
        Returns:
            Normalized query embedding
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        try:
            # Tokenize and embed
            tokens = self.tokenizer([query_text.strip()])
            
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                # Normalize embedding
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy()[0]  # Return first (and only) embedding
            
        except Exception as e:
            logger.error(f"Failed to embed query '{query_text}': {e}")
            raise RuntimeError(f"Query embedding failed: {e}")
    
    def search(self, query_request: QueryRequest) -> QueryResponse:
        """
        Perform similarity search for a query.
        
        Args:
            query_request: Query request with parameters
            
        Returns:
            QueryResponse with ranked results
        """
        start_time = time.time()
        
        logger.info(f"Processing query: '{query_request.query_text}'")
        
        try:
            # Embed the query
            query_embedding = self.embed_query(query_request.query_text)
            
            # Build metadata filter
            where_filter = self._build_where_filter(query_request)
            
            # Perform similarity search
            search_results = self.vector_client.query_segments(
                query_embedding=query_embedding.tolist(),
                k=query_request.k,
                where_filter=where_filter
            )
            
            if not search_results["success"]:
                raise RuntimeError(f"Vector search failed: {search_results.get('message', 'Unknown error')}")
            
            # Convert results to QueryResult objects
            query_results = []
            for i, result in enumerate(search_results["results"]):
                # Reconstruct VideoSegment from result
                metadata = EmbeddingMetadata(**result["metadata"])
                segment = VideoSegment(
                    id=result["id"],
                    embedding=result["embedding"] if result["embedding"] else [],
                    metadata=metadata,
                    content=result["document"]
                )
                
                query_result = QueryResult(
                    segment=segment,
                    similarity_score=result["similarity_score"],
                    rank=i + 1
                )
                query_results.append(query_result)
            
            search_time = time.time() - start_time
            
            logger.info(f"Query completed in {search_time:.3f}s, found {len(query_results)} results")
            
            return QueryResponse(
                query=query_request.query_text,
                results=query_results,
                total_found=len(query_results),
                search_time_seconds=search_time,
                collection_name=query_request.collection_name
            )
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Query failed after {search_time:.3f}s: {e}")
            
            return QueryResponse(
                query=query_request.query_text,
                results=[],
                total_found=0,
                search_time_seconds=search_time,
                collection_name=query_request.collection_name
            )
    
    def _build_where_filter(self, query_request: QueryRequest) -> Optional[Dict[str, Any]]:
        """
        Build ChromaDB where filter from query request parameters.
        
        Args:
            query_request: Query request with filter parameters
            
        Returns:
            ChromaDB where filter dictionary or None
        """
        where_filter = {}
        
        # Video ID filter
        if query_request.video_id_filter:
            where_filter["video_id"] = query_request.video_id_filter
        
        # Modality filter
        if query_request.modality_filter:
            where_filter["modality"] = query_request.modality_filter
        
        # Time range filter (requires special handling)
        if query_request.time_range_filter:
            start_time, end_time = query_request.time_range_filter
            # Find segments that overlap with the time range
            where_filter["$and"] = [
                {"start": {"$lte": end_time}},    # Segment starts before range ends
                {"end": {"$gte": start_time}}     # Segment ends after range starts
            ]
        
        return where_filter if where_filter else None
    
    def search_by_text(self, query_text: str, k: int = 10, 
                      video_id: Optional[str] = None,
                      modality: Optional[str] = None) -> QueryResponse:
        """
        Convenience method for simple text-based search.
        
        Args:
            query_text: Natural language query
            k: Number of results to return
            video_id: Optional video ID filter
            modality: Optional modality filter ("audio" or "frame")
            
        Returns:
            QueryResponse with search results
        """
        query_request = QueryRequest(
            query_text=query_text,
            k=k,
            video_id_filter=video_id,
            modality_filter=modality
        )
        
        return self.search(query_request)
    
    def search_by_time_range(self, query_text: str, start_time: float, end_time: float,
                            k: int = 10, video_id: Optional[str] = None) -> QueryResponse:
        """
        Search within a specific time range of a video.
        
        Args:
            query_text: Natural language query
            start_time: Start time in seconds
            end_time: End time in seconds  
            k: Number of results to return
            video_id: Optional video ID filter
            
        Returns:
            QueryResponse with search results
        """
        query_request = QueryRequest(
            query_text=query_text,
            k=k,
            video_id_filter=video_id,
            time_range_filter=(start_time, end_time)
        )
        
        return self.search(query_request)
    
    def get_segments_for_video(self, video_id: str, modality: Optional[str] = None) -> List[VideoSegment]:
        """
        Retrieve all segments for a specific video, optionally filtered by modality.
        
        Args:
            video_id: Video identifier
            modality: Optional modality filter ("audio" or "frame")
            
        Returns:
            List of VideoSegment objects
        """
        raw_segments = self.vector_client.get_segments_by_video_id(video_id)
        
        segments = []
        for seg_data in raw_segments:
            # Filter by modality if specified
            if modality and seg_data["metadata"].get("modality") != modality:
                continue
                
            metadata = EmbeddingMetadata(**seg_data["metadata"])
            segment = VideoSegment(
                id=seg_data["id"],
                embedding=seg_data["embedding"] if seg_data["embedding"] else [],
                metadata=metadata,
                content=seg_data["document"]
            )
            segments.append(segment)
        
        # Sort by start time
        segments.sort(key=lambda x: x.metadata.start)
        
        return segments
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection.
        
        Returns:
            Dictionary with collection statistics
        """
        collection_info = self.vector_client.get_collection_info()
        videos = self.vector_client.list_videos()
        
        # Get modality breakdown
        all_segments = []
        for video_id in videos:
            segments = self.vector_client.get_segments_by_video_id(video_id)
            all_segments.extend(segments)
        
        audio_count = sum(1 for seg in all_segments if seg["metadata"].get("modality") == "audio")
        frame_count = sum(1 for seg in all_segments if seg["metadata"].get("modality") == "frame")
        
        return {
            "collection_name": collection_info["name"],
            "total_segments": collection_info["count"],
            "total_videos": len(videos),
            "audio_segments": audio_count,
            "frame_segments": frame_count,
            "video_ids": videos,
            "embedding_dimension": self.embedding_dim
        }


def search_videos(query_text: str, k: int = 10, video_id: Optional[str] = None) -> QueryResponse:
    """
    Convenience function for simple video search.
    
    Args:
        query_text: Natural language query
        k: Number of results to return
        video_id: Optional video ID filter
        
    Returns:
        QueryResponse with search results
    """
    retriever = VectorRetriever()
    return retriever.search_by_text(query_text, k=k, video_id=video_id) 