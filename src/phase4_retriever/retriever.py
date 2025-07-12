"""
Phase 4: Retrieval Service Implementation

Main retriever class implementing the interface specified in the development plan:
`Retriever.search(query: str, k:int=10) -> List[Document]`

This module provides the core retrieval functionality that interfaces with 
Phase 3's vector store to perform similarity search and return Document objects 
compatible with LLM generation.

Key Features:
- Single cosine similarity search in ChromaDB
- Rank-ordered results with Document interface
- Text content for audio, "<IMAGE_FRAME>" for frames
- Comprehensive filtering and metadata handling
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Optional, Union

# Add Phase 3 to path for vector store access
sys.path.insert(0, str(Path(__file__).parent.parent / "phase3_db"))

from .models import Document, RetrievalRequest, RetrievalResponse, RetrievalStats
from .embed_query import QueryEmbedder

# Import Phase 3 components for vector store access
try:
    from client import VectorStoreClient
    from models import QueryRequest, VideoSegment, EmbeddingMetadata
    PHASE3_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        from src.phase3_db.client import VectorStoreClient
        from src.phase3_db.models import QueryRequest, VideoSegment, EmbeddingMetadata
        PHASE3_AVAILABLE = True
    except ImportError:
        print("⚠️  Phase 3 vector store not available")
        VectorStoreClient = None
        QueryRequest = None
        VideoSegment = None
        EmbeddingMetadata = None
        PHASE3_AVAILABLE = False

logger = logging.getLogger(__name__)


class Retriever:
    """
    Phase 4 Retrieval Service
    
    Implements the retrieval interface specified in the development plan:
    `Retriever.search(query: str, k:int=10) -> List[Document]`
    
    Executes single cosine similarity search in ChromaDB and returns rank-ordered
    Document objects for seamless integration with LLM generation.
    """
    
    def __init__(self, vector_client: Optional[VectorStoreClient] = None,
                 persist_directory: str = "data/chroma", 
                 clip_model: str = "ViT-B-32"):
        """
        Initialize the Phase 4 retriever.
        
        Args:
            vector_client: Optional VectorStoreClient instance
            persist_directory: ChromaDB persistence directory
            clip_model: CLIP model for query embedding
        """
        self.persist_directory = persist_directory
        self.clip_model = clip_model
        
        # Initialize vector store client
        if vector_client is not None:
            self.vector_client = vector_client
        elif PHASE3_AVAILABLE:
            self.vector_client = VectorStoreClient(persist_directory=persist_directory)
        else:
            self.vector_client = None
            logger.warning("Phase 3 vector store not available - retrieval will be limited")
        
        # Initialize query embedder
        self.query_embedder = QueryEmbedder(model_name=clip_model)
        
        logger.info(f"Phase 4 Retriever initialized with {clip_model}")
    
    def search(self, query: str, k: int = 10) -> List[Document]:
        """
        Main search interface as specified in the development plan.
        
        Executes single cosine similarity search in ChromaDB and returns 
        rank-ordered list of Document objects.
        
        Args:
            query: Natural language query string
            k: Number of documents to return (default 10)
            
        Returns:
            List of Document objects with text content for audio segments
            and "<IMAGE_FRAME>" placeholder for visual segments
            
        Raises:
            RuntimeError: If vector store is not available
            ValueError: If query is empty
        """
        if not PHASE3_AVAILABLE or self.vector_client is None:
            raise RuntimeError("Phase 3 vector store not available for search")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Searching for: '{query}' (k={k})")
        
        try:
            # Embed the query using Phase 4 embedder
            query_embedding = self.query_embedder.embed_query(query.strip())
            
            # Perform similarity search via Phase 3 client
            search_results = self.vector_client.query_segments(
                query_embedding=query_embedding.tolist(),
                k=k,
                where_filter=None  # No filters for basic search
            )
            
            if not search_results["success"]:
                raise RuntimeError(f"Vector search failed: {search_results.get('message', 'Unknown error')}")
            
            # Convert Phase 3 results to Phase 4 Document objects
            documents = []
            for result in search_results["results"]:
                doc = self._convert_to_document(result)
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents for query: '{query}'")
            return documents
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            raise RuntimeError(f"Search operation failed: {e}")
    
    def search_with_filters(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Enhanced search with filtering and detailed response.
        
        Args:
            request: Retrieval request with query and filters
            
        Returns:
            Detailed retrieval response with statistics
        """
        start_time = time.time()
        
        if not PHASE3_AVAILABLE or self.vector_client is None:
            raise RuntimeError("Phase 3 vector store not available for search")
        
        logger.info(f"Processing retrieval request: '{request.query}'")
        
        try:
            # Embed the query
            query_embedding = self.query_embedder.embed_query(request.query)
            
            # Build Phase 3 query request with filters
            phase3_request = QueryRequest(
                query_text=request.query,
                k=request.k,
                video_id_filter=request.video_id,
                modality_filter=request.modality,
                time_range_filter=request.time_range
            )
            
            # Build where filter for Phase 3
            where_filter = self._build_where_filter(phase3_request)
            
            # Perform similarity search
            search_results = self.vector_client.query_segments(
                query_embedding=query_embedding.tolist(),
                k=request.k,
                where_filter=where_filter
            )
            
            if not search_results["success"]:
                raise RuntimeError(f"Vector search failed: {search_results.get('message', 'Unknown error')}")
            
            # Convert to Document objects
            documents = []
            for result in search_results["results"]:
                doc = self._convert_to_document(result)
                documents.append(doc)
            
            search_time = time.time() - start_time
            
            # Build response
            response = RetrievalResponse(
                query=request.query,
                documents=documents,
                total_found=len(documents),
                search_time_seconds=search_time
            )
            
            # Calculate statistics
            response.audio_documents = sum(1 for doc in documents if doc.is_audio_segment())
            response.frame_documents = sum(1 for doc in documents if doc.is_frame_segment())
            
            logger.info(f"Retrieved {len(documents)} documents in {search_time:.3f}s")
            return response
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Search with filters failed after {search_time:.3f}s: {e}")
            raise RuntimeError(f"Filtered search operation failed: {e}")
    
    def _convert_to_document(self, phase3_result: dict) -> Document:
        """
        Convert Phase 3 search result to Phase 4 Document.
        
        Args:
            phase3_result: Result from Phase 3 vector search
            
        Returns:
            Document object with appropriate content format
        """
        metadata = phase3_result["metadata"]
        modality = metadata.get("modality", "unknown")
        
        if modality == "audio":
            # Use actual transcript text for audio segments
            # First try caption from metadata (our fix), then fall back to document field
            page_content = (metadata.get("caption") or 
                          phase3_result.get("document", ""))
            
            # If still empty, try to get content from other sources
            if not page_content or not page_content.strip():
                page_content = f"Audio segment at {metadata.get('start', 0):.1f}s"
                
        elif modality == "frame":
            # Use placeholder for visual segments as specified
            page_content = "<IMAGE_FRAME>"
        else:
            # Fallback for unknown modality
            page_content = phase3_result.get("document", "")
        
        # Build document metadata
        doc_metadata = {
            "video_id": metadata.get("video_id", "unknown"),
            "modality": modality,
            "start": metadata.get("start", 0.0),
            "end": metadata.get("end", 0.0)
        }
        
        # Add optional metadata fields
        if "path" in metadata:
            doc_metadata["path"] = metadata["path"]
        if "segment_index" in metadata:
            doc_metadata["segment_index"] = metadata["segment_index"] 
        if "word_count" in metadata:
            doc_metadata["word_count"] = metadata["word_count"]
        if "duration" in metadata:
            doc_metadata["duration"] = metadata["duration"]
        
        return Document(
            page_content=page_content,
            metadata=doc_metadata
        )
    
    def _build_where_filter(self, query_request: QueryRequest) -> Optional[dict]:
        """
        Build ChromaDB where filter from query request.
        
        Args:
            query_request: Phase 3 query request with filters
            
        Returns:
            ChromaDB where filter or None
        """
        filters = []
        
        # Video ID filter
        if query_request.video_id_filter:
            filters.append({"video_id": query_request.video_id_filter})
        
        # Modality filter
        if query_request.modality_filter:
            filters.append({"modality": query_request.modality_filter})
        
        # Time range filter
        if query_request.time_range_filter:
            start_time, end_time = query_request.time_range_filter
            filters.append({"start": {"$lte": end_time}})
            filters.append({"end": {"$gte": start_time}})
        
        # Combine filters
        if len(filters) == 0:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            return {"$and": filters}
    
    def get_stats(self) -> RetrievalStats:
        """
        Get retrieval system statistics.
        
        Returns:
            Statistics about the vector collection
        """
        if not PHASE3_AVAILABLE or self.vector_client is None:
            raise RuntimeError("Phase 3 vector store not available for statistics")
        
        try:
            # Get collection information
            collection_info = self.vector_client.get_collection_info()
            videos = self.vector_client.list_videos()
            
            # Get modality breakdown
            all_segments = []
            for video_id in videos:
                segments = self.vector_client.get_segments_by_video_id(video_id)
                all_segments.extend(segments)
            
            audio_count = sum(1 for seg in all_segments if seg["metadata"].get("modality") == "audio")
            frame_count = sum(1 for seg in all_segments if seg["metadata"].get("modality") == "frame")
            
            return RetrievalStats(
                total_documents=collection_info["count"],
                total_videos=len(videos),
                audio_documents=audio_count,
                frame_documents=frame_count,
                video_ids=videos,
                embedding_dimension=self.query_embedder.embedding_dim
            )
            
        except Exception as e:
            logger.error(f"Failed to get retrieval statistics: {e}")
            raise RuntimeError(f"Statistics retrieval failed: {e}")
    
    def search_by_video(self, query: str, video_id: str, k: int = 10) -> List[Document]:
        """
        Search within a specific video.
        
        Args:
            query: Natural language query
            video_id: Video identifier to search within
            k: Number of documents to return
            
        Returns:
            List of Document objects from the specified video
        """
        request = RetrievalRequest(
            query=query,
            k=k,
            video_id=video_id
        )
        
        response = self.search_with_filters(request)
        return response.documents
    
    def search_by_modality(self, query: str, modality: str, k: int = 10) -> List[Document]:
        """
        Search within a specific content modality.
        
        Args:
            query: Natural language query
            modality: Content modality ("audio" or "frame")
            k: Number of documents to return
            
        Returns:
            List of Document objects of the specified modality
        """
        request = RetrievalRequest(
            query=query,
            k=k,
            modality=modality
        )
        
        response = self.search_with_filters(request)
        return response.documents


# Convenience function for simple usage
def search_videos(query: str, k: int = 10, persist_directory: str = "data/chroma") -> List[Document]:
    """
    Convenience function for simple video search.
    
    Args:
        query: Natural language query
        k: Number of documents to return
        persist_directory: ChromaDB persistence directory
        
    Returns:
        List of Document objects
    """
    retriever = Retriever(persist_directory=persist_directory)
    return retriever.search(query, k=k) 