"""
ChromaDB Vector Store Client

Provides a local ChromaDB client for video segment storage and retrieval
without requiring Docker containerization.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .models import VideoSegment, EmbeddingMetadata

logger = logging.getLogger(__name__)


class VectorStoreClient:
    """
    Local ChromaDB client for video RAG embeddings.
    
    Manages a single collection 'video_segments' containing both audio and visual embeddings
    with unified metadata schema.
    """
    
    def __init__(self, persist_directory: str = "data/chroma", collection_name: str = "video_segments"):
        """
        Initialize ChromaDB client with local persistence.
        
        Args:
            persist_directory: Local directory for ChromaDB storage
            collection_name: Name of the collection to store video segments
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing VectorStoreClient with persist_directory={persist_directory}")
        
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client with local persistence."""
        if self._client is not None:
            return
            
        logger.info("Initializing ChromaDB client")
        
        # Configure ChromaDB for local persistence without Docker
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.persist_directory),
            anonymized_telemetry=False  # Disable telemetry for privacy
        )
        
        try:
            self._client = chromadb.Client(settings)
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")
    
    def _get_collection(self) -> chromadb.Collection:
        """Get or create the video segments collection."""
        if self._collection is not None:
            return self._collection
            
        self._initialize_client()
        
        try:
            # Try to get existing collection first
            self._collection = self._client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except ValueError:
            # Collection doesn't exist, create it
            logger.info(f"Creating new collection: {self.collection_name}")
            
            # Create collection without embedding function (we provide embeddings directly)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"description": "Video RAG segments with unified audio/visual embeddings"}
            )
            logger.info(f"Created collection: {self.collection_name}")
            
        return self._collection
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        collection = self._get_collection()
        
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
    
    def add_segments(self, segments: List[VideoSegment]) -> Dict[str, Any]:
        """
        Add video segments to the collection.
        
        Args:
            segments: List of VideoSegment objects to add
            
        Returns:
            Dictionary with operation results
        """
        if not segments:
            return {"success": False, "message": "No segments provided"}
        
        collection = self._get_collection()
        
        # Prepare data for ChromaDB
        ids = [segment.id for segment in segments]
        embeddings = [segment.embedding for segment in segments]
        metadatas = [segment.metadata.to_chroma_metadata() for segment in segments]
        
        # Optional: Add documents (text content) for easier debugging
        documents = []
        for segment in segments:
            if segment.content:
                documents.append(segment.content)
            else:
                # Fallback content for frames or empty segments
                documents.append(f"{segment.metadata.modality} segment at {segment.metadata.start:.1f}s")
        
        try:
            logger.info(f"Adding {len(segments)} segments to collection")
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            new_count = collection.count()
            logger.info(f"Successfully added {len(segments)} segments. Total count: {new_count}")
            
            return {
                "success": True,
                "segments_added": len(segments),
                "total_segments": new_count,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to add segments: {e}")
            return {
                "success": False,
                "message": f"Failed to add segments: {e}",
                "segments_failed": len(segments)
            }
    
    def query_segments(self, query_embedding: List[float], k: int = 10, 
                      where_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query segments by embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            Dictionary with query results
        """
        collection = self._get_collection()
        
        try:
            logger.info(f"Querying collection for top-{k} similar segments")
            
            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=['metadatas', 'documents', 'distances', 'embeddings']
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    segment_data = {
                        'id': results['ids'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'document': results['documents'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'embedding': results['embeddings'][0][i] if results['embeddings'] else None
                    }
                    formatted_results.append(segment_data)
            
            return {
                "success": True,
                "results": formatted_results,
                "total_found": len(formatted_results),
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "success": False,
                "message": f"Query failed: {e}",
                "results": []
            }
    
    def delete_by_video_id(self, video_id: str) -> Dict[str, Any]:
        """
        Delete all segments for a specific video.
        
        Args:
            video_id: Video ID to delete
            
        Returns:
            Dictionary with deletion results
        """
        collection = self._get_collection()
        
        try:
            # Query to find segments for this video
            existing = collection.get(
                where={"video_id": video_id},
                include=['metadatas']
            )
            
            if not existing['ids']:
                return {
                    "success": True,
                    "message": f"No segments found for video_id: {video_id}",
                    "segments_deleted": 0
                }
            
            # Delete the segments
            collection.delete(
                where={"video_id": video_id}
            )
            
            deleted_count = len(existing['ids'])
            logger.info(f"Deleted {deleted_count} segments for video_id: {video_id}")
            
            return {
                "success": True,
                "segments_deleted": deleted_count,
                "video_id": video_id
            }
            
        except Exception as e:
            logger.error(f"Failed to delete segments for video_id {video_id}: {e}")
            return {
                "success": False,
                "message": f"Deletion failed: {e}"
            }
    
    def get_segments_by_video_id(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all segments for a specific video.
        
        Args:
            video_id: Video ID to retrieve
            
        Returns:
            List of segment data dictionaries
        """
        collection = self._get_collection()
        
        try:
            results = collection.get(
                where={"video_id": video_id},
                include=['metadatas', 'documents', 'embeddings']
            )
            
            segments = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    segment_data = {
                        'id': results['ids'][i],
                        'metadata': results['metadatas'][i],
                        'document': results['documents'][i],
                        'embedding': results['embeddings'][i] if results['embeddings'] else None
                    }
                    segments.append(segment_data)
            
            return segments
            
        except Exception as e:
            logger.error(f"Failed to get segments for video_id {video_id}: {e}")
            return []
    
    def list_videos(self) -> List[str]:
        """
        Get list of all video IDs in the collection.
        
        Returns:
            List of unique video IDs
        """
        collection = self._get_collection()
        
        try:
            # Get all metadata to extract video IDs
            results = collection.get(include=['metadatas'])
            
            video_ids = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if 'video_id' in metadata:
                        video_ids.add(metadata['video_id'])
            
            return sorted(list(video_ids))
            
        except Exception as e:
            logger.error(f"Failed to list videos: {e}")
            return []
    
    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear all data from the collection.
        
        Returns:
            Dictionary with operation results
        """
        try:
            if self._collection:
                self._client.delete_collection(name=self.collection_name)
                self._collection = None
                logger.info(f"Cleared collection: {self.collection_name}")
            
            return {
                "success": True,
                "message": f"Collection {self.collection_name} cleared"
            }
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return {
                "success": False,
                "message": f"Failed to clear collection: {e}"
            }
    
    def close(self) -> None:
        """Close the client connection."""
        if self._client:
            # ChromaDB doesn't require explicit closing, but we can reset our references
            self._collection = None
            self._client = None
            logger.info("VectorStoreClient closed") 