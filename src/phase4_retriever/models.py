"""
Phase 4 Retrieval Service Models

Defines the Document-based data models for the retrieval service as specified
in the development plan. These models provide the interface between the 
retrieval service and the LLM generation phase.

Key differences from Phase 3 models:
- Document-centric interface (not VideoSegment)
- Text content for audio, "<IMAGE_FRAME>" placeholder for frames
- Simplified metadata focused on timestamp and video information
- Compatible with LangChain Document format for LLM generation
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
import uuid


class Document(BaseModel):
    """
    Document representation for retrieval results.
    
    Compatible with LangChain Document format for seamless LLM integration.
    As specified in the development plan: text content for audio segments,
    "<IMAGE_FRAME>" placeholder for visual segments.
    """
    
    page_content: str = Field(..., description="Text content for audio or '<IMAGE_FRAME>' for visual")
    metadata: Dict[str, Any] = Field(..., description="Document metadata including timing and source")
    
    @classmethod
    def from_audio_segment(cls, segment_content: str, video_id: str, 
                          start: float, end: float, **extra_metadata) -> 'Document':
        """Create Document from audio segment."""
        metadata = {
            "video_id": video_id,
            "modality": "audio", 
            "start": start,
            "end": end,
            **extra_metadata
        }
        
        return cls(
            page_content=segment_content,
            metadata=metadata
        )
    
    @classmethod 
    def from_frame_segment(cls, video_id: str, start: float, end: float,
                          frame_path: Optional[str] = None, **extra_metadata) -> 'Document':
        """Create Document from frame segment."""
        metadata = {
            "video_id": video_id,
            "modality": "frame",
            "start": start, 
            "end": end,
            **extra_metadata
        }
        
        if frame_path:
            metadata["path"] = frame_path
            
        return cls(
            page_content="<IMAGE_FRAME>",
            metadata=metadata
        )
    
    def get_timing_info(self) -> str:
        """Get formatted timing information."""
        start = self.metadata.get("start", 0)
        end = self.metadata.get("end", 0)
        return f"{start:.1f}s-{end:.1f}s"
    
    def is_audio_segment(self) -> bool:
        """Check if this is an audio segment."""
        return self.metadata.get("modality") == "audio"
    
    def is_frame_segment(self) -> bool:
        """Check if this is a frame segment."""
        return self.metadata.get("modality") == "frame"


class RetrievalRequest(BaseModel):
    """Request for document retrieval."""
    
    query: str = Field(..., description="Natural language query text")
    k: int = Field(10, ge=1, le=50, description="Number of documents to retrieve")
    
    # Optional filters
    video_id: Optional[str] = Field(None, description="Filter by specific video ID")
    modality: Optional[Literal["audio", "frame"]] = Field(None, description="Filter by content type")
    time_range: Optional[tuple[float, float]] = Field(None, description="Filter by time range (start, end)")
    
    @validator('query')
    def query_must_not_be_empty(cls, v):
        """Ensure query is not empty."""
        if not v or not v.strip():
            raise ValueError('query cannot be empty')
        return v.strip()


class RetrievalResponse(BaseModel):
    """Response from document retrieval."""
    
    query: str = Field(..., description="Original query text")
    documents: List[Document] = Field(..., description="Retrieved documents in rank order")
    total_found: int = Field(..., description="Total number of documents found")
    search_time_seconds: float = Field(..., description="Time taken for search")
    
    # Optional statistics
    audio_documents: int = Field(0, description="Number of audio documents returned")
    frame_documents: int = Field(0, description="Number of frame documents returned")
    
    def __post_init__(self):
        """Calculate document type statistics."""
        self.audio_documents = sum(1 for doc in self.documents if doc.is_audio_segment())
        self.frame_documents = sum(1 for doc in self.documents if doc.is_frame_segment())
    
    def get_summary(self) -> str:
        """Get a summary of the retrieval results."""
        return (f"Retrieved {self.total_found} documents "
                f"({self.audio_documents} audio, {self.frame_documents} frame) "
                f"in {self.search_time_seconds:.3f}s")


class QueryEmbeddingRequest(BaseModel):
    """Request for query text embedding."""
    
    query_text: str = Field(..., description="Text to embed")
    
    @validator('query_text')
    def text_must_not_be_empty(cls, v):
        """Ensure text is not empty."""
        if not v or not v.strip():
            raise ValueError('query_text cannot be empty')
        return v.strip()


class QueryEmbeddingResponse(BaseModel):
    """Response from query embedding."""
    
    embedding: List[float] = Field(..., description="Query embedding vector")
    embedding_dim: int = Field(..., description="Embedding dimension")
    processing_time_seconds: float = Field(..., description="Time taken for embedding")


class RetrievalStats(BaseModel):
    """Statistics about the retrieval system."""
    
    total_documents: int = Field(..., description="Total documents in collection")
    total_videos: int = Field(..., description="Total videos indexed")
    audio_documents: int = Field(..., description="Total audio documents")
    frame_documents: int = Field(..., description="Total frame documents") 
    video_ids: List[str] = Field(..., description="List of indexed video IDs")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")


# Response models for API endpoints
class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field("healthy", description="Service status")
    version: str = Field("1.0.0", description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    models_loaded: bool = Field(..., description="Whether CLIP models are loaded")


class ErrorResponse(BaseModel):
    """Error response format."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request identifier") 