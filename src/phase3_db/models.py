"""
Data Models for Video RAG Vector Store

Defines Pydantic models for type-safe database operations following
the schema specification from the development plan.
"""

import uuid
from typing import List, Optional, Dict, Any, Literal, Union, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np


class EmbeddingMetadata(BaseModel):
    """Metadata associated with each video segment embedding."""
    
    video_id: str = Field(..., description="Unique identifier for the source video")
    modality: Literal["audio", "frame"] = Field(..., description="Type of content: audio transcript or visual frame")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., gt=0, description="End time in seconds") 
    path: Optional[str] = Field(None, description="File path for frame images, null for audio")
    
    # Additional metadata fields from existing pipeline
    segment_index: Optional[int] = Field(None, description="Index of segment within video")
    content_type: Optional[str] = Field(None, description="Content classification (speech, silence, etc.)")
    word_count: Optional[int] = Field(None, description="Number of words in audio segments")
    duration: Optional[float] = Field(None, description="Segment duration in seconds")
    overlap_added: Optional[float] = Field(None, description="Overlap duration added for context")
    
    @validator('end')
    def end_must_be_after_start(cls, v, values):
        """Ensure end time is after start time."""
        if 'start' in values and v <= values['start']:
            raise ValueError('end time must be after start time')
        return v
    
    @validator('path')
    def path_required_for_frames(cls, v, values):
        """Ensure frame modality has a path."""
        if values.get('modality') == 'frame' and (v is None or v == ''):
            raise ValueError('frame modality must have a path')
        return v
    
    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata format (only JSON-serializable types)."""
        metadata = {
            "video_id": self.video_id,
            "modality": self.modality,
            "start": float(self.start),
            "end": float(self.end),
        }
        
        # Add optional fields if present
        if self.path is not None:
            metadata["path"] = self.path
        if self.segment_index is not None:
            metadata["segment_index"] = int(self.segment_index)
        if self.content_type is not None:
            metadata["content_type"] = self.content_type
        if self.word_count is not None:
            metadata["word_count"] = int(self.word_count)
        if self.duration is not None:
            metadata["duration"] = float(self.duration)
        if self.overlap_added is not None:
            metadata["overlap_added"] = float(self.overlap_added)
            
        return metadata


class VideoSegment(BaseModel):
    """Complete video segment with embedding and metadata."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique segment identifier")
    embedding: List[float] = Field(..., description="Dense embedding vector")
    metadata: EmbeddingMetadata = Field(..., description="Segment metadata")
    
    # Optional content for debugging/inspection
    content: Optional[str] = Field(None, description="Text content for audio segments or description for frames")
    
    @validator('embedding')
    def embedding_must_be_valid(cls, v):
        """Ensure embedding is a valid non-empty vector."""
        if not v:
            raise ValueError('embedding cannot be empty')
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError('embedding must contain only numeric values')
        return v
    
    @classmethod
    def from_phase1_output(cls, embedding_data: Dict[str, Any]) -> 'VideoSegment':
        """Create VideoSegment from Phase 1 (audio) output format."""
        metadata_dict = embedding_data.get('metadata', {})
        
        # Extract metadata fields
        metadata = EmbeddingMetadata(
            video_id=metadata_dict.get('video_id', 'unknown'),
            modality="audio",
            start=metadata_dict.get('start', 0.0),
            end=metadata_dict.get('end', 0.0),
            segment_index=metadata_dict.get('segment_index'),
            content_type=metadata_dict.get('content_type'),
            word_count=metadata_dict.get('word_count'),
            duration=metadata_dict.get('duration'),
            overlap_added=metadata_dict.get('overlap_added', 0.0)
        )
        
        return cls(
            embedding=embedding_data['embedding'],
            metadata=metadata,
            content=embedding_data.get('caption', '')
        )
    
    @classmethod 
    def from_phase2_output(cls, frame_embedding_data: Dict[str, Any]) -> 'VideoSegment':
        """Create VideoSegment from Phase 2 (visual) output format."""
        
        # Handle both direct FrameEmbedding objects and dictionaries
        if hasattr(frame_embedding_data, 'to_dict'):
            data = frame_embedding_data.to_dict()
        else:
            data = frame_embedding_data
            
        metadata = EmbeddingMetadata(
            video_id=data['video_id'],
            modality="frame", 
            start=data['start'],
            end=data['end'],
            path=data['frame_path'],
            duration=data['end'] - data['start']
        )
        
        return cls(
            embedding=data['embedding'] if isinstance(data['embedding'], list) else data['embedding'].tolist(),
            metadata=metadata,
            content=f"Frame at {data['timestamp']}s"
        )


class QueryResult(BaseModel):
    """Result from similarity search query."""
    
    segment: VideoSegment = Field(..., description="Retrieved video segment")
    similarity_score: float = Field(..., ge=0, le=1, description="Cosine similarity score")
    rank: int = Field(..., ge=1, description="Rank in search results (1-based)")
    
    def get_timing_info(self) -> str:
        """Get formatted timing information."""
        return f"{self.segment.metadata.start:.1f}s-{self.segment.metadata.end:.1f}s"
    
    def get_summary(self) -> str:
        """Get summary string for the result."""
        timing = self.get_timing_info()
        content_preview = (self.segment.content[:50] + "...") if self.segment.content and len(self.segment.content) > 50 else (self.segment.content or "")
        return f"[{timing}] {self.segment.metadata.modality}: {content_preview} (score: {self.similarity_score:.3f})"


class BatchIngestRequest(BaseModel):
    """Request for batch ingestion of video segments."""
    
    segments: List[VideoSegment] = Field(..., description="List of segments to ingest")
    collection_name: str = Field("video_segments", description="Target collection name")
    
    @validator('segments')
    def segments_must_not_be_empty(cls, v):
        """Ensure at least one segment is provided."""
        if not v:
            raise ValueError('segments list cannot be empty')
        return v


class BatchIngestResponse(BaseModel):
    """Response from batch ingestion operation."""
    
    success: bool = Field(..., description="Whether ingestion succeeded")
    segments_processed: int = Field(..., description="Number of segments successfully processed")
    segments_failed: int = Field(0, description="Number of segments that failed")
    error_messages: List[str] = Field(default_factory=list, description="Error messages if any failures occurred")
    collection_name: str = Field(..., description="Target collection name")
    total_segments_in_collection: Optional[int] = Field(None, description="Total segments now in collection")


class QueryRequest(BaseModel):
    """Request for similarity search query."""
    
    query_text: str = Field(..., description="Natural language query")
    k: int = Field(10, ge=1, le=50, description="Number of results to return")
    collection_name: str = Field("video_segments", description="Collection to search")
    
    # Optional filters
    video_id_filter: Optional[str] = Field(None, description="Filter by specific video ID")
    modality_filter: Optional[Literal["audio", "frame"]] = Field(None, description="Filter by modality")
    time_range_filter: Optional[Tuple[float, float]] = Field(None, description="Filter by time range (start, end)")


class QueryResponse(BaseModel):
    """Response from similarity search query."""
    
    query: str = Field(..., description="Original query text")
    results: List[QueryResult] = Field(..., description="Ranked search results")
    total_found: int = Field(..., description="Total number of results found")
    search_time_seconds: float = Field(..., description="Time taken for search")
    collection_name: str = Field(..., description="Collection searched") 