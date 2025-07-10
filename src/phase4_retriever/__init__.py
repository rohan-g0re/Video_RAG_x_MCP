"""
Phase 4: Retrieval Service Module

Provides query processing and similarity search capabilities for the Video RAG pipeline.
This module implements the retrieval interface as specified in the development plan,
serving as a standalone service that interfaces with Phase 3's vector storage.

Key Features:
- Query text embedding using CLIP text encoder
- Cosine similarity search against stored video segments  
- Rank-ordered results with metadata
- Independent CLI and API interfaces
- Document-based result format compatible with LLM generation

Deliverables:
- retriever.py: Main retriever class with search endpoint
- models.py: Pydantic models for requests/responses  
- cli.py: Command-line interface for standalone execution
- api.py: FastAPI wrapper for HTTP endpoints
"""

from .retriever import Retriever, search_videos
from .models import Document, RetrievalRequest, RetrievalResponse
from .embed_query import QueryEmbedder

__version__ = "1.0.0"
__all__ = [
    "Retriever",
    "Document", 
    "RetrievalRequest",
    "RetrievalResponse",
    "QueryEmbedder",
    "search_videos"
] 