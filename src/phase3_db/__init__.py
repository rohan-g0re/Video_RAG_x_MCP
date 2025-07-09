"""
Phase 3: Vector Store Service (ChromaDB)

This package provides ChromaDB integration for the Video RAG pipeline.
"""

try:
    # Try importing ChromaDB-dependent components
    from .models import VideoSegment, EmbeddingMetadata
    from .client import VectorStoreClient
    from .ingest import BatchIngestor
    from .retriever import VectorRetriever
    
    __all__ = [
        'VideoSegment',
        'EmbeddingMetadata', 
        'VectorStoreClient',
        'BatchIngestor',
        'VectorRetriever'
    ]
    
except ImportError as e:
    # ChromaDB not available, only export models
    try:
        from .models import VideoSegment, EmbeddingMetadata
        __all__ = ['VideoSegment', 'EmbeddingMetadata']
    except ImportError:
        __all__ = []

__version__ = "1.0.0" 