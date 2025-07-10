"""
Phase 4: Query Embedding Service

Handles query text embedding using CLIP text encoder as specified in the 
development plan. This module provides query embedding functionality that 
is compatible with the embeddings stored by Phase 1 and Phase 2.

Key Features:
- CLIP ViT-B/32 text encoder (same as Phase 1)
- Normalized embeddings for cosine similarity
- Batch processing capability
- Performance monitoring and logging
"""

import time
import logging
from typing import List, Union
import torch
import open_clip
import numpy as np

from .models import QueryEmbeddingRequest, QueryEmbeddingResponse

logger = logging.getLogger(__name__)


class QueryEmbedder:
    """
    Handles query text embedding using CLIP text encoder.
    
    Uses the same CLIP model as Phase 1 to ensure embedding compatibility
    for accurate similarity search.
    """
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None):
        """
        Initialize the query embedder.
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights to use
            device: Device for inference (auto-detect if None)
        """
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self.embedding_dim = None
        self._load_model()
        
        logger.info(f"QueryEmbedder initialized with {model_name} on {self.device}")
    
    def _load_model(self) -> None:
        """Load CLIP model and tokenizer."""
        logger.info(f"Loading CLIP model: {self.model_name}")
        
        try:
            # Load model and tokenizer
            self.model, _, _ = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Determine embedding dimension
            with torch.no_grad():
                dummy_text = self.tokenizer(["test"])
                dummy_embedding = self.model.encode_text(dummy_text)
                self.embedding_dim = dummy_embedding.shape[1]
            
            logger.info(f"CLIP model loaded successfully: {self.embedding_dim}D embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"CLIP model initialization failed: {e}")
    
    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Embed a single query text.
        
        Args:
            query_text: Query text to embed
            
        Returns:
            Normalized embedding vector
            
        Raises:
            ValueError: If query text is empty
            RuntimeError: If embedding fails
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        if self.model is None:
            raise RuntimeError("CLIP model not loaded")
        
        try:
            # Tokenize text
            tokens = self.tokenizer([query_text.strip()])
            
            # Generate embedding
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                # Normalize for cosine similarity
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embedding = text_features.cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed query '{query_text}': {e}")
            raise RuntimeError(f"Query embedding failed: {e}")
    
    def embed_batch(self, query_texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple query texts in batch for efficiency.
        
        Args:
            query_texts: List of query texts to embed
            
        Returns:
            List of normalized embedding vectors
            
        Raises:
            ValueError: If any query text is empty
            RuntimeError: If embedding fails
        """
        if not query_texts:
            return []
        
        # Validate all texts
        for text in query_texts:
            if not text or not text.strip():
                raise ValueError("All query texts must be non-empty")
        
        if self.model is None:
            raise RuntimeError("CLIP model not loaded")
        
        try:
            # Tokenize all texts
            clean_texts = [text.strip() for text in query_texts]
            tokens = self.tokenizer(clean_texts)
            
            # Generate embeddings
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                # Normalize for cosine similarity
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Convert to list of numpy arrays
            embeddings = [text_features[i].cpu().numpy() for i in range(len(query_texts))]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed batch of {len(query_texts)} queries: {e}")
            raise RuntimeError(f"Batch embedding failed: {e}")
    
    def process_request(self, request: QueryEmbeddingRequest) -> QueryEmbeddingResponse:
        """
        Process a query embedding request.
        
        Args:
            request: Query embedding request
            
        Returns:
            Query embedding response with timing information
        """
        start_time = time.time()
        
        try:
            # Embed the query
            embedding = self.embed_query(request.query_text)
            
            processing_time = time.time() - start_time
            
            return QueryEmbeddingResponse(
                embedding=embedding.tolist(),
                embedding_dim=self.embedding_dim,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Request processing failed after {processing_time:.3f}s: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "model_loaded": self.model is not None
        }


# Convenience function for standalone usage
def embed_query_text(query_text: str, model_name: str = "ViT-B-32", device: str = None) -> np.ndarray:
    """
    Convenience function to embed a single query text.
    
    Args:
        query_text: Query text to embed
        model_name: CLIP model to use
        device: Device for inference
        
    Returns:
        Normalized embedding vector
    """
    embedder = QueryEmbedder(model_name=model_name, device=device)
    return embedder.embed_query(query_text) 