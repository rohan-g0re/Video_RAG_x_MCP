#!/usr/bin/env python3
"""
Phase 1-C: Text Embedding

Generates embeddings for text segments using CLIP ViT-B/32 text encoder.
Implements batch processing and prepares data for Phase 3 database persistence.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

import torch
import open_clip
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPTextEmbedder:
    """CLIP text encoder for generating text embeddings."""
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """
        Initialize CLIP text encoder.
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights to use
        """
        logger.info(f"Loading CLIP model: {model_name} ({pretrained})")
        
        # Load CLIP model and tokenizer
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Use CPU for consistency with development plan
        self.device = "cpu"
        self.model = self.model.to(self.device)
        
        logger.info(f"CLIP model loaded successfully on {self.device}")
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_text = self.tokenizer(["test"])
            dummy_embedding = self.model.encode_text(dummy_text)
            self.embedding_dim = dummy_embedding.shape[1]
            
        logger.info(f"Text embedding dimension: {self.embedding_dim}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Maximum batch size for processing
            
        Returns:
            NumPy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        return np.vstack(embeddings)
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a single batch of texts.
        
        Args:
            texts: Batch of text strings
            
        Returns:
            NumPy array of embeddings for this batch
        """
        # Handle empty strings
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                # Use a placeholder for empty text to get consistent embeddings
                processed_texts.append("silence")
            else:
                processed_texts.append(text)
        
        # Tokenize texts
        tokens = self.tokenizer(processed_texts)
        
        # Generate embeddings
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            # Normalize embeddings (CLIP standard practice)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()


class DatabaseClient:
    """
    Stub interface for Phase 3 database integration.
    Will be replaced with actual gRPC client when Phase 3 is implemented.
    """
    
    def __init__(self):
        self.stored_embeddings = []
        logger.info("Database client initialized (stub mode)")
    
    def add_batch(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> bool:
        """
        Add a batch of embeddings with metadata to the database.
        
        Args:
            vectors: Embedding vectors (n_vectors, embedding_dim)
            metadatas: List of metadata dictionaries
            
        Returns:
            True if successful
        """
        if len(vectors) != len(metadatas):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        # Store in memory for now (Phase 3 will implement actual persistence)
        for vector, metadata in zip(vectors, metadatas):
            self.stored_embeddings.append({
                'embedding': vector.tolist(),
                'metadata': metadata
            })
        
        logger.debug(f"Stored {len(vectors)} embeddings (total: {len(self.stored_embeddings)})")
        return True
    
    def get_count(self) -> int:
        """Get total number of stored embeddings."""
        return len(self.stored_embeddings)
    
    def save_to_file(self, output_file: str):
        """Save stored embeddings to JSON file for Phase 3 integration."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.stored_embeddings, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.stored_embeddings)} embeddings to {output_file}")


class TextEmbeddingProcessor:
    """Main processor for text embedding workflow."""
    
    def __init__(self, clip_model: str = "ViT-B-32", batch_size: int = 32):
        self.embedder = CLIPTextEmbedder(clip_model)
        self.db_client = DatabaseClient()
        self.batch_size = min(batch_size, 32)  # Enforce development plan limit
        logger.info(f"Text embedding processor initialized (batch_size={self.batch_size})")
    
    def process_segmented_transcript(self, segmented_data: Dict[str, Any], 
                                   output_file: str = None) -> Dict[str, Any]:
        """
        Process segmented transcript and generate embeddings.
        
        Args:
            segmented_data: Segmented transcript data
            output_file: Optional file to save embeddings
            
        Returns:
            Processing results and performance metrics
        """
        segments = segmented_data.get('segments', [])
        video_id = segmented_data.get('video_id', 'unknown')
        
        if not segments:
            logger.warning("No segments found in transcript")
            return {'status': 'error', 'message': 'No segments to process'}
        
        logger.info(f"Processing {len(segments)} segments for video: {video_id}")
        
        # Extract texts and prepare metadata
        texts = []
        metadatas = []
        
        for i, segment in enumerate(segments):
            texts.append(segment.get('text', ''))
            
            metadata = {
                'video_id': video_id,
                'modality': 'audio',
                'start': segment.get('start', 0.0),
                'end': segment.get('end', 0.0),
                'segment_index': i,
                'word_count': segment.get('word_count', 0),
                'path': None  # No path for audio segments
            }
            metadatas.append(metadata)
        
        # Benchmark embedding performance
        start_time = time.time()
        
        # Generate embeddings
        embeddings = self.embedder.embed_texts(texts, self.batch_size)
        
        embedding_time = time.time() - start_time
        
        # Performance validation (N segments in ≤ N/8 seconds)
        max_time = len(segments) / 8.0
        performance_ok = embedding_time <= max_time
        
        logger.info(f"Embedding performance: {embedding_time:.2f}s for {len(segments)} segments")
        logger.info(f"Performance target: ≤{max_time:.2f}s - {'✓ PASS' if performance_ok else '✗ FAIL'}")
        
        # Store embeddings
        self.db_client.add_batch(embeddings, metadatas)
        
        # Save to file if requested
        if output_file:
            self.db_client.save_to_file(output_file)
        
        # Return results
        results = {
            'status': 'success',
            'video_id': video_id,
            'segments_processed': len(segments),
            'embeddings_generated': len(embeddings),
            'embedding_dimension': self.embedder.embedding_dim,
            'processing_time_seconds': embedding_time,
            'performance_target_seconds': max_time,
            'performance_ok': performance_ok,
            'embeddings_stored': self.db_client.get_count()
        }
        
        return results


def process_segmented_file(input_file: str, output_file: str = None, 
                          clip_model: str = "ViT-B-32", batch_size: int = 32) -> Dict[str, Any]:
    """
    Process a segmented transcript file and generate embeddings.
    
    Args:
        input_file: Path to segmented transcript JSON
        output_file: Path to save embeddings JSON
        clip_model: CLIP model to use
        batch_size: Batch size for processing
        
    Returns:
        Processing results
    """
    # Load segmented transcript
    with open(input_file, 'r', encoding='utf-8') as f:
        segmented_data = json.load(f)
    
    # Create processor
    processor = TextEmbeddingProcessor(clip_model, batch_size)
    
    # Process embeddings
    results = processor.process_segmented_transcript(segmented_data, output_file)
    
    return results


def main():
    """CLI entry point for text embedding."""
    parser = argparse.ArgumentParser(description="Generate CLIP text embeddings for segmented transcripts")
    parser.add_argument("input_file", help="Path to segmented transcript JSON file")
    parser.add_argument("--output-file", "-o", help="Path to save embeddings JSON file")
    parser.add_argument("--clip-model", default="ViT-B-32", 
                       help="CLIP model to use (default: ViT-B-32)")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                       help="Batch size for processing (default: 32, max: 32)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set default output file if not provided
    if args.output_file is None:
        input_path = Path(args.input_file)
        args.output_file = input_path.parent / f"{input_path.stem}_embeddings.json"
    
    # Process embeddings
    results = process_segmented_file(
        args.input_file,
        args.output_file,
        args.clip_model,
        args.batch_size
    )
    
    # Print results
    if results['status'] == 'success':
        print(f"✓ Successfully processed: {args.input_file}")
        print(f"✓ Segments processed: {results['segments_processed']}")
        print(f"✓ Embeddings generated: {results['embeddings_generated']}")
        print(f"✓ Embedding dimension: {results['embedding_dimension']}")
        print(f"✓ Processing time: {results['processing_time_seconds']:.2f}s")
        print(f"✓ Performance target: {results['performance_target_seconds']:.2f}s")
        print(f"✓ Performance: {'PASS' if results['performance_ok'] else 'FAIL'}")
        if args.output_file:
            print(f"✓ Embeddings saved to: {args.output_file}")
    else:
        print(f"✗ Error: {results.get('message', 'Unknown error')}")
        exit(1)


if __name__ == "__main__":
    main() 