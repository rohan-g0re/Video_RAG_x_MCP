#!/usr/bin/env python3
"""
Phase 1-C Enhanced: Text Embedding for Simplified Semantic Segments

Updated text embedding that works with simplified semantic-aware segmentation.
Handles the simplified metadata structure without pause/topic detection.
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


class SemanticEmbeddingProcessor:
    """Enhanced text embedding processor for simplified semantic segments."""
    
    def __init__(self, clip_model: str = "ViT-B-32", batch_size: int = 32):
        self.clip_model = clip_model
        self.batch_size = min(batch_size, 32)
        self._load_clip_model()
        self.stored_embeddings = []
        logger.info(f"Semantic embedding processor initialized (batch_size={self.batch_size})")
    
    def _load_clip_model(self):
        """Load CLIP model for text embedding."""
        logger.info(f"Loading CLIP model: {self.clip_model}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.clip_model, pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer(self.clip_model)
        
        # Use CPU for consistency
        self.device = "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_text = self.tokenizer(["test"])
            dummy_embedding = self.model.encode_text(dummy_text)
            self.embedding_dim = dummy_embedding.shape[1]
            
        logger.info(f"CLIP model loaded: {self.embedding_dim}D embeddings")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding captions"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a single batch of texts."""
        # Handle empty captions
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("silence")
            else:
                processed_texts.append(text)
        
        # Tokenize texts
        tokens = self.tokenizer(processed_texts)
        
        # Generate embeddings
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def process_semantic_segments(self, segmented_data: Dict[str, Any], 
                                output_file: str = None) -> Dict[str, Any]:
        """Process simplified semantic segmented transcript and generate embeddings."""
        
        segments = segmented_data.get('segments', [])
        video_id = segmented_data.get('video_id', 'unknown')
        
        if not segments:
            logger.warning("No segments found in transcript")
            return {'status': 'error', 'message': 'No segments to process'}
        
        logger.info(f"Processing {len(segments)} simplified semantic segments for video: {video_id}")
        
        # Extract captions and prepare enhanced metadata
        captions = []
        metadatas = []
        
        for i, segment in enumerate(segments):
            captions.append(segment.get('caption', ''))
            
            # Create enhanced metadata combining segment and content info
            metadata = {
                'video_id': video_id,
                'modality': 'audio',
                'start': segment.get('start', 0.0),
                'end': segment.get('end', 0.0),
                'segment_index': i,
                'word_count': segment.get('word_count', 0),
                'path': None,  # No path for audio segments
                # Add semantic-specific metadata (simplified)
                'segmentation_method': segmented_data.get('segmentation_method', 'semantic_adaptive_overlap'),
                'duration': segment.get('metadata', {}).get('duration', 0.0),
                'content_type': segment.get('metadata', {}).get('content_type', 'unknown'),
                'sentence_count': segment.get('metadata', {}).get('sentence_count', 0),
                'timing_formatted': segment.get('metadata', {}).get('timing_formatted', ''),
                # Add overlap information if available
                'overlap_added': segment.get('metadata', {}).get('overlap_added', 0.0),
                'original_start': segment.get('metadata', {}).get('original_start'),
                'original_end': segment.get('metadata', {}).get('original_end')
            }
            metadatas.append(metadata)
        
        # Benchmark embedding performance
        start_time = time.time()
        embeddings = self.embed_texts(captions)
        embedding_time = time.time() - start_time
        
        # Performance validation (N segments in ≤ N/8 seconds)
        max_time = len(segments) / 8.0
        performance_ok = embedding_time <= max_time
        
        logger.info(f"Embedding performance: {embedding_time:.2f}s for {len(segments)} segments")
        logger.info(f"Performance target: ≤{max_time:.2f}s - {'✓ PASS' if performance_ok else '✗ FAIL'}")
        
        # Store embeddings with enhanced metadata
        for embedding, metadata, caption in zip(embeddings, metadatas, captions):
            self.stored_embeddings.append({
                'embedding': embedding.tolist(),
                'metadata': metadata,
                'caption': caption  # ← FIX: Include actual transcript text
            })
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.stored_embeddings, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.stored_embeddings)} embeddings to {output_file}")
        
        # Return results with enhanced metrics
        results = {
            'status': 'success',
            'video_id': video_id,
            'segmentation_method': segmented_data.get('segmentation_method', 'semantic_adaptive_overlap'),
            'segments_processed': len(segments),
            'embeddings_generated': len(embeddings),
            'embedding_dimension': self.embedding_dim,
            'processing_time_seconds': embedding_time,
            'performance_target_seconds': max_time,
            'performance_ok': performance_ok,
            'embeddings_stored': len(self.stored_embeddings),
            # Simplified semantic-specific metrics
            'content_type_distribution': self._analyze_content_types(segments),
            'avg_segment_duration': sum(seg.get('metadata', {}).get('duration', 0) for seg in segments) / len(segments),
            'overlap_enabled': any(seg.get('metadata', {}).get('overlap_added', 0) > 0 for seg in segments),
            'readability_score': self._calculate_readability_score(segments)
        }
        
        return results
    
    def _analyze_content_types(self, segments: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of content types."""
        content_types = {}
        for segment in segments:
            content_type = segment.get('metadata', {}).get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        return content_types
    
    def _calculate_readability_score(self, segments: List[Dict]) -> float:
        """Calculate a simple readability score based on caption quality."""
        if not segments:
            return 0.0
        
        readable_segments = 0
        for segment in segments:
            caption = segment.get('caption', '')
            # Consider readable if has punctuation, capitalization, and reasonable length
            if (len(caption) > 10 and 
                any(c in caption for c in '.!?') and 
                any(c.isupper() for c in caption)):
                readable_segments += 1
        
        return readable_segments / len(segments)


def process_semantic_segmented_file(input_file: str, output_file: str = None, 
                                  clip_model: str = "ViT-B-32", batch_size: int = 32) -> Dict[str, Any]:
    """Process a simplified semantic segmented transcript file and generate embeddings."""
    
    # Load segmented transcript
    with open(input_file, 'r', encoding='utf-8') as f:
        segmented_data = json.load(f)
    
    # Create processor
    processor = SemanticEmbeddingProcessor(clip_model, batch_size)
    
    # Process embeddings
    results = processor.process_semantic_segments(segmented_data, output_file)
    
    return results


def main():
    """CLI entry point for simplified semantic text embedding."""
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for simplified semantic segments")
    parser.add_argument("input_file", help="Path to semantic segmented transcript JSON file")
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
    results = process_semantic_segmented_file(
        args.input_file,
        args.output_file,
        args.clip_model,
        args.batch_size
    )
    
    # Print results
    if results['status'] == 'success':
        print(f"✓ Successfully processed: {args.input_file}")
        print(f"✓ Method: {results['segmentation_method']}")
        print(f"✓ Segments processed: {results['segments_processed']}")
        print(f"✓ Embeddings generated: {results['embeddings_generated']}")
        print(f"✓ Embedding dimension: {results['embedding_dimension']}")
        print(f"✓ Processing time: {results['processing_time_seconds']:.2f}s")
        print(f"✓ Performance: {'PASS' if results['performance_ok'] else 'FAIL'}")
        print(f"✓ Readability score: {results['readability_score']:.1%}")
        print(f"✓ Overlap enabled: {results['overlap_enabled']}")
        print(f"✓ Avg segment duration: {results['avg_segment_duration']:.1f}s")
        
        print("\n📊 Content Type Distribution:")
        for content_type, count in results['content_type_distribution'].items():
            percentage = (count / results['segments_processed']) * 100
            print(f"  • {content_type}: {count} ({percentage:.1f}%)")
        
        if args.output_file:
            print(f"✓ Embeddings saved to: {args.output_file}")
    else:
        print(f"✗ Error: {results.get('message', 'Unknown error')}")
        exit(1)


if __name__ == "__main__":
    main() 