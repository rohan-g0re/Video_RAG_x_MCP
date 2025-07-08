"""
Frame Embedding Module - Phase 2B

Embeds extracted video frames using CLIP ViT-B/32 image encoder.
Processes frames in batches and stores embeddings with metadata.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
from tqdm import tqdm
import hashlib

from .sample_frames import FrameMetadata


@dataclass
class FrameEmbedding:
    """Container for frame embedding with metadata."""
    video_id: str
    start: float
    end: float
    frame_path: str
    timestamp: float
    embedding: np.ndarray
    embedding_dim: int
    model_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert numpy array to list for JSON serialization
        result['embedding'] = self.embedding.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrameEmbedding':
        """Create from dictionary."""
        # Convert list back to numpy array
        data['embedding'] = np.array(data['embedding'])
        return cls(**data)


class FrameDataset(Dataset):
    """Dataset for loading and preprocessing frame images."""
    
    def __init__(self, frame_metadata: List[FrameMetadata], preprocess_fn):
        self.frame_metadata = frame_metadata
        self.preprocess_fn = preprocess_fn
    
    def __len__(self) -> int:
        return len(self.frame_metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, FrameMetadata]:
        frame_meta = self.frame_metadata[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(frame_meta.frame_path).convert('RGB')
            image_tensor = self.preprocess_fn(image)
            return image_tensor, frame_meta
        except Exception as e:
            raise RuntimeError(f"Failed to load image {frame_meta.frame_path}: {e}")


class FrameEmbedder:
    """
    Embeds video frames using CLIP ViT-B/32 image encoder.
    
    Usage:
        embedder = FrameEmbedder()
        embeddings = embedder.embed_frames(frame_metadata)
    """
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None):
        """
        Initialize the frame embedder.
        
        Args:
            model_name: CLIP model architecture (default: ViT-B-32)
            pretrained: Pretrained weights source (default: openai)
            device: Device to run inference on (auto-detect if None)
        """
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔧 Initializing CLIP {model_name} on {self.device}")
        
        # Load CLIP model and preprocessing
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.model.eval()  # Set to evaluation mode
            
            # Get embedding dimension
            with torch.no_grad():
                dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_embedding = self.model.encode_image(dummy_image)
                self.embedding_dim = dummy_embedding.shape[-1]
            
            print(f"✅ CLIP model loaded successfully")
            print(f"   Model: {model_name} ({pretrained})")
            print(f"   Embedding dimension: {self.embedding_dim}")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")
    
    def embed_frames(self, frame_metadata: List[FrameMetadata], batch_size: int = 64) -> List[FrameEmbedding]:
        """
        Embed a list of frames using CLIP image encoder.
        
        Args:
            frame_metadata: List of frame metadata objects
            batch_size: Batch size for processing (default: 64)
            
        Returns:
            List of FrameEmbedding objects
        """
        if not frame_metadata:
            return []
        
        print(f"🖼️ Embedding {len(frame_metadata)} frames (batch_size={batch_size})")
        
        # Create dataset and dataloader
        dataset = FrameDataset(frame_metadata, self.preprocess)
        
        # Custom collate function to handle FrameMetadata objects
        def collate_fn(batch):
            images = torch.stack([item[0] for item in batch])
            metadata = [item[1] for item in batch]  # Keep as list
            return images, metadata
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        embeddings = []
        
        with torch.no_grad():
            for batch_images, batch_metadata in tqdm(dataloader, desc="Embedding frames"):
                # Move images to device
                batch_images = batch_images.to(self.device)
                
                # Generate embeddings
                batch_embeddings = self.model.encode_image(batch_images)
                
                # Convert to numpy and normalize
                batch_embeddings = batch_embeddings.cpu().numpy()
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                
                # Create FrameEmbedding objects
                for i, (embedding, frame_meta) in enumerate(zip(batch_embeddings, batch_metadata)):
                    frame_embedding = FrameEmbedding(
                        video_id=frame_meta.video_id,
                        start=frame_meta.start,
                        end=frame_meta.end,
                        frame_path=frame_meta.frame_path,
                        timestamp=frame_meta.timestamp,
                        embedding=embedding,
                        embedding_dim=self.embedding_dim,
                        model_name=f"{self.model_name}_{self.pretrained}"
                    )
                    embeddings.append(frame_embedding)
        
        print(f"✅ Successfully embedded {len(embeddings)} frames")
        return embeddings
    
    def save_embeddings(self, embeddings: List[FrameEmbedding], output_path: str, format: str = "json") -> str:
        """
        Save embeddings to file.
        
        Args:
            embeddings: List of frame embeddings
            output_path: Path to save embeddings
            format: File format ("json" or "pickle")
            
        Returns:
            SHA256 checksum of the saved file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == "json":
            # Convert to JSON-serializable format
            data = {
                "model_info": {
                    "model_name": self.model_name,
                    "pretrained": self.pretrained,
                    "embedding_dim": self.embedding_dim
                },
                "total_embeddings": len(embeddings),
                "embeddings": [emb.to_dict() for emb in embeddings]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == "pickle":
            # Save as pickle (more efficient for large arrays)
            data = {
                "model_info": {
                    "model_name": self.model_name,
                    "pretrained": self.pretrained,
                    "embedding_dim": self.embedding_dim
                },
                "total_embeddings": len(embeddings),
                "embeddings": embeddings
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")
        
        # Calculate checksum for validation
        checksum = self._calculate_file_checksum(output_path)
        
        print(f"✅ Embeddings saved to: {output_path}")
        print(f"   Format: {format}")
        print(f"   File size: {os.path.getsize(output_path)} bytes")
        print(f"   SHA256: {checksum}")
        
        return checksum
    
    def load_embeddings(self, input_path: str, format: str = None) -> List[FrameEmbedding]:
        """
        Load embeddings from file.
        
        Args:
            input_path: Path to load embeddings from
            format: File format (auto-detect if None)
            
        Returns:
            List of FrameEmbedding objects
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Embeddings file not found: {input_path}")
        
        # Auto-detect format from extension
        if format is None:
            if input_path.endswith('.json'):
                format = "json"
            elif input_path.endswith('.pkl') or input_path.endswith('.pickle'):
                format = "pickle"
            else:
                raise ValueError(f"Cannot auto-detect format for: {input_path}")
        
        if format == "json":
            with open(input_path, 'r') as f:
                data = json.load(f)
            embeddings = [FrameEmbedding.from_dict(emb_data) for emb_data in data['embeddings']]
        
        elif format == "pickle":
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            embeddings = data['embeddings']
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ Loaded {len(embeddings)} embeddings from {input_path}")
        return embeddings
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


def main():
    """Command-line interface for frame embedding."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embed video frames using CLIP image encoder")
    parser.add_argument("frames_metadata", help="Path to frame metadata JSON file")
    parser.add_argument("--output", required=True, help="Output path for embeddings")
    parser.add_argument("--format", choices=["json", "pickle"], default="json", help="Output format")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--model", default="ViT-B-32", help="CLIP model name")
    parser.add_argument("--pretrained", default="openai", help="Pretrained weights source")
    
    args = parser.parse_args()
    
    try:
        # Load frame metadata
        with open(args.frames_metadata, 'r') as f:
            metadata_data = json.load(f)
        
        # Convert to FrameMetadata objects
        frame_metadata = []
        for frame_data in metadata_data['frames']:
            frame_meta = FrameMetadata(
                video_id=frame_data['video_id'],
                start=frame_data['start'],
                end=frame_data['end'],
                frame_path=frame_data['frame_path'],
                timestamp=frame_data['timestamp']
            )
            frame_metadata.append(frame_meta)
        
        print(f"📁 Loaded {len(frame_metadata)} frame metadata entries")
        
        # Initialize embedder
        embedder = FrameEmbedder(model_name=args.model, pretrained=args.pretrained)
        
        # Embed frames
        embeddings = embedder.embed_frames(frame_metadata, batch_size=args.batch_size)
        
        # Save embeddings
        checksum = embedder.save_embeddings(embeddings, args.output, format=args.format)
        
        print(f"\n✅ Frame embedding completed!")
        print(f"   Embeddings generated: {len(embeddings)}")
        print(f"   Output file: {args.output}")
        print(f"   Checksum: {checksum}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 