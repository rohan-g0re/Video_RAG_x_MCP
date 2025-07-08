"""
Phase 2: Visual Frame Extraction & Embedding

This module handles visual processing for the Video RAG pipeline:
- Frame sampling at 10-second intervals
- CLIP image embedding of extracted frames
"""

__version__ = "1.0.0"

from .sample_frames import FrameSampler
from .embed_frames import FrameEmbedder

__all__ = ["FrameSampler", "FrameEmbedder"] 