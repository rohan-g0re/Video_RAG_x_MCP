"""
MCP Video RAG System - A local video retrieval-augmented generation system powered by MCP.

This package provides a complete solution for processing videos, extracting content,
and enabling intelligent search and question-answering capabilities.
"""

__version__ = "0.1.0"
__author__ = "MCP Video RAG Team"
__email__ = "support@mcpvideorag.com"
__license__ = "MIT"

# Core package metadata
__title__ = "MCP Video RAG"
__description__ = "Local video retrieval-augmented generation system powered by MCP"
__url__ = "https://github.com/mcpvideorag/mcp-video-rag"

# Version information
VERSION = __version__
VERSION_INFO = tuple(map(int, __version__.split('.')))

# Import main components for easy access
from .core import *
from .models import *
from .config import *

# Export public API
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "__title__",
    "__description__",
    "__url__",
    "VERSION",
    "VERSION_INFO",
] 