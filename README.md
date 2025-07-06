# MCP Video RAG System

A local video retrieval-augmented generation system powered by MCP (Model Context Protocol).

## Overview

The MCP Video RAG System is a comprehensive solution for processing videos, extracting content, and enabling intelligent search and question-answering capabilities. It leverages the power of MCP to integrate with advanced AI models while maintaining local processing capabilities.

## Features

- **Video Processing**: Extract frames, audio, and metadata from various video formats
- **Audio Transcription**: Generate accurate transcriptions with word-level timestamps
- **Visual Analysis**: Analyze video content using AI vision models via MCP
- **Content Enrichment**: Create searchable metadata and semantic representations
- **Vector Search**: Efficient similarity search across video content
- **RAG Integration**: Generate intelligent answers based on video content
- **CLI Interface**: User-friendly command-line interface for all operations
- **Video Clip Generation**: Extract relevant video segments based on queries

## System Requirements

- Python 3.9+
- FFmpeg (for video processing)
- Git (for version control)
- Cursor Pro (for MCP integration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mcpvideorag/mcp-video-rag.git
   cd mcp-video-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/macOS
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the system:
   ```bash
   video-rag setup
   ```

## Quick Start

1. **Ingest Videos**: Add videos to the system for processing
   ```bash
   video-rag ingest /path/to/your/videos
   ```

2. **Query Content**: Ask questions about your video content
   ```bash
   video-rag query "What topics are discussed in the meeting?"
   ```

3. **Generate Clips**: Extract relevant video segments
   ```bash
   video-rag clips "Show me the discussion about project timeline"
   ```

## Architecture

The system is organized into several key modules:

- **Core**: Foundational classes and interfaces
- **Video**: Video processing and frame extraction
- **MCP**: Model Context Protocol bridge for AI integration
- **Vector Store**: Embedding storage and similarity search
- **CLI**: Command-line interface and user interactions
- **Storage**: Database and file management
- **Config**: Configuration management

## Development

### Project Structure

```
mcp-video-rag/
├── src/
│   └── video_rag/
│       ├── core/          # Core functionality
│       ├── video/         # Video processing
│       ├── mcp/           # MCP bridge
│       ├── vector_store/  # Vector database
│       ├── cli/           # Command-line interface
│       └── ...
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
├── docs/                  # Documentation
├── config/                # Configuration files
└── examples/              # Usage examples
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/video_rag

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks for quality checks

Run quality checks:
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [MCP (Model Context Protocol)](https://github.com/modelcontextprotocol)
- Powered by [Cursor Pro](https://cursor.sh/) for AI integration
- Video processing via [FFmpeg](https://ffmpeg.org/)
- Vector search via [ChromaDB](https://www.trychroma.com/)

## Support

For support, please open an issue on GitHub or contact us at support@mcpvideorag.com.

---

**Note**: This project is in active development. Features and APIs may change between versions. 