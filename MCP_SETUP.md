# Video RAG MCP Server Setup

This document explains how to set up and use the Video RAG MCP (Model Context Protocol) server with Claude Desktop.

## 🏗️ Architecture Overview

The MCP server exposes the Video RAG pipeline through Claude Desktop, allowing you to:
- Process videos through phases 1-4 (audio transcription, visual analysis, embedding, and retrieval)
- Search video content using natural language queries
- Get structured results that Claude can use for analysis and generation

**Phase Flow:**
1. **Phase 1**: Audio extraction → Whisper transcription → semantic segmentation → text embedding
2. **Phase 2**: Frame sampling → CLIP visual embedding
3. **Phase 3**: ChromaDB vector storage
4. **Phase 4**: Semantic search and retrieval
5. **Claude Desktop**: Uses retrieved content for LLM generation and analysis

## 🚀 Setup Instructions

### Prerequisites
- **Python 3.11+** (required for ChromaDB compatibility)
- **uv** package manager (for dependency management)
- **FFmpeg** (for video processing)
- **Claude Desktop** application

### 1. Install uv (if not already installed)

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Project Environment

```bash
# Navigate to project directory
cd "Video RAG using MCP"

# Sync dependencies (already done if you followed the setup)
uv sync

# Verify installation
uv run python -c "import fastmcp; print('FastMCP installed successfully')"
```

### 3. Prepare Video Files

Place your video files in the `videos/` directory:
```bash
mkdir -p videos
# Copy your .mp4, .avi, .mov, .mkv, .wmv, .flv, or .webm files here
```

### 4. Install MCP Server for Claude Desktop

Install the MCP server so Claude Desktop can use it:

```bash
# Install the MCP server (this configures Claude Desktop automatically)
uv run fastmcp install mcp_server.py --name "Video RAG Pipeline"
```

This command:
- Creates an isolated environment for the MCP server
- Adds the server configuration to Claude Desktop's config file
- Installs all required dependencies

### 5. Alternative: Manual Claude Desktop Configuration

If the automatic installation doesn't work, manually add to Claude Desktop config:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "video-rag": {
      "command": "uv",
      "args": [
        "run", 
        "--directory",
        "D:/STUFF/Projects/Video RAG using MCP",
        "python",
        "mcp_server.py"
      ],
      "env": {
        "UV_PROJECT_ENVIRONMENT": ".venv"
      }
    }
  }
}
```

**Important:** Replace the directory path with your actual project path.

## 🎯 Usage Guide

### Step 1: Start Claude Desktop
Launch Claude Desktop - it will automatically connect to the Video RAG MCP server.

### Step 2: Process Videos
```
I need to process the videos in my collection. Can you run the video processing pipeline?
```

Claude will use the `process_videos` tool to:
- Extract and transcribe audio
- Sample and embed video frames  
- Store everything in the vector database

### Step 3: Search and Analyze
```
Search for content about "artificial intelligence" in my videos and provide a comprehensive analysis.
```

Claude will:
- Use the `search_videos` tool to find relevant segments
- Analyze both audio transcripts and visual frames
- Provide detailed responses with proper citations

## 🔧 Available MCP Tools

### Core Tools
- **`process_videos`**: Run the complete video processing pipeline
- **`search_videos`**: Search video content using natural language queries  
- **`get_database_stats`**: Get statistics about processed videos
- **`list_video_files`**: List available video files

### Resources
- **`video-rag://stats`**: Access database statistics
- **`video-rag://videos/{directory}`**: List videos in a directory

### Prompts
- **`video_analysis_prompt`**: Template for comprehensive video analysis
- **`video_processing_help`**: Workflow guidance

## 📝 Example Interactions

### Processing Videos
```
User: I have some videos about machine learning. Can you process them for analysis?

Claude: I'll help you process your machine learning videos through the Video RAG pipeline. Let me start by checking what videos are available and then running the processing pipeline.

[Uses list_video_files and process_videos tools]
```

### Searching Content
```
User: What concepts are explained in my videos about neural networks?

Claude: I'll search your processed videos for neural network content and provide a comprehensive analysis.

[Uses search_videos tool and analyzes results]
```

### Getting Statistics
```
User: How much video content do I have processed?

Claude: Let me check your video database statistics.

[Uses get_database_stats tool]
```

## 🛠️ Development and Testing

### Test the MCP Server Directly
```bash
# Test with MCP Inspector (development mode)
uv run fastmcp dev mcp_server.py

# Run server manually
uv run python mcp_server.py

# Run with different transport
uv run python mcp_server.py --transport sse --port 8000
```

### Debug Issues
```bash
# Check if dependencies are properly installed
uv run python -c "import chromadb, torch, whisper; print('All core dependencies available')"

# Test video processing pipeline directly
uv run python mcp_driver.py --video videos

# Check ChromaDB status
ls data/chroma/
```

## 🔍 Troubleshooting

### Common Issues

**1. "No video files found"**
- Ensure videos are in the `videos/` directory
- Check supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm

**2. "Components not available"**
- Run `uv sync` to ensure all dependencies are installed
- Check that ChromaDB is properly installed: `uv run python -c "import chromadb"`

**3. "MCP server not connecting"**
- Verify Claude Desktop configuration
- Check that the project path in config is absolute and correct
- Restart Claude Desktop after configuration changes

**4. "Processing fails"**
- Ensure FFmpeg is installed and in PATH
- Check video file integrity
- Verify sufficient disk space in `data/` directory

### Performance Notes
- **Memory**: Video processing requires ~2-4GB RAM per video
- **Storage**: ~100-500MB per hour of video content  
- **Processing Time**: ~2-5 minutes per minute of video content

## 🎉 Success Indicators

When everything is working correctly:
1. Claude Desktop shows the Video RAG MCP server as connected
2. You can ask Claude to process videos and it uses the tools
3. Search queries return relevant video segments with timestamps
4. Results include both audio transcripts and visual frame descriptions

## 📚 Architecture Benefits

**Why MCP for Video RAG?**
- **Separation of Concerns**: Video processing runs locally, LLM generation in Claude
- **Security**: No need to send video content to external APIs
- **Efficiency**: Claude Desktop handles the conversation, MCP handles video processing
- **Flexibility**: Can switch between different LLM clients while keeping same video processing
- **Cost-Effective**: Leverages Claude Desktop's built-in models [[memory:2315250]]

## 🚧 Next Steps

Once your MCP server is running:
1. Process your video collection
2. Experiment with different search queries
3. Ask Claude to analyze themes, extract insights, or create summaries
4. Use the citation information to reference specific moments in videos

The MCP server provides the retrieval capabilities, while Claude Desktop provides the intelligence to analyze and generate insights from your video content.