#!/usr/bin/env python3
"""
Video RAG MCP Server

FastMCP-based server that exposes the Video RAG pipeline as MCP tools.
Claude Desktop can use these tools to process videos and retrieve relevant content.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastmcp import FastMCP, Context
from mcp_driver import VideoRAGMCPDriver

# Create FastMCP server
mcp = FastMCP(
    # name="Video RAG Pipeline",
    # dependencies=[
    #     "openai-whisper",
    #     "ffmpeg-python", 
    #     "open-clip-torch",
    #     "torch",
    #     "torchvision",
    #     "torchaudio",
    #     "numpy",
    #     "pandas",
    #     "chromadb",
    #     "grpcio",
    #     "grpcio-tools",
    #     "tqdm",
    #     "python-dotenv",
    #     "pydantic"
    # ]
)

# Global driver instance
_driver: Optional[VideoRAGMCPDriver] = None


def get_driver() -> VideoRAGMCPDriver:
    """Get or create the video RAG driver instance."""
    global _driver
    if _driver is None:
        try:
            _driver = VideoRAGMCPDriver(videos_dir="videos")
        except ValueError as e:
            raise ValueError(f"Failed to initialize Video RAG driver: {e}")
    return _driver


class ProcessVideosRequest(BaseModel):
    """Request model for processing videos."""
    videos_dir: str = Field(default="videos", description="Directory containing video files to process")
    force_reprocess: bool = Field(default=False, description="Force reprocessing even if data already exists")


class SearchRequest(BaseModel):
    """Request model for video search."""
    query: str = Field(description="Natural language query to search for in videos")
    k: int = Field(default=5, description="Number of results to return (1-20)")
    
    class Config:
        schema_extra = {
            "examples": [
                {
                    "query": "What topics are discussed about AI?",
                    "k": 5
                }
            ]
        }


@mcp.tool()
async def process_videos(request: ProcessVideosRequest, ctx: Context) -> Dict[str, Any]:
    """
    Process all videos in the specified directory through the Video RAG pipeline.
    
    This runs the complete pipeline:
    - Phase 1: Audio extraction, transcription with Whisper, semantic segmentation, text embedding
    - Phase 2: Frame extraction and CLIP-based visual embedding  
    - Phase 3: ChromaDB ingestion for vector storage
    
    The processed videos will be ready for semantic search queries.
    """
    await ctx.info(f"🚀 Starting video processing pipeline for directory: {request.videos_dir}")
    
    try:
        # Initialize or reinitialize driver with specified directory
        global _driver
        _driver = VideoRAGMCPDriver(videos_dir=request.videos_dir)
        
        await ctx.info(f"📁 Found {len(_driver.video_files)} video(s) to process")
        
        # Process all videos
        success = _driver.process_all_videos()
        
        if success:
            # Get final stats
            stats = _driver.get_stats()
            
            result = {
                "success": True,
                "message": "All videos processed successfully",
                "videos_processed": len(_driver.video_files),
                "database_stats": stats,
                "ready_for_search": True
            }
            
            await ctx.info("✅ Video processing completed successfully")
            return result
        else:
            result = {
                "success": False,
                "message": "Video processing failed",
                "videos_processed": 0,
                "ready_for_search": False
            }
            await ctx.error("❌ Video processing failed")
            return result
            
    except Exception as e:
        error_msg = f"Error processing videos: {str(e)}"
        await ctx.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "ready_for_search": False
        }


@mcp.tool()
async def search_videos(request: SearchRequest, ctx: Context) -> Dict[str, Any]:
    """
    Search processed videos using natural language queries.
    
    Returns relevant video segments (both audio transcripts and visual frames) 
    with timing information, content, and metadata. Results are ranked by 
    semantic similarity to the query.
    """
    await ctx.info(f"🔍 Searching videos for: '{request.query}'")
    
    try:
        driver = get_driver()
        
        # Validate k parameter
        k = max(1, min(20, request.k))
        if k != request.k:
            await ctx.info(f"📝 Adjusted k from {request.k} to {k} (valid range: 1-20)")
        
        # Execute search
        documents, json_file = driver.search_videos(request.query, k=k)
        
        if not documents:
            return {
                "success": True,
                "query": request.query,
                "total_results": 0,
                "results": [],
                "message": "No matching content found"
            }
        
        # Format results for MCP client
        results = []
        for i, doc in enumerate(documents):
            result = {
                "rank": i + 1,
                "content": doc.page_content,
                "video_id": doc.metadata.get('video_id', 'unknown'),
                "timing": doc.get_timing_info(),
                "start_time": doc.metadata.get('start', 0),
                "end_time": doc.metadata.get('end', 0),
                "modality": doc.metadata.get('modality', 'unknown'),
                "is_audio": doc.is_audio_segment(),
                "is_visual": doc.is_frame_segment(),
                "citation": f"[{doc.get_timing_info()}] {doc.metadata.get('modality')} from {doc.metadata.get('video_id')}"
            }
            
            # Add specific content based on modality
            if doc.is_audio_segment():
                result["transcript_text"] = doc.page_content
                result["word_count"] = doc.metadata.get('word_count', 0)
            elif doc.is_frame_segment():
                result["frame_description"] = doc.page_content
                result["frame_path"] = doc.metadata.get('path')
            
            results.append(result)
        
        # Summary statistics
        audio_segments = [r for r in results if r["is_audio"]]
        visual_segments = [r for r in results if r["is_visual"]]
        
        response = {
            "success": True,
            "query": request.query,
            "total_results": len(results),
            "audio_segments": len(audio_segments),
            "visual_segments": len(visual_segments),
            "results": results,
            "json_file": json_file,
            "summary": {
                "videos_included": len(set(r["video_id"] for r in results)),
                "time_range": f"{min(r['start_time'] for r in results):.1f}s - {max(r['end_time'] for r in results):.1f}s" if results else "0s - 0s",
                "total_words": sum(r.get("word_count", 0) for r in audio_segments)
            }
        }
        
        await ctx.info(f"✅ Found {len(results)} relevant segments ({len(audio_segments)} audio, {len(visual_segments)} visual)")
        return response
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        await ctx.error(error_msg)
        return {
            "success": False,
            "query": request.query,
            "message": error_msg,
            "total_results": 0,
            "results": []
        }


@mcp.tool()
async def get_database_stats(ctx: Context) -> Dict[str, Any]:
    """
    Get statistics about the processed video database.
    
    Returns information about the number of videos, audio segments, 
    visual frames, and total content available for search.
    """
    await ctx.info("📊 Retrieving database statistics...")
    
    try:
        driver = get_driver()
        stats = driver.get_stats()
        
        if "error" in stats:
            await ctx.warning(f"Database not ready: {stats['error']}")
            return {
                "success": False,
                "message": stats['error'],
                "ready": False
            }
        
        response = {
            "success": True,
            "ready": True,
            "total_documents": stats["total_documents"],
            "audio_documents": stats["audio_documents"], 
            "frame_documents": stats["frame_documents"],
            "total_videos": stats["total_videos"],
            "database_location": "data/chroma"
        }
        
        await ctx.info(f"📊 Database contains {stats['total_documents']} documents from {stats['total_videos']} videos")
        return response
        
    except Exception as e:
        error_msg = f"Failed to get database stats: {str(e)}"
        await ctx.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "ready": False
        }


@mcp.tool()
async def list_video_files(videos_dir: str = "videos", ctx: Context = None) -> Dict[str, Any]:
    """
    List all video files found in the specified directory.
    
    Helps identify which videos are available for processing.
    """
    if ctx:
        await ctx.info(f"📁 Scanning directory: {videos_dir}")
    
    try:
        videos_path = Path(videos_dir)
        if not videos_path.exists():
            return {
                "success": False,
                "message": f"Directory '{videos_dir}' does not exist",
                "videos": []
            }
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(videos_path.glob(f"*{ext}"))
        
        videos_info = []
        for video_path in video_files:
            try:
                size_mb = video_path.stat().st_size / (1024 * 1024)
                videos_info.append({
                    "filename": video_path.name,
                    "path": str(video_path),
                    "size_mb": round(size_mb, 1),
                    "video_id": video_path.stem
                })
            except Exception as e:
                if ctx:
                    await ctx.warning(f"Could not get info for {video_path.name}: {e}")
        
        response = {
            "success": True,
            "directory": videos_dir,
            "total_videos": len(videos_info),
            "videos": videos_info
        }
        
        if ctx:
            await ctx.info(f"📁 Found {len(videos_info)} video files")
        
        return response
        
    except Exception as e:
        error_msg = f"Failed to list video files: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "videos": []
        }


# Resources for data access
@mcp.resource("video-rag://stats")
async def get_stats_resource() -> str:
    """Get current database statistics."""
    try:
        driver = get_driver()
        stats = driver.get_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource("video-rag://videos/{directory}")
async def get_videos_resource(directory: str) -> str:
    """Get list of videos in a directory."""
    try:
        result = await list_video_files(directory)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# Prompts for common operations
@mcp.prompt()
def video_analysis_prompt(query: str) -> str:
    """Generate a prompt for analyzing video content."""
    return f"""Please analyze the video content related to: "{query}"

Use the search_videos tool to find relevant segments, then provide:
1. A comprehensive summary of the findings
2. Key topics and themes discussed
3. Timeline of important moments with timestamps
4. Any visual elements that support the content

Focus on providing accurate information with proper citations using the timing and video_id information."""


@mcp.prompt() 
def video_processing_help() -> str:
    """Get help with video processing workflow."""
    return """Video RAG Processing Workflow:

1. **Setup**: Place video files in the 'videos' directory
2. **Process**: Use process_videos tool to run the complete pipeline
3. **Search**: Use search_videos tool to find relevant content
4. **Analyze**: Combine results to answer questions about video content

Available tools:
- list_video_files: See what videos are available
- process_videos: Run the complete processing pipeline  
- search_videos: Search for content using natural language
- get_database_stats: Check processing status and statistics

The system processes both audio (transcripts) and visual (frames) content for comprehensive multimodal search."""


def main():
    """Main function to run the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video RAG MCP Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "http"], 
                       help="Transport protocol to use")
    parser.add_argument("--host", default="localhost", help="Host to bind to for HTTP/SSE")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to for HTTP/SSE")
    
    args = parser.parse_args()
    
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    elif args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port, path="/mcp")


if __name__ == "__main__":
    main()