#!/usr/bin/env python3

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastmcp import FastMCP, Context
from mcp_driver import VideoRAGMCPDriver

# Create FastMCP server
mcp = FastMCP()


class VideoQueryRequest(BaseModel):
    """Request model for video search."""
    query: str = Field(description="Natural language query to search for in videos")
    k: int = Field(default=5, description="Number of results to return (1-20)", ge=1, le=20)
    videos_dir: str = Field(default="videos", description="Directory containing video files to process")


@mcp.tool()
async def search_video_content(request: VideoQueryRequest, ctx: Context) -> Dict[str, Any]:
    """
    Search video content using natural language queries.
    
    This tool automatically:
    1. Processes all videos in the directory (if not already processed)
    2. Searches the vector database for relevant content
    3. Returns top k relevant chunks with full content and metadata
    
    Returns both audio transcript segments and visual frame descriptions 
    ranked by semantic similarity to your query.
    """
    await ctx.info(f"SEARCH: Searching for: '{request.query}' (top {request.k} results)")
    
    try:
        # Initialize driver
        driver = VideoRAGMCPDriver(videos_dir=request.videos_dir)
        await ctx.info(f"FILES: Found {len(driver.video_files)} video(s) in {request.videos_dir}")
        
        # Use complete search workflow - driver handles everything
        response = driver.search_and_format_for_mcp(request.query, k=request.k)
        
        # Log results
        if response.get("success"):
            summary = response.get("summary", {})
            await ctx.info(f"SUCCESS: Found {response.get('total_results', 0)} segments ({summary.get('audio_segments', 0)} audio, {summary.get('visual_segments', 0)} visual)")
        else:
            await ctx.error(f"ERROR: Search failed: {response.get('error', 'Unknown error')}")
        
        return response
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        await ctx.error(error_msg)
        return {
            "success": False,
            "query": request.query,
            "error": error_msg,
            "results": []
        }


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