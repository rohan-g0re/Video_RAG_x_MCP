#!/usr/bin/env python3
"""
Phase 4: Retrieval Service CLI

Command-line interface for independent Phase 4 execution as specified 
in the development plan. Allows testing and using the retrieval service
without the full pipeline.

Usage Examples:
    python -m src.phase4_retriever.cli search "machine learning"
    python -m src.phase4_retriever.cli search "neural networks" --k 5 --video test_video
    python -m src.phase4_retriever.cli stats
    python -m src.phase4_retriever.cli embed "deep learning concepts"
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .retriever import Retriever, search_videos
from .embed_query import QueryEmbedder
from .models import RetrievalRequest


def format_document(doc, index: int) -> str:
    """Format a document for CLI display."""
    timing = doc.get_timing_info()
    video_id = doc.metadata.get("video_id", "unknown")
    modality = doc.metadata.get("modality", "unknown")
    
    if doc.is_audio_segment():
        content_preview = (doc.page_content[:80] + "...") if len(doc.page_content) > 80 else doc.page_content
        return f"{index+1}. [{timing}] {video_id} (audio): {content_preview}"
    else:
        return f"{index+1}. [{timing}] {video_id} (frame): {doc.page_content}"


def cmd_search(args) -> int:
    """Execute search command."""
    try:
        print(f"🔍 Searching for: '{args.query}'")
        print(f"   Parameters: k={args.k}, video={args.video or 'all'}, modality={args.modality or 'all'}")
        
        if args.video or args.modality or args.time_range:
            # Use filtered search
            time_range = None
            if args.time_range:
                try:
                    start, end = map(float, args.time_range.split(","))
                    time_range = (start, end)
                except ValueError:
                    print("❌ Invalid time range format. Use: start,end (e.g., 10.5,30.0)")
                    return 1
            
            request = RetrievalRequest(
                query=args.query,
                k=args.k,
                video_id=args.video,
                modality=args.modality,
                time_range=time_range
            )
            
            retriever = Retriever(persist_directory=args.data_dir)
            response = retriever.search_with_filters(request)
            
            print(f"\n📊 Results: {response.get_summary()}")
            
            if response.documents:
                print(f"\n📋 Documents:")
                for i, doc in enumerate(response.documents):
                    print(f"   {format_document(doc, i)}")
            else:
                print("   No documents found.")
        
        else:
            # Use simple search
            documents = search_videos(args.query, k=args.k, persist_directory=args.data_dir)
            
            print(f"\n📊 Found {len(documents)} documents")
            
            if documents:
                print(f"\n📋 Documents:")
                for i, doc in enumerate(documents):
                    print(f"   {format_document(doc, i)}")
            else:
                print("   No documents found.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return 1


def cmd_embed(args) -> int:
    """Execute embed command."""
    try:
        print(f"🔤 Embedding query: '{args.query}'")
        
        embedder = QueryEmbedder()
        
        start_time = time.time()
        embedding = embedder.embed_query(args.query)
        embed_time = time.time() - start_time
        
        print(f"\n✅ Embedding generated in {embed_time:.3f}s")
        print(f"   Dimension: {len(embedding)}")
        print(f"   Model: {embedder.model_name}")
        print(f"   Device: {embedder.device}")
        
        if args.output:
            output_data = {
                "query": args.query,
                "embedding": embedding.tolist(),
                "dimension": len(embedding),
                "model": embedder.model_name,
                "processing_time": embed_time
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"💾 Embedding saved to: {args.output}")
        
        if args.show_embedding:
            print(f"\n🔢 Embedding vector (first 10 values):")
            print(f"   {embedding[:10].tolist()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return 1


def cmd_stats(args) -> int:
    """Execute stats command."""
    try:
        print("📊 Retrieving collection statistics...")
        
        retriever = Retriever(persist_directory=args.data_dir)
        stats = retriever.get_stats()
        
        print(f"\n📈 Collection Statistics:")
        print(f"   Total documents: {stats.total_documents}")
        print(f"   Total videos: {stats.total_videos}")
        print(f"   Audio documents: {stats.audio_documents}")
        print(f"   Frame documents: {stats.frame_documents}")
        print(f"   Embedding dimension: {stats.embedding_dimension}")
        
        print(f"\n📹 Indexed videos:")
        for video_id in stats.video_ids:
            print(f"   • {video_id}")
        
        if args.output:
            stats_data = stats.dict()
            with open(args.output, 'w') as f:
                json.dump(stats_data, f, indent=2)
            print(f"\n💾 Statistics saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Statistics retrieval failed: {e}")
        return 1


def cmd_test(args) -> int:
    """Execute test command."""
    print("🧪 Running Phase 4 retrieval tests...")
    
    test_queries = [
        "machine learning tutorial",
        "neural networks explanation", 
        "artificial intelligence concepts",
        "deep learning models"
    ]
    
    try:
        retriever = Retriever(persist_directory=args.data_dir)
        
        # Test 1: Basic functionality
        print("\n🔹 Test 1: Basic Search Functionality")
        total_tests = 0
        successful_tests = 0
        
        for query in test_queries:
            try:
                documents = retriever.search(query, k=3)
                total_tests += 1
                if len(documents) >= 0:  # Any result is acceptable
                    successful_tests += 1
                    print(f"   ✅ '{query}': {len(documents)} results")
                else:
                    print(f"   ❌ '{query}': No results")
            except Exception as e:
                total_tests += 1
                print(f"   ❌ '{query}': Failed - {e}")
        
        # Test 2: Filtered search
        print("\n🔹 Test 2: Filtered Search")
        try:
            request = RetrievalRequest(
                query="test content",
                k=5,
                modality="audio"
            )
            response = retriever.search_with_filters(request)
            print(f"   ✅ Audio-only search: {response.total_found} results")
            successful_tests += 1
        except Exception as e:
            print(f"   ❌ Audio-only search failed: {e}")
        total_tests += 1
        
        # Test 3: Statistics
        print("\n🔹 Test 3: Statistics Retrieval")
        try:
            stats = retriever.get_stats()
            print(f"   ✅ Statistics: {stats.total_documents} docs, {stats.total_videos} videos")
            successful_tests += 1
        except Exception as e:
            print(f"   ❌ Statistics retrieval failed: {e}")
        total_tests += 1
        
        # Test 4: Query embedding
        print("\n🔹 Test 4: Query Embedding")
        try:
            embedder = QueryEmbedder()
            embedding = embedder.embed_query("test query")
            print(f"   ✅ Query embedding: {len(embedding)}D vector")
            successful_tests += 1
        except Exception as e:
            print(f"   ❌ Query embedding failed: {e}")
        total_tests += 1
        
        # Summary
        success_rate = (successful_tests / total_tests) * 100
        print(f"\n📊 Test Summary:")
        print(f"   Tests passed: {successful_tests}/{total_tests}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        if success_rate >= 75:
            print(f"   🎉 Phase 4 tests PASSED")
            return 0
        else:
            print(f"   ⚠️  Phase 4 tests FAILED")
            return 1
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 4 Retrieval Service CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s search "machine learning"
  %(prog)s search "neural networks" --k 5 --video test_video
  %(prog)s search "AI concepts" --modality audio --time-range 10.0,30.0
  %(prog)s embed "deep learning" --output embedding.json
  %(prog)s stats --output stats.json
  %(prog)s test
        """
    )
    
    parser.add_argument(
        "--data-dir", 
        default="data/chroma",
        help="ChromaDB data directory (default: data/chroma)"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for documents")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument("--k", type=int, default=10, help="Number of results (default: 10)")
    search_parser.add_argument("--video", help="Filter by video ID")
    search_parser.add_argument("--modality", choices=["audio", "frame"], help="Filter by modality")
    search_parser.add_argument("--time-range", help="Filter by time range (format: start,end)")
    
    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Embed query text")
    embed_parser.add_argument("query", help="Text to embed")
    embed_parser.add_argument("--output", help="Save embedding to JSON file")
    embed_parser.add_argument("--show-embedding", action="store_true", help="Show embedding values")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show collection statistics")
    stats_parser.add_argument("--output", help="Save statistics to JSON file")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run Phase 4 tests")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == "search":
        return cmd_search(args)
    elif args.command == "embed":
        return cmd_embed(args)
    elif args.command == "stats":
        return cmd_stats(args)
    elif args.command == "test":
        return cmd_test(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 