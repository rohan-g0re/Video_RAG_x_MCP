#!/usr/bin/env python3
"""
ChromaDB Embeddings Verification Script

This script allows you to inspect and verify the embeddings stored in your ChromaDB database.
Use this to check:
- How many embeddings are stored
- What types of segments (audio/video) are present  
- Sample content from stored segments
- Search functionality testing
"""

import sys
from pathlib import Path
import json

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.phase3_db.client import VectorStoreClient
    from src.phase3_db.retriever import VectorRetriever
    from src.phase3_db.models import VideoSegment
except ImportError as e:
    print(f"❌ Failed to import ChromaDB components: {e}")
    print("💡 Make sure you're running this from the project root with venv activated")
    sys.exit(1)


def inspect_chromadb_database(persist_directory: str = "data/chroma"):
    """Inspect the ChromaDB database and show what's stored."""
    
    print("🔍 ChromaDB Embeddings Verification")
    print("=" * 60)
    
    try:
        # Initialize client
        print("🔹 Step 1: Initializing ChromaDB Client")
        client = VectorStoreClient(persist_directory=persist_directory)
        
        # Get collection info
        print("\n🔹 Step 2: Collection Information")
        collection_info = client.get_collection_info()
        print(f"✅ Collection Name: {collection_info['name']}")
        print(f"✅ Total Segments: {collection_info['count']}")
        print(f"✅ Metadata: {collection_info.get('metadata', {})}")
        
        if collection_info['count'] == 0:
            print("⚠️  No embeddings found in database!")
            return False
        
        # Get segments for inspection
        print("\n🔹 Step 3: Sample Segments Inspection")
        retriever = VectorRetriever(vector_client=client)
        
        # Get collection stats
        stats = retriever.get_collection_stats()
        print(f"✅ Audio segments: {stats['audio_segments']}")
        print(f"✅ Frame segments: {stats['frame_segments']}")
        print(f"✅ Total videos: {stats['total_videos']}")
        print(f"✅ Video IDs: {stats['video_ids']}")
        
        # Show sample segments from each type
        if stats['video_ids']:
            video_id = stats['video_ids'][0]
            print(f"\n🔹 Step 4: Sample Content for Video '{video_id}'")
            
            video_segments = retriever.get_segments_for_video(video_id)
            
            # Group by modality
            audio_segments = [s for s in video_segments if s.metadata.modality == "audio"]
            frame_segments = [s for s in video_segments if s.metadata.modality == "frame"]
            
            print(f"📢 Audio Segments Found: {len(audio_segments)}")
            if audio_segments:
                sample_audio = audio_segments[0]
                print(f"   Sample Audio Content: {sample_audio.content[:100]}...")
                print(f"   Time Range: {sample_audio.metadata.start:.1f}s - {sample_audio.metadata.end:.1f}s")
                print(f"   Word Count: {sample_audio.metadata.word_count}")
                print(f"   Embedding Dimension: {len(sample_audio.embedding)}")
            
            print(f"\n🖼️  Frame Segments Found: {len(frame_segments)}")
            if frame_segments:
                sample_frame = frame_segments[0]
                print(f"   Sample Frame Path: {sample_frame.metadata.path}")
                print(f"   Time: {sample_frame.metadata.start:.1f}s")
                print(f"   Embedding Dimension: {len(sample_frame.embedding)}")
        
        # Test search functionality
        print("\n🔹 Step 5: Search Functionality Test")
        test_queries = ["machine learning", "neural network", "technology", "tutorial"]
        
        working_searches = 0
        for query in test_queries:
            try:
                response = retriever.search_by_text(query, k=3)
                if response.total_found > 0:
                    working_searches += 1
                    best_match = response.results[0]
                    print(f"✅ Query '{query}': {response.total_found} results (best score: {best_match.similarity_score:.3f})")
                else:
                    print(f"⚠️  Query '{query}': No results")
            except Exception as e:
                print(f"❌ Query '{query}' failed: {e}")
        
        print(f"\n📊 Search Summary: {working_searches}/{len(test_queries)} queries successful")
        
        # Summary
        print("\n🔹 Step 6: Verification Summary")
        total_expected = 8 + 6  # From pipeline report: 8 audio + 6 frame segments
        total_actual = collection_info['count']
        
        print(f"✅ Expected segments: {total_expected}")
        print(f"✅ Actual segments in DB: {total_actual}")
        print(f"✅ Data integrity: {'✅ PASS' if total_actual == total_expected else '❌ MISMATCH'}")
        print(f"✅ Search functionality: {'✅ WORKING' if working_searches > 0 else '❌ BROKEN'}")
        
        success = (total_actual > 0 and working_searches > 0)
        
        if success:
            print(f"\n🎉 VERIFICATION SUCCESS!")
            print(f"✅ ChromaDB is working properly")
            print(f"✅ Embeddings are stored and searchable")
            print(f"✅ Video RAG system is ready for use")
        else:
            print(f"\n⚠️  VERIFICATION ISSUES DETECTED")
            print(f"❌ Check the issues above and re-run the pipeline")
        
        return success
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def search_embeddings(query: str, k: int = 5, persist_directory: str = "data/chroma"):
    """Search embeddings by text query."""
    
    print(f"🔍 Searching for: '{query}'")
    print("=" * 50)
    
    try:
        client = VectorStoreClient(persist_directory=persist_directory)
        retriever = VectorRetriever(vector_client=client)
        response = retriever.search_by_text(query, k=k)
        
        if response.total_found == 0:
            print("❌ No results found")
            return
        
        print(f"✅ Found {response.total_found} results:")
        print()
        
        for i, result in enumerate(response.results, 1):
            print(f"📍 Result {i} (Score: {result.similarity_score:.3f})")
            print(f"   Type: {result.metadata.modality}")
            print(f"   Time: {result.get_timing_info()}")
            if result.content:
                content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
                print(f"   Content: {content_preview}")
            if hasattr(result.metadata, 'path') and result.metadata.path:
                print(f"   Path: {result.metadata.path}")
            print()
            
    except Exception as e:
        print(f"❌ Search failed: {e}")


def export_embeddings_summary(output_file: str = "embeddings_summary.json", persist_directory: str = "data/chroma"):
    """Export a summary of all embeddings to a JSON file."""
    
    print(f"📤 Exporting embeddings summary to: {output_file}")
    
    try:
        client = VectorStoreClient(persist_directory=persist_directory)
        retriever = VectorRetriever(vector_client=client)
        stats = retriever.get_collection_stats()
        
        summary = {
            "collection_stats": stats,
            "segments_by_video": {},
            "sample_content": {}
        }
        
        for video_id in stats['video_ids']:
            segments = retriever.get_segments_for_video(video_id)
            
            video_summary = {
                "total_segments": len(segments),
                "audio_segments": len([s for s in segments if s.metadata.modality == "audio"]),
                "frame_segments": len([s for s in segments if s.metadata.modality == "frame"]),
                "duration": max(s.metadata.end for s in segments) if segments else 0
            }
            
            summary["segments_by_video"][video_id] = video_summary
            
            # Add sample content
            if segments:
                sample_audio = next((s for s in segments if s.metadata.modality == "audio"), None)
                sample_frame = next((s for s in segments if s.metadata.modality == "frame"), None)
                
                samples = {}
                if sample_audio:
                    samples["audio_sample"] = {
                        "content": sample_audio.content[:200] if sample_audio.content else None,
                        "time_range": f"{sample_audio.metadata.start:.1f}s - {sample_audio.metadata.end:.1f}s",
                        "embedding_dimension": len(sample_audio.embedding)
                    }
                
                if sample_frame:
                    samples["frame_sample"] = {
                        "path": sample_frame.metadata.path,
                        "time": f"{sample_frame.metadata.start:.1f}s",
                        "embedding_dimension": len(sample_frame.embedding)
                    }
                
                summary["sample_content"][video_id] = samples
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Summary exported to: {output_file}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")


def main():
    """Main function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ChromaDB Embeddings Verification Tool")
    parser.add_argument("--action", choices=["inspect", "search", "export"], default="inspect",
                        help="Action to perform")
    parser.add_argument("--query", type=str, help="Search query (for search action)")
    parser.add_argument("--k", type=int, default=5, help="Number of search results")
    parser.add_argument("--output", type=str, default="embeddings_summary.json", 
                        help="Output file for export action")
    parser.add_argument("--persist-dir", type=str, default="data/chroma",
                        help="ChromaDB persist directory")
    
    args = parser.parse_args()
    
    if args.action == "inspect":
        inspect_chromadb_database(args.persist_dir)
    elif args.action == "search":
        if not args.query:
            print("❌ Search query required. Use --query 'your search text'")
            return
        search_embeddings(args.query, args.k, args.persist_dir)
    elif args.action == "export":
        export_embeddings_summary(args.output, args.persist_dir)


if __name__ == "__main__":
    main() 