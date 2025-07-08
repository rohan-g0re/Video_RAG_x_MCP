#!/usr/bin/env python3
"""
Test ChromaDB and Phase 3 components
"""

import sys
from pathlib import Path

print("🧪 Testing ChromaDB Integration")
print("="*50)

# Test 1: Basic ChromaDB import
try:
    import chromadb
    print(f"✅ ChromaDB imported: version {chromadb.__version__}")
except ImportError as e:
    print(f"❌ ChromaDB import failed: {e}")
    sys.exit(1)

# Test 2: Phase 3 components
try:
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from phase3_db.models import VideoSegment, EmbeddingMetadata
    print("✅ Phase 3 models imported")
    
    from phase3_db.client import VectorStoreClient
    print("✅ VectorStoreClient imported")
    
    from phase3_db.ingest import BatchIngestor
    print("✅ BatchIngestor imported")
    
    from phase3_db.retriever import VectorRetriever
    print("✅ VectorRetriever imported")
    
except ImportError as e:
    print(f"❌ Phase 3 import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create a ChromaDB client
try:
    client = VectorStoreClient(persist_directory="data/chroma_test")
    print("✅ VectorStoreClient created successfully")
    
    info = client.get_collection_info()
    print(f"✅ Collection info retrieved: {info}")
    
except Exception as e:
    print(f"❌ ChromaDB client test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All ChromaDB tests passed!")
print("ChromaDB is ready for use with Phase 3 components.") 