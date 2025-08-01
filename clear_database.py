#!/usr/bin/env python3
"""Clear the corrupted database and rebuild from scratch."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from phase3_db.client import VectorStoreClient
    
    print("🗑️  Clearing corrupted database...")
    client = VectorStoreClient()
    
    # Get current stats
    info = client.get_collection_info()
    print(f"📊 Current vectors: {info['count']}")
    
    # Clear everything
    result = client.clear_collection()
    
    if result['success']:
        print("✅ Database cleared successfully!")
        
        # Confirm it's empty
        new_info = client.get_collection_info()
        print(f"📊 Vectors after clearing: {new_info['count']}")
    else:
        print(f"❌ Failed to clear database: {result['message']}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()