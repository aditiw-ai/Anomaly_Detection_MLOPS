import os
import io
import sys
import asyncio
import pandas as pd

sys.path.insert(0, '/app')

from app.core.storage import storage_service

async def verify_dataset():
    storage_path = "datasets/raw/raw_20260403T173915Z_859c2348/data.csv"
    try:
        content = await storage_service.download_dataset(storage_path)
    except Exception as e:
        print(f"Failed to download blob: {e}")
        return
        
    raw_lines = content.count(b'\n')
    print(f"Raw newline count (file lines): {raw_lines}")
    print(f"File size: {len(content)} bytes")
    
    # Try different pandas parsing
    try:
        df1 = pd.read_csv(io.BytesIO(content))
        print(f"Pandas default row count: {len(df1)}")
    except Exception as e:
        print(f"Pandas fail: {e}")

if __name__ == "__main__":
    asyncio.run(verify_dataset())
