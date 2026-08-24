import os
import io
import sys
import asyncio
import traceback

sys.path.insert(0, '/app')

try:
    from app.core.storage import storage_service
    import pandas as pd
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

async def verify_new_dataset():
    storage_path = "datasets/raw/raw_20260406T102712Z_f4f57855/data.csv"
    try:
        content = await storage_service.download_dataset(storage_path)
    except Exception as e:
        print(f"Failed to download blob: {e}")
        return
        
    raw_lines = content.count(b'\n')
    print(f"Exact physical newline characters in the file: {raw_lines}")
    print(f"Total file size in bytes: {len(content)}")
    
    try:
        df = pd.read_csv(io.BytesIO(content))
        print(f"Pandas parsed row count: {len(df)}")
        print(f"Pandas parsed column count: {len(df.columns)}")
        print("\nFirst 3 rows:")
        print(df.head(3))
    except Exception as e:
        print(f"Pandas parsing failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_new_dataset())
