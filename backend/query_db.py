import asyncio
import sys

async def main():
    try:
        from app.core.database import async_engine
        from sqlalchemy import text
        from app.core.database import SessionLocal
        
        async with async_engine.connect() as conn:
            print("--- DATASETS ---")
            result = await conn.execute(text("SELECT id, name FROM datasets ORDER BY created_at DESC LIMIT 5;"))
            for row in result:
                print(row)
                
            print("\n--- TRAINING JOBS ---")
            result = await conn.execute(text("SELECT id, name, metrics->>'train_dataset_id' as tdi FROM training_jobs ORDER BY created_at DESC LIMIT 3;"))
            for row in result:
                print(row)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
