import asyncio
import sys

async def main():
    try:
        from app.core.database import engine
        from sqlalchemy import text
        
        async with engine.connect() as conn:
            print("--- FAILED FEATURE SETS ---")
            result = await conn.execute(text("SELECT id, name, status, error_message, created_at FROM feature_sets WHERE status = 'FAILED' ORDER BY created_at DESC LIMIT 3;"))
            for row in result:
                print(row)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
