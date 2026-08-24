import asyncio
import os
import sys

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def migrate_enum():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set")
        return

    # Use a sync connection via async engine just for this DDL
    engine = create_async_engine(DATABASE_URL)
    
    try:
        async with engine.connect() as conn:
            print("🚀 Altering enum type 'alerttype' to add 'TRAINING'...")
            # Note: Postgres doesn't allow ALTER TYPE ... ADD VALUE inside a transaction block 
            # if we are not careful, but SQLAlchemy async connect is usually fine here.
            # We use execution_options(isolation_level="AUTOCOMMIT") for DDLs like this.
            await conn.execution_options(isolation_level="AUTOCOMMIT").execute(
                text("ALTER TYPE alerttype ADD VALUE IF NOT EXISTS 'TRAINING'")
            )
            print("✅ Successfully updated alerttype enum!")
    except Exception as e:
        if "already exists" in str(e):
             print("ℹ️  Value 'TRAINING' already exists in alerttype enum.")
        else:
             print(f"❌ Migration failed: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(migrate_enum())
