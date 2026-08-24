import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

async def test():
    try:
        from app.core.config import settings
        url = settings.DATABASE_URL
        print("URL is:", url)
        engine = create_async_engine(
            url,
            pool_size=10,
            max_overflow=5,
            pool_pre_ping=True,
            echo=False,
        )
        async with engine.begin() as conn:
            from sqlalchemy import text
            result = await conn.execute(text("SELECT 1"))
            print("SQLAlchemy connection successful:", result.fetchone())
    except Exception as e:
        print(f"SQLAlchemy connection failed: {e}")

asyncio.run(test())
