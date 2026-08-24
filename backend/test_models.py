from app.core.database import SessionLocal
from app.services.training_service import ModelService
import asyncio

async def run():
    db = SessionLocal()
    s = ModelService(db)
    res = await s.list_models()
    print(res)

asyncio.run(run())
