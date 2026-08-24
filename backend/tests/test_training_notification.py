import asyncio
import os
import sys

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.services.alert_service import AlertService

import logging
import time

logging.basicConfig(level=logging.INFO)

async def test_training_notification():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set")
        return

    engine = create_async_engine(DATABASE_URL)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as db:
        svc = AlertService(db)
        print(f"🚀 Triggering mock training success notification...")
        
        # We append a timestamp to the model_id so the system doesn't think 
        # this is spam and ignore it due to the 1-hour deduplication logic.
        unique_model_id = f"test-model-id-{int(time.time())}"
        
        alert = await svc.create_training_success_alert(
            model_id=unique_model_id,
            job_id="test-job-id",
            algorithm="xgboost",
            duration=45.2,
            metrics={"f1": 0.925, "precision": 0.910, "recall": 0.940}
        )
        print(f"✅ Alert created: ID = {alert.id}")
        print("Check your Slack channel for the 🎓 icon!")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_training_notification())
