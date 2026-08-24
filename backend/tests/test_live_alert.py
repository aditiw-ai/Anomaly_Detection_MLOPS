import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, "/app")

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.models import *  # ensure all models are registered
from app.services.alert_service import AlertService, AlertCreate
from app.models.alert import AlertType, AlertSeverity

async def main():
    DATABASE_URL = os.environ["DATABASE_URL"]
    engine = create_async_engine(DATABASE_URL)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as db:
        svc = AlertService(db)
        # We add a timestamp to the title so it is mathematically impossible to be caught by the 1-hour deduplicator
        unique_title = f"🚨 [LIVE NOTIFICATION] Real-Time Alert {datetime.now().strftime('%H:%M:%S')} 🚨"
        
        alert = await svc.create_alert(AlertCreate(
            model_id="live-demo-model-001",
            alert_type=AlertType.SYSTEM,
            severity=AlertSeverity.CRITICAL,
            title=unique_title,
            message="This is a real notification sent directly from the Fraud MLOps platform backend to prove your Slack webhook is 100% fully operational right now!",
            details={"triggered_by": "manual_user_demo", "timestamp": datetime.now().isoformat()}
        ))
        print(f"✅ Alert successfully injected into DB! ID: {alert.id}")
        print(f"✅ Unique Title Used: {unique_title}")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
