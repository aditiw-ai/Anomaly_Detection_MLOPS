"""
Direct test: creates a CRITICAL alert via AlertService and fires Slack notification.
Run inside the backend container: docker compose exec backend python test_alert.py
"""
import asyncio
import os
import sys

sys.path.insert(0, "/app")

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.models import *  # ensure all models are registered
from app.services.alert_service import AlertService, AlertCreate
from app.models.alert import AlertType, AlertSeverity, Alert

async def main():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        print("DATABASE_URL is not set; skipping manual alert test.")
        return

    engine = create_async_engine(DATABASE_URL)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "NOT SET")
    print(f"SLACK_WEBHOOK_URL in env: {'✅ SET' if 'hooks.slack.com' in SLACK_WEBHOOK_URL else '❌ NOT SET'}")

    async with Session() as db:
        svc = AlertService(db)
        alert = await svc.create_alert(AlertCreate(
            model_id="model-slack-test",
            alert_type=AlertType.SYSTEM,
            severity=AlertSeverity.CRITICAL,
            title="[TEST] Real-Time Slack Alert",
            message="This is a real notification sent from the Fraud MLOps platform. Slack integration is working!",
            details={"triggered_by": "manual_test", "source": "alert_service"}
        ))
        print(f"Alert created: ID = {alert.id}")
        print("✅ Slack notification should now appear in your channel!")

    await engine.dispose()

asyncio.run(main())