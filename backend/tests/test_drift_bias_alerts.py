import asyncio
import os
import sys
import time

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.services.alert_service import AlertService, AlertCreate
from app.models.alert import AlertType, AlertSeverity

async def test_monitoring_alerts():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set")
        return

    engine = create_async_engine(DATABASE_URL)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as db:
        svc = AlertService(db)
        
        # Unique Model ID to bypass deduplication logic
        unique_model_id = f"test-monitoring-{int(time.time())}"
        
        print("🚀 Triggering DRIFT alert...")
        drift_alert = await svc.create_drift_alert(
            model_id=unique_model_id,
            feature="transaction_amount",
            psi=0.35,  # Above 0.25 is Critical
            threshold=0.25
        )
        print(f"✅ Drift alert created: ID = {drift_alert.id}")
        
        # Wait a moment to avoid Slack rate limits
        await asyncio.sleep(2)
        
        print("\n🚀 Triggering BIAS (Fairness) alert...")
        # Since there's no pre-defined create_bias_alert in the service right now, we use the base method
        bias_alert = await svc.create_alert(AlertCreate(
            model_id=unique_model_id,
            alert_type=AlertType.BIAS,
            severity=AlertSeverity.WARNING,
            title="Bias Detected: age_group",
            message="Protected attribute 'age_group' shows WARNING bias — Disparate Impact=0.742, DP Diff=0.180",
            details={
                "attribute": "age_group", 
                "disparate_impact": 0.742, 
                "demographic_parity_diff": 0.180
            },
        ))
        print(f"✅ Bias alert created: ID = {bias_alert.id}")
        
        print("\n🎉 Check your Slack channel! You should see two new alerts: 📊 for Drift and ⚖️ for Bias.")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_monitoring_alerts())
