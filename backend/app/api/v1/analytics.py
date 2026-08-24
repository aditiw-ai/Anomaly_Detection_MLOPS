"""
Analytics API Endpoints
Aggregates time-series data and system health metrics.
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/dashboard")
async def get_dashboard_analytics(
    days: int = Query(7, ge=1, le=30, description="Days of history to fetch"),
    model_id: Optional[str] = Query(None, description="Optional model ID to filter traffic data"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get aggregated analytics for dashboard charts.
    """
    service = AnalyticsService(db)
    
    health = await service.get_system_health()
    alerts_dist = await service.get_alerts_distribution()
    traffic = await service.get_inference_traffic(days=days, model_id=model_id)
    
    return {
        "data": {
            "health": health,
            "alerts_distribution": alerts_dist,
            "traffic": traffic,
            "period_days": days
        }
    }
