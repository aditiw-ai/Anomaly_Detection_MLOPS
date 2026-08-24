"""
Analytics Service
Aggregates time-series data and system health metrics for the dashboard.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from sqlalchemy import select, func, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.time import now_ist
from app.models.alert import Alert, AlertStatus
from app.models.inference_log import InferenceLog
from app.models.training_job import TrainingJob
from app.models.ml_model import MLModel

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Service for computing dashboard analytics and time-series aggregations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_alerts_distribution(self) -> Dict[str, Any]:
        """
        Group active alerts by severity and type.
        """
        # By Severity
        severity_query = select(
            Alert.severity, 
            func.count(Alert.id).label("count")
        ).where(
            Alert.status == AlertStatus.ACTIVE
        ).group_by(Alert.severity)
        
        sev_result = await self.db.execute(severity_query)
        by_severity = {row.severity.value: row.count for row in sev_result.all()}
        
        # By Type
        type_query = select(
            Alert.alert_type, 
            func.count(Alert.id).label("count")
        ).where(
            Alert.status == AlertStatus.ACTIVE
        ).group_by(Alert.alert_type)
        
        type_result = await self.db.execute(type_query)
        by_type = {row.alert_type.value: row.count for row in type_result.all()}
        
        return {
            "by_severity": by_severity,
            "by_type": by_type
        }

    async def get_inference_traffic(self, days: int = 7, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get daily inference request counts for the last `days` days, optionally filtered by model_id.
        """
        cutoff_date = (now_ist() - timedelta(days=days)).replace(tzinfo=None)
        
        # PostgreSQL specific date truncation
        query = select(
            func.date_trunc(text("'day'"), InferenceLog.created_at).label("day"),
            func.count(InferenceLog.id).label("request_count"),
            func.avg(InferenceLog.response_time_ms).label("avg_latency_ms")
        )
        
        if model_id:
            query = query.where(InferenceLog.model_id == model_id)
            
        query = query.where(
            InferenceLog.created_at >= cutoff_date
        ).group_by(
            func.date_trunc(text("'day'"), InferenceLog.created_at)
        ).order_by("day")
        
        result = await self.db.execute(query)
        
        traffic_data = []
        for row in result.all():
            if row.day:
                traffic_data.append({
                    "date": row.day.strftime("%Y-%m-%d"),
                    "request_count": row.request_count,
                    "avg_latency_ms": round(row.avg_latency_ms or 0, 2)
                })
                
        return traffic_data

    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health metrics.
        """
        # Average global inference latency over the last 24 hours
        cutoff = (now_ist() - timedelta(hours=24)).replace(tzinfo=None)
        
        latency_query = select(func.avg(InferenceLog.response_time_ms)).where(InferenceLog.created_at >= cutoff)
        avg_latency = await self.db.scalar(latency_query)
        
        # Active Models 
        active_models = await self.db.scalar(
            select(func.count(MLModel.id)).where(MLModel.status == "PRODUCTION")
        )
        
        # Failed Training Jobs (Last 7 days)
        failed_jobs_query = select(func.count(TrainingJob.id)).where(
            TrainingJob.status == "FAILED",
            TrainingJob.created_at >= (now_ist() - timedelta(days=7)).replace(tzinfo=None)
        )
        failed_jobs = await self.db.scalar(failed_jobs_query)
        
        # Determine status
        status = "HEALTHY"
        if failed_jobs and failed_jobs > 5:
            status = "DEGRADED"
        if avg_latency and avg_latency > 500:
            status = "DEGRADED"
            
        return {
            "status": status,
            "avg_inference_latency_ms": round(avg_latency or 0, 2),
            "production_models_online": active_models or 0,
            "recent_training_failures": failed_jobs or 0,
            "last_updated": now_ist().isoformat()
        }
