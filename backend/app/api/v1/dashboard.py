"""
Dashboard Statistics API Endpoints
Provides aggregated statistics for the dashboard.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from uuid import UUID

from app.core.database import get_db
from app.core.dependencies import require_auth
from app.core.auth import User
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.training_job import TrainingJob

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


def _safe_uuid_user_id(user_id: str) -> str | None:
    try:
        return str(UUID(str(user_id)))
    except (ValueError, TypeError):
        return None


@router.get("/stats")
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    Get aggregated statistics for the dashboard.
    
    Returns:
    - total_datasets: Total number of datasets
    - total_training_jobs: Total number of training jobs
    - active_training_jobs: Number of active/running training jobs
    - production_models: Number of models in production
    - active_alerts: Number of active alerts (placeholder for now)
    """
    
    safe_user_id = _safe_uuid_user_id(str(getattr(current_user, "id", "")))

    # Count raw datasets only (parent_id IS NULL) — same filter as the Data Registry page.
    # Merged and split datasets have parent_id set and are excluded, just like the list endpoint.
    datasets_query = select(func.count(Dataset.id)).where(Dataset.parent_id.is_(None))
    if safe_user_id:
        datasets_query = datasets_query.where(Dataset.created_by == safe_user_id)
    datasets_count = await db.scalar(datasets_query)
    
    # Count all training jobs
    total_jobs_query = select(func.count(TrainingJob.id))
    if safe_user_id:
        total_jobs_query = total_jobs_query.where(TrainingJob.created_by == safe_user_id)
    total_jobs = await db.scalar(total_jobs_query)

    # Count active training jobs: QUEUED, RUNNING, or DATA_PREPARED
    active_jobs_query = select(func.count(TrainingJob.id)).where(
        TrainingJob.status.in_(["QUEUED", "RUNNING", "DATA_PREPARED"])
    )
    if safe_user_id:
        active_jobs_query = active_jobs_query.where(TrainingJob.created_by == safe_user_id)
    active_jobs = await db.scalar(active_jobs_query)
    
    # Count production models
    production_models_query = select(func.count(MLModel.id)).where(MLModel.status == "PRODUCTION")
    if safe_user_id:
        production_models_query = production_models_query.where(MLModel.created_by == safe_user_id)
    production_models = await db.scalar(production_models_query)
    
    # Query active alerts
    from app.models.alert import Alert, AlertStatus
    active_alerts_query = select(func.count(Alert.id)).where(Alert.status == AlertStatus.ACTIVE)
    if safe_user_id and hasattr(Alert, "created_by"):
        active_alerts_query = active_alerts_query.where(Alert.created_by == safe_user_id)
    active_alerts = await db.scalar(active_alerts_query)
    
    return {
        "data": {
            "total_datasets": datasets_count or 0,
            "total_training_jobs": total_jobs or 0,
            "active_training_jobs": active_jobs or 0,
            "production_models": production_models or 0,
            "active_alerts": active_alerts or 0,
        }
    }
