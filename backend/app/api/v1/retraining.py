"""
Retraining API Endpoints
Manage automated model retraining.
"""
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_db
from app.core.dependencies import require_auth, can_train_models
from app.core.auth import User
from app.services.retraining_service import (
    get_retraining_pipeline,
    RetrainReason,
    RetrainStatus,
    RetrainConfig,
)

router = APIRouter(prefix="/retraining", tags=["Retraining"])


class TriggerRetrainRequest(BaseModel):
    """Request to trigger retraining."""
    model_id: str
    reason: str
    algorithm: str = "xgboost"
    use_latest_data: bool = True
    data_window_days: int = 90
    hyperparameter_tuning: bool = True
    fairness_constraint: bool = True
    auto_promote: bool = False


class RetrainJobResponse(BaseModel):
    """Retraining job response."""
    id: str
    model_id: str
    reason: str
    status: str
    current_step: str
    progress: float
    started_at: str
    completed_at: Optional[str] = None
    new_model_id: Optional[str] = None


@router.post("/trigger")
async def trigger_retraining(
    request: TriggerRetrainRequest,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(can_train_models),
):
    """Trigger model retraining."""
    pipeline = get_retraining_pipeline()
    
    # Normalize caller input, but do not silently downgrade to MANUAL.
    normalized_reason = request.reason.strip().upper()
    if normalized_reason != RetrainReason.MANUAL.value:
        raise HTTPException(
            status_code=422,
            detail="Retraining is manual-only. Use reason 'MANUAL'.",
        )
    try:
        reason = RetrainReason(normalized_reason)
    except ValueError:
        allowed = ", ".join([r.value for r in RetrainReason])
        raise HTTPException(
            status_code=422,
            detail=f"Invalid retraining reason '{request.reason}'. Allowed: {allowed}",
        )
    
    config = RetrainConfig(
        algorithm=request.algorithm,
        use_latest_data=request.use_latest_data,
        data_window_days=request.data_window_days,
        hyperparameter_tuning=request.hyperparameter_tuning,
        fairness_constraint=request.fairness_constraint,
        auto_promote=request.auto_promote,
    )
    
    job = pipeline.trigger_retraining(
        model_id=request.model_id,
        reason=reason,
        config=config,
    )
    
    return {
        "data": {
            "id": job.id,
            "model_id": job.model_id,
            "reason": job.reason.value,
            "status": job.status.value,
            "current_step": job.current_step,
            "progress": job.progress,
            "started_at": job.started_at.isoformat(),
        },
        "message": "Retraining triggered"
    }


@router.post("/{job_id}/run")
async def run_retraining(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(can_train_models),
):
    """Execute retraining pipeline."""
    pipeline = get_retraining_pipeline()
    
    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    active_states = {
        RetrainStatus.DATA_PREPARATION,
        RetrainStatus.TRAINING,
        RetrainStatus.VALIDATION,
        RetrainStatus.COMPARISON,
    }
    terminal_states = {
        RetrainStatus.COMPLETED,
        RetrainStatus.REJECTED,
        RetrainStatus.FAILED,
    }

    if job.status in active_states:
        raise HTTPException(
            status_code=409,
            detail=f"Job already running (status={job.status.value})",
        )

    if job.status in terminal_states:
        raise HTTPException(
            status_code=400,
            detail=f"Job cannot be run from terminal state {job.status.value}",
        )

    if job.status != RetrainStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Job cannot be run from state {job.status.value}",
        )

    if job.current_step == "Queued for background execution":
        raise HTTPException(
            status_code=409,
            detail="Job is already queued for execution",
        )

    # Enqueue background execution in worker so API reload/request lifecycle does not interrupt runs.
    from app.workers.retraining_worker import run_retraining_job
    previous_step = job.current_step
    job.current_step = "Queued for background execution"
    pipeline._persist_state()
    try:
        run_retraining_job.delay(job_id)
    except Exception:
        job.current_step = previous_step
        pipeline._persist_state()
        raise
    
    return {
        "data": {
            "id": job.id,
            "status": job.status.value,
            "current_step": job.current_step,
            "progress": job.progress,
            "metrics": job.metrics,
            "comparison_result": job.comparison_result,
            "new_model_id": job.new_model_id,
            "error": job.error,
        }
    }


@router.get("/{job_id}")
async def get_retraining_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(require_auth),
):
    """Get retraining job details."""
    pipeline = get_retraining_pipeline()
    await pipeline.reconcile_stale_jobs()
    
    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "data": {
            "id": job.id,
            "model_id": job.model_id,
            "reason": job.reason.value,
            "status": job.status.value,
            "current_step": job.current_step,
            "progress": job.progress,
            "started_at": job.started_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "metrics": job.metrics,
            "comparison_result": job.comparison_result,
            "new_model_id": job.new_model_id,
            "error": job.error,
        }
    }


@router.get("")
async def list_retraining_jobs(
    model_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(require_auth),
):
    """List retraining jobs."""
    pipeline = get_retraining_pipeline()
    await pipeline.reconcile_stale_jobs()
    
    st = RetrainStatus(status) if status else None
    jobs = pipeline.list_jobs(model_id=model_id, status=st, limit=limit)
    
    return {
        "data": [
            {
                "id": j.id,
                "model_id": j.model_id,
                "reason": j.reason.value,
                "status": j.status.value,
                "current_step": j.current_step,
                "progress": j.progress,
                "started_at": j.started_at.isoformat(),
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                "new_model_id": j.new_model_id,
            }
            for j in jobs
        ],
        "meta": {
            "total": len(jobs),
        }
    }


@router.post("/{job_id}/promote")
async def promote_model(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(can_train_models),
):
    """Promote retrained model to production."""
    pipeline = get_retraining_pipeline()
    
    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != RetrainStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if not job.new_model_id:
        raise HTTPException(status_code=400, detail="No new model available")
    
    # In production, update model registry to promote new model
    
    return {
        "message": "Model promoted to production",
        "new_model_id": job.new_model_id,
    }


@router.delete("/{job_id}")
async def delete_retraining_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(can_train_models),
):
    """Delete a retraining job."""
    pipeline = get_retraining_pipeline()
    try:
        deleted = pipeline.delete_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"message": "Retraining job deleted", "id": job_id}


@router.get("/reasons/available")
async def get_retrain_reasons(
    _current_user: User = Depends(require_auth),
):
    """Get available retraining reasons."""
    return {
        "data": [
            {
                "reason": RetrainReason.MANUAL.value,
                "description": "Manually triggered",
            }
        ]
    }
