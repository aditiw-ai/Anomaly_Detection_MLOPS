"""
Scheduled Jobs API
Manage scheduled monitoring jobs.
"""
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_db
from app.workers.scheduler import get_scheduler, JobType, JobStatus

router = APIRouter(prefix="/jobs", tags=["Jobs"])


class JobCreateRequest(BaseModel):
    """Request to create a job."""
    job_type: str
    model_id: Optional[str] = None
    schedule: Optional[str] = None
    config: Optional[dict] = None


class JobResponse(BaseModel):
    """Job response."""
    id: str
    job_type: str
    schedule: str
    model_id: Optional[str]
    enabled: bool
    last_run: Optional[str]
    next_run: str
    status: str


@router.get("")
async def list_jobs(
    job_type: Optional[str] = None,
    model_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all scheduled jobs."""
    scheduler = get_scheduler()
    
    jt = JobType(job_type) if job_type else None
    jobs = scheduler.list_jobs(job_type=jt, model_id=model_id)
    
    return {
        "data": [
            {
                "id": j.id,
                "job_type": j.job_type.value,
                "schedule": j.schedule,
                "model_id": j.model_id,
                "enabled": j.enabled,
                "last_run": j.last_run.isoformat() if j.last_run else None,
                "next_run": j.next_run.isoformat(),
                "status": j.status.value,
                "config": j.config,
            }
            for j in jobs
        ],
        "meta": {
            "total": len(jobs),
        }
    }


@router.post("")
async def create_job(
    request: JobCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new scheduled job."""
    scheduler = get_scheduler()
    
    try:
        job_type = JobType(request.job_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid job type. Valid types: {[t.value for t in JobType]}"
        )
    
    job = scheduler.create_job(
        job_type=job_type,
        model_id=request.model_id,
        schedule=request.schedule,
        config=request.config,
    )
    
    return {
        "data": {
            "id": job.id,
            "job_type": job.job_type.value,
            "schedule": job.schedule,
            "model_id": job.model_id,
            "enabled": job.enabled,
            "next_run": job.next_run.isoformat(),
            "status": job.status.value,
        },
        "message": "Job created"
    }


@router.get("/{job_id}")
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get job details."""
    scheduler = get_scheduler()
    jobs = scheduler.list_jobs()
    
    job = next((j for j in jobs if j.id == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "data": {
            "id": job.id,
            "job_type": job.job_type.value,
            "schedule": job.schedule,
            "model_id": job.model_id,
            "enabled": job.enabled,
            "last_run": job.last_run.isoformat() if job.last_run else None,
            "next_run": job.next_run.isoformat(),
            "status": job.status.value,
            "config": job.config,
        }
    }


@router.post("/{job_id}/run")
async def run_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Manually trigger a job run."""
    scheduler = get_scheduler()
    
    try:
        run = await scheduler.run_job(job_id)
        return {
            "data": {
                "run_id": run.id,
                "job_id": run.job_id,
                "status": run.status.value,
                "started_at": run.started_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "result": run.result,
                "error": run.error,
            },
            "message": "Job executed"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{job_id}/enable")
async def enable_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Enable a job."""
    scheduler = get_scheduler()
    
    if scheduler.enable_job(job_id):
        return {"message": "Job enabled"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")


@router.post("/{job_id}/disable")
async def disable_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Disable a job."""
    scheduler = get_scheduler()
    
    if scheduler.disable_job(job_id):
        return {"message": "Job disabled"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a job."""
    scheduler = get_scheduler()
    
    if scheduler.delete_job(job_id):
        return {"message": "Job deleted"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")


@router.get("/{job_id}/runs")
async def get_job_runs(
    job_id: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """Get job run history."""
    scheduler = get_scheduler()
    runs = scheduler.get_job_runs(job_id=job_id, limit=limit)
    
    return {
        "data": [
            {
                "id": r.id,
                "job_id": r.job_id,
                "job_type": r.job_type.value,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "status": r.status.value,
                "result": r.result,
                "error": r.error,
            }
            for r in runs
        ]
    }


@router.get("/types/available")
async def get_job_types():
    """Get available job types."""
    return {
        "data": [
            {
                "type": jt.value,
                "description": {
                    JobType.DRIFT_CHECK: "Check for data drift in model features",
                    JobType.BIAS_CHECK: "Check for bias across protected attributes",
                    JobType.PERFORMANCE_CHECK: "Check model performance against baselines",
                    JobType.MODEL_RETRAIN: "Trigger model retraining",
                    JobType.DATA_CLEANUP: "Clean up old data and artifacts",
                }.get(jt, ""),
            }
            for jt in JobType
        ]
    }
