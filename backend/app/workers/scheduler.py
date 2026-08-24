"""
Job Scheduler
Manages scheduled monitoring and maintenance tasks.
"""
from app.core.time import IST, now_ist
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import json
import os

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobType(str, Enum):
    DRIFT_CHECK = "DRIFT_CHECK"
    BIAS_CHECK = "BIAS_CHECK"
    PERFORMANCE_CHECK = "PERFORMANCE_CHECK"
    MODEL_RETRAIN = "MODEL_RETRAIN"
    DATA_CLEANUP = "DATA_CLEANUP"


@dataclass
class ScheduledJob:
    """Represents a scheduled job."""
    id: str
    job_type: JobType
    schedule: str  # Cron expression or interval
    model_id: Optional[str]
    enabled: bool
    last_run: Optional[datetime]
    next_run: datetime
    status: JobStatus
    config: Dict[str, Any]


@dataclass
class JobRun:
    """Record of a job execution."""
    id: str
    job_id: str
    job_type: JobType
    started_at: datetime
    completed_at: Optional[datetime]
    status: JobStatus
    result: Optional[Dict]
    error: Optional[str]


class JobScheduler:
    """
    Manages scheduled monitoring jobs.
    
    Default scheduled jobs:
    - Drift check: Every hour
    - Bias check: Every 6 hours
    - Performance check: Every hour
    """
    
    # Default schedules
    DEFAULT_SCHEDULES = {
        JobType.DRIFT_CHECK: "0 * * * *",      # Every hour
        JobType.BIAS_CHECK: "0 */6 * * *",     # Every 6 hours
        JobType.PERFORMANCE_CHECK: "30 * * * *", # Every hour at :30
    }
    _STATE_KEY = "scheduler:state:v1"
    
    def __init__(self):
        self._jobs: Dict[str, ScheduledJob] = {}
        self._runs: Dict[str, JobRun] = {}
        self._handlers: Dict[JobType, Callable] = {}
        
        # Register default handlers
        self._register_default_handlers()
        self._load_state()

    def _get_redis_client(self):
        """Build a short-timeout Redis client for state snapshots."""
        try:
            import redis
            url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            return redis.from_url(url, decode_responses=True, socket_timeout=2)
        except Exception as exc:
            logger.warning(f"Scheduler state Redis init failed: {exc}")
            return None

    def _serialize_job(self, job: ScheduledJob) -> Dict[str, Any]:
        return {
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

    def _deserialize_job(self, data: Dict[str, Any]) -> ScheduledJob:
        return ScheduledJob(
            id=data["id"],
            job_type=JobType(data["job_type"]),
            schedule=data.get("schedule", "0 * * * *"),
            model_id=data.get("model_id"),
            enabled=bool(data.get("enabled", True)),
            last_run=datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None,
            next_run=datetime.fromisoformat(data["next_run"]) if data.get("next_run") else now_ist() + timedelta(hours=1),
            status=JobStatus(data.get("status", JobStatus.PENDING.value)),
            config=data.get("config") or {},
        )

    def _serialize_run(self, run: JobRun) -> Dict[str, Any]:
        return {
            "id": run.id,
            "job_id": run.job_id,
            "job_type": run.job_type.value,
            "started_at": run.started_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "status": run.status.value,
            "result": run.result,
            "error": run.error,
        }

    def _deserialize_run(self, data: Dict[str, Any]) -> JobRun:
        return JobRun(
            id=data["id"],
            job_id=data["job_id"],
            job_type=JobType(data["job_type"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            status=JobStatus(data.get("status", JobStatus.PENDING.value)),
            result=data.get("result"),
            error=data.get("error"),
        )

    def _persist_state(self):
        """Persist in-memory scheduler state to Redis."""
        client = self._get_redis_client()
        if not client:
            return
        payload = {
            "jobs": [self._serialize_job(j) for j in self._jobs.values()],
            "runs": [self._serialize_run(r) for r in self._runs.values()],
        }
        try:
            client.set(self._STATE_KEY, json.dumps(payload, default=str))
        except Exception as exc:
            logger.warning(f"Scheduler state persist failed: {exc}")

    def _load_state(self):
        """Load scheduler state snapshot from Redis if available."""
        client = self._get_redis_client()
        if not client:
            return
        try:
            raw = client.get(self._STATE_KEY)
            if not raw:
                return
            payload = json.loads(raw)
            self._jobs = {}
            for j in payload.get("jobs", []):
                try:
                    job = self._deserialize_job(j)
                    self._jobs[job.id] = job
                except Exception as exc:
                    logger.warning(f"Skipping invalid persisted scheduler job: {exc}")

            self._runs = {}
            for r in payload.get("runs", []):
                try:
                    run = self._deserialize_run(r)
                    self._runs[run.id] = run
                except Exception as exc:
                    logger.warning(f"Skipping invalid persisted scheduler run: {exc}")
        except Exception as exc:
            logger.warning(f"Scheduler state load failed: {exc}")
    
    def _register_default_handlers(self):
        """Register default job handlers."""
        self._handlers[JobType.DRIFT_CHECK] = self._run_drift_check
        self._handlers[JobType.BIAS_CHECK] = self._run_bias_check
        self._handlers[JobType.PERFORMANCE_CHECK] = self._run_performance_check
    
    async def _run_drift_check(self, job: ScheduledJob) -> Dict:
        """Dispatch drift check Celery task."""
        logger.info(f"Dispatching drift check for model {job.model_id}")
        from app.workers.monitoring_worker import compute_drift_metrics
        task = compute_drift_metrics.delay(model_id=job.model_id)
        return {
            "status": "dispatched",
            "task_id": task.id,
            "model_id": job.model_id,
        }
    
    async def _run_bias_check(self, job: ScheduledJob) -> Dict:
        """Dispatch bias check Celery task."""
        logger.info(f"Dispatching bias check for model {job.model_id}")
        from app.workers.monitoring_worker import compute_bias_metrics
        task = compute_bias_metrics.delay(model_id=job.model_id)
        return {
            "status": "dispatched",
            "task_id": task.id,
            "model_id": job.model_id,
        }
    
    async def _run_performance_check(self, job: ScheduledJob) -> Dict:
        """Dispatch performance baseline check Celery task."""
        logger.info(f"Dispatching performance check for model {job.model_id}")
        from app.workers.monitoring_worker import check_performance_baselines
        if not job.model_id:
            raise ValueError("PERFORMANCE_CHECK requires a model_id")
        task = check_performance_baselines.delay(model_id=job.model_id)
        return {
            "status": "dispatched",
            "task_id": task.id,
            "model_id": job.model_id,
        }
    
    def create_job(
        self,
        job_type: JobType,
        model_id: Optional[str] = None,
        schedule: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> ScheduledJob:
        """Create a new scheduled job."""
        from uuid import uuid4
        
        job = ScheduledJob(
            id=str(uuid4()),
            job_type=job_type,
            schedule=schedule or self.DEFAULT_SCHEDULES.get(job_type, "0 * * * *"),
            model_id=model_id,
            enabled=True,
            last_run=None,
            next_run=now_ist() + timedelta(hours=1),
            status=JobStatus.PENDING,
            config=config or {},
        )
        
        self._jobs[job.id] = job
        self._persist_state()
        logger.info(f"Created job {job.id}: {job_type.value}")
        
        return job
    
    async def run_job(self, job_id: str) -> JobRun:
        """Manually trigger a job run."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        from uuid import uuid4
        
        run = JobRun(
            id=str(uuid4()),
            job_id=job_id,
            job_type=job.job_type,
            started_at=now_ist(),
            completed_at=None,
            status=JobStatus.RUNNING,
            result=None,
            error=None,
        )
        
        self._runs[run.id] = run
        job.status = JobStatus.RUNNING
        self._persist_state()
        
        try:
            handler = self._handlers.get(job.job_type)
            if handler:
                result = await handler(job)
                run.result = result
                run.status = JobStatus.COMPLETED
                job.status = JobStatus.COMPLETED
            else:
                raise ValueError(f"No handler for job type {job.job_type}")
                
        except Exception as e:
            run.error = str(e)
            run.status = JobStatus.FAILED
            job.status = JobStatus.FAILED
            logger.error(f"Job {job_id} failed: {e}")
        
        run.completed_at = now_ist()
        job.last_run = run.completed_at
        
        # Schedule next run
        job.next_run = self._calculate_next_run(job.schedule)
        self._persist_state()
        
        return run
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate next run time from cron expression."""
        # Simplified - just add 1 hour
        # In production, use croniter library
        return now_ist() + timedelta(hours=1)
    
    def list_jobs(
        self,
        job_type: Optional[JobType] = None,
        model_id: Optional[str] = None,
    ) -> List[ScheduledJob]:
        """List scheduled jobs."""
        jobs = list(self._jobs.values())
        
        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        
        if model_id:
            jobs = [j for j in jobs if j.model_id == model_id]
        
        return jobs
    
    def get_job_runs(
        self,
        job_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[JobRun]:
        """Get job run history."""
        runs = list(self._runs.values())
        
        if job_id:
            runs = [r for r in runs if r.job_id == job_id]
        
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs[:limit]
    
    def enable_job(self, job_id: str) -> bool:
        """Enable a job."""
        job = self._jobs.get(job_id)
        if job:
            job.enabled = True
            self._persist_state()
            return True
        return False
    
    def disable_job(self, job_id: str) -> bool:
        """Disable a job."""
        job = self._jobs.get(job_id)
        if job:
            job.enabled = False
            self._persist_state()
            return True
        return False
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._persist_state()
            return True
        return False


# Singleton scheduler instance
_scheduler: Optional[JobScheduler] = None


def get_scheduler() -> JobScheduler:
    """Get the global job scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = JobScheduler()
    return _scheduler
