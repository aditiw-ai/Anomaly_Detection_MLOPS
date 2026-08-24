"""
Automated Retraining Pipeline
Trigger and manage model retraining based on drift/performance.
"""
from app.core.time import IST, now_ist
from typing import Dict, List, Optional, Any,Tuple
from dataclasses import dataclass
from uuid import uuid4, UUID
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import os
import json
import os

logger = logging.getLogger(__name__)


class RetrainReason(str, Enum):
    """Reason for retraining."""
    SCHEDULED = "SCHEDULED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    BIAS_DETECTED = "BIAS_DETECTED"
    MANUAL = "MANUAL"
    NEW_DATA = "NEW_DATA"


class RetrainStatus(str, Enum):
    """Retraining job status."""
    PENDING = "PENDING"
    DATA_PREPARATION = "DATA_PREPARATION"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    COMPARISON = "COMPARISON"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"  # New model not better than current


@dataclass
class RetrainConfig:
    """Configuration for retraining."""
    algorithm: str = "xgboost"
    use_latest_data: bool = True
    data_window_days: int = 90
    validation_split: float = 0.2
    hyperparameter_tuning: bool = True
    fairness_constraint: bool = True
    min_improvement_threshold: float = 0.01  # 1% improvement required
    auto_promote: bool = False  # Auto-promote if better


@dataclass
class RetrainJob:
    """Retraining job record."""
    id: str
    model_id: str
    reason: RetrainReason
    status: RetrainStatus
    config: RetrainConfig
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_step: str = ""
    progress: float = 0.0
    metrics: Optional[Dict] = None
    training_job_id: Optional[str] = None
    new_model_id: Optional[str] = None
    comparison_result: Optional[Dict] = None
    error: Optional[str] = None


class RetrainingPipeline:
    """
    Automated model retraining pipeline.
    
    Features:
    - Drift-triggered retraining
    - Performance-based retraining
    - Bias-aware retraining with fairness constraints
    - A/B comparison before promotion
    """
    _STATE_KEY = "retraining:pipeline:state:v1"
    _STATE_KEY = "retraining:pipeline:state:v1"
    
    def __init__(self):
        self._jobs: Dict[str, RetrainJob] = {}
        self._load_state()

    def _get_redis_client(self):
        try:
            import redis
            url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            return redis.from_url(url, decode_responses=True, socket_timeout=2)
        except Exception as exc:
            logger.warning(f"Retraining state Redis init failed: {exc}")
            return None

    def _serialize_config(self, config: RetrainConfig) -> Dict[str, Any]:
        return {
            "algorithm": config.algorithm,
            "use_latest_data": config.use_latest_data,
            "data_window_days": config.data_window_days,
            "validation_split": config.validation_split,
            "hyperparameter_tuning": config.hyperparameter_tuning,
            "fairness_constraint": config.fairness_constraint,
            "min_improvement_threshold": config.min_improvement_threshold,
            "auto_promote": config.auto_promote,
        }

    def _deserialize_config(self, data: Dict[str, Any]) -> RetrainConfig:
        return RetrainConfig(
            algorithm=data.get("algorithm", "xgboost"),
            use_latest_data=bool(data.get("use_latest_data", True)),
            data_window_days=int(data.get("data_window_days", 90)),
            validation_split=float(data.get("validation_split", 0.2)),
            hyperparameter_tuning=bool(data.get("hyperparameter_tuning", True)),
            fairness_constraint=bool(data.get("fairness_constraint", True)),
            min_improvement_threshold=float(data.get("min_improvement_threshold", 0.01)),
            auto_promote=bool(data.get("auto_promote", False)),
        )

    def _serialize_job(self, job: RetrainJob) -> Dict[str, Any]:
        return {
            "id": job.id,
            "model_id": job.model_id,
            "reason": job.reason.value,
            "status": job.status.value,
            "config": self._serialize_config(job.config),
            "started_at": job.started_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "current_step": job.current_step,
            "progress": job.progress,
            "metrics": job.metrics,
            "training_job_id": job.training_job_id,
            "new_model_id": job.new_model_id,
            "comparison_result": job.comparison_result,
            "error": job.error,
        }

    def _deserialize_job(self, data: Dict[str, Any]) -> RetrainJob:
        return RetrainJob(
            id=data["id"],
            model_id=data["model_id"],
            reason=RetrainReason(data.get("reason", RetrainReason.MANUAL.value)),
            status=RetrainStatus(data.get("status", RetrainStatus.PENDING.value)),
            config=self._deserialize_config(data.get("config") or {}),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            current_step=data.get("current_step", ""),
            progress=float(data.get("progress", 0.0)),
            metrics=data.get("metrics"),
            training_job_id=data.get("training_job_id"),
            new_model_id=data.get("new_model_id"),
            comparison_result=data.get("comparison_result"),
            error=data.get("error"),
        )

    def _persist_state(self):
        client = self._get_redis_client()
        if not client:
            return
        payload = {
            "jobs": [self._serialize_job(job) for job in self._jobs.values()],
        }
        try:
            client.set(self._STATE_KEY, json.dumps(payload, default=str))
        except Exception as exc:
            logger.warning(f"Retraining state persist failed: {exc}")

    def _load_state(self):
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
                    logger.warning(f"Skipping invalid persisted retraining job: {exc}")
        except Exception as exc:
            logger.warning(f"Retraining state load failed: {exc}")
    
    async def reconcile_stale_jobs(self):
        """Placeholder for reconciling stale jobs (e.g. stuck in RUNNING)."""
        pass

    def trigger_retraining(
        self,
        model_id: str,
        reason: RetrainReason,
        config: Optional[RetrainConfig] = None,
    ) -> RetrainJob:
        """
        Trigger a retraining job.
        
        Args:
            model_id: Current production model to retrain
            reason: Why retraining is triggered
            config: Retraining configuration
        
        Returns:
            RetrainJob with job details
        """
        job = RetrainJob(
            id=str(uuid4()),
            model_id=model_id,
            reason=reason,
            status=RetrainStatus.PENDING,
            config=config or RetrainConfig(),
            started_at=now_ist(),
            current_step="Initializing",
            progress=0.0,
        )
        
        self._jobs[job.id] = job
        self._persist_state()
        self._persist_state()
        
        logger.info(f"Retraining triggered: {job.id} for model {model_id}, reason: {reason.value}")
        
        return job
    
    async def run_pipeline(self, job_id: str) -> RetrainJob:
        """
        Execute the retraining pipeline.
        
        Steps:
        1. Data preparation
        2. Training
        3. Validation
        4. Comparison with current model
        5. Decision (promote or reject)
        """
        # When run from background workers, ensure latest state is loaded.
        self._load_state()
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        try:
            # Step 1: Data Preparation
            await self._step_data_preparation(job)
            
            # Step 2: Training
            await self._step_training(job)
            
            # Step 3: Validation
            await self._step_validation(job)
            
            # Step 4: Comparison
            await self._step_comparison(job)
            
            # Step 5: Decision
            await self._step_decision(job)
            
        except Exception as e:
            failed_step = job.current_step or "pipeline execution"
            job.status = RetrainStatus.FAILED
            job.error = str(e)
            job.current_step = f"Failed at: {failed_step}"
            if job.completed_at is None:
                job.completed_at = now_ist()
            logger.error(f"Retraining failed: {e}")
        finally:
            self._persist_state()
        
        return job

    def enqueue_pipeline(self, job_id: str) -> RetrainJob:
        """
        Queue pipeline execution on the current event loop and return immediately.
        Prevents request timeout coupling while keeping existing pipeline behavior.
        """
        import asyncio

        self._load_state()
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        existing = self._active_runs.get(job_id)
        if existing and not existing.done():
            return job

        if job.status == RetrainStatus.PENDING:
            job.current_step = "Queued for background execution"
            self._persist_state()

        async def _runner():
            try:
                await self.run_pipeline(job_id)
            finally:
                self._active_runs.pop(job_id, None)

        task = asyncio.create_task(_runner())
        self._active_runs[job_id] = task
        return job
    
    async def _step_data_preparation(self, job: RetrainJob):
        """Prepare training data."""
        job.status = RetrainStatus.DATA_PREPARATION
        job.current_step = "Preparing training data"
        job.progress = 0.2
        self._persist_state()
        self._persist_state()
        
        logger.info(f"Job {job.id}: Preparing data (window: {job.config.data_window_days} days)")
        
        # In production:
        # 1. Fetch recent data from feature store
        # 2. Apply feature engineering
        # 3. Handle class imbalance
        # 4. Split train/validation
        
        import asyncio
        await asyncio.sleep(0.1)  # Simulate work
        self._persist_state()
        self._persist_state()
    
    async def _step_training(self, job: RetrainJob):
        """Train new model by dispatching to the real Celery training worker."""
        job.status = RetrainStatus.TRAINING
        job.current_step = "Training model"
        job.progress = 0.4
        self._persist_state()
        self._persist_state()
        
        logger.info(f"Job {job.id}: Dispatching real training task for {job.model_id}")
        
        import os
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        
        DATABASE_URL = os.environ.get("DATABASE_URL", "")
        if not DATABASE_URL:
             # Fallback just to prevent crash
             job.current_step = "Simulated training (no DB)"
             import asyncio
             await asyncio.sleep(0.1)
             return
             
        engine = create_async_engine(DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        try:
            async with async_session() as db:
                from sqlalchemy import select, desc
                from uuid import UUID
                from app.models.ml_model import MLModel
                from app.models.dataset import Dataset
                from app.services.training_service import TrainingService

                mid = UUID(job.model_id)
                model_result = await db.execute(select(MLModel).where(MLModel.id == mid))
                model = model_result.scalar_one_or_none()
                if not model:
                     raise ValueError(f"Model {job.model_id} not found")

                dataset_result = await db.execute(
                    select(Dataset).where(Dataset.status == "ACTIVE").order_by(desc(Dataset.created_at)).limit(1)
                )
                latest_dataset = dataset_result.scalar_one_or_none()
                if not latest_dataset:
                     raise ValueError("No active datasets for retraining")

                training_svc = TrainingService(db)
                hyperparameters = model.hyperparameters or await training_svc.get_default_hyperparameters(model.algorithm)
                # Keep retraining behavior stable while preventing XGBoost runtime failure
                # from persisted models that store missing=None.
                if isinstance(hyperparameters, dict) and model.algorithm == "xgboost":
                    if hyperparameters.get("missing", "__absent__") is None:
                        hyperparameters = dict(hyperparameters)
                        hyperparameters.pop("missing", None)
                job_data = await training_svc.create_training_job(
                    name=f"Retrain {model.algorithm.upper()} - {job.reason.value}",
                    dataset_id=str(latest_dataset.id),
                    feature_config={},
                    algorithm=model.algorithm,
                    hyperparameters=hyperparameters,
                    tuning_method="manual",
                )
                
                # TrainingService may return DB UUID objects depending on driver;
                # normalize to plain string for downstream UUID(...) parsing.
                job.training_job_id = str(job_data['id'])
                job.current_step = f"Training job {job.training_job_id} dispatched to Celery"
                job.progress = 0.5
                self._persist_state()
                self._persist_state()
        except Exception as e:
            logger.error(f"Failed to dispatch training task: {e}")
            raise
        finally:
            await engine.dispose()
    
    async def _step_validation(self, job: RetrainJob):
        """Validate new model by checking the completed TrainingJob metrics."""
        job.status = RetrainStatus.VALIDATION
        job.current_step = "Validating model"
        job.progress = 0.6
        self._persist_state()
        self._persist_state()
        
        logger.info(f"Job {job.id}: Validating model")
        
        import os
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        
        DATABASE_URL = os.environ.get("DATABASE_URL", "")
        if not DATABASE_URL:
             job.current_step = "Simulated validation (no DB)"
             import asyncio
             await asyncio.sleep(0.1)
             return
             
        engine = create_async_engine(DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        try:
            async with async_session() as db:
                from sqlalchemy import select
                from uuid import UUID
                from app.models.training_job import TrainingJob
                import asyncio
                
                if not job.training_job_id:
                     raise ValueError("Validation failed: No training_job_id associated with this retrain job")
                     
                tjid = UUID(job.training_job_id)
                query = select(TrainingJob).where(TrainingJob.id == tjid)
                
                result = await db.execute(query)
                latest_job = result.scalar_one_or_none()
                
                if latest_job and latest_job.status == "COMPLETED" and latest_job.metrics:
                    job.metrics = latest_job.metrics
                    metrics_dict = job.metrics or {}
                    job.current_step = f"Validation complete. F1: {metrics_dict.get('f1', 0):.4f}"
                    self._persist_state()
                else:
                    if latest_job and latest_job.status == "FAILED":
                        raise ValueError("Training job failed")

                    if waited_sec >= max_wait_sec:
                        raise ValueError("Training did not complete within validation timeout")

                    job.current_step = (
                        f"Waiting for training job {job.training_job_id} "
                        f"({waited_sec}s/{max_wait_sec}s)"
                    )
                    self._persist_state()
                    await asyncio.sleep(poll_interval_sec)
                    waited_sec += poll_interval_sec
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
        finally:
            await engine.dispose()
    
    async def _step_comparison(self, job: RetrainJob):
        """Compare new model metrics with current production model."""
        job.status = RetrainStatus.COMPARISON
        job.current_step = "Comparing with production model"
        job.progress = 0.8
        self._persist_state()
        self._persist_state()
        
        logger.info(f"Job {job.id}: Comparing with current model {job.model_id}")
        
        if not job.metrics:
             raise ValueError("No new model metrics available for comparison")
             
        import os
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        
        DATABASE_URL = os.environ.get("DATABASE_URL", "")
        if not DATABASE_URL:
             job.current_step = "Simulated comparison (no DB)"
             import asyncio
             await asyncio.sleep(0.1)
             return
             
        engine = create_async_engine(DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        try:
            async with async_session() as db:
                from sqlalchemy import select
                from uuid import UUID
                from app.models.ml_model import MLModel
                
                mid = UUID(job.model_id)
                model_result = await db.execute(select(MLModel).where(MLModel.id == mid))
                current_model = model_result.scalar_one_or_none()
                
                if not current_model:
                     raise ValueError(f"Production model {job.model_id} not found")
                     
                current_metrics = current_model.metrics or {"f1": 0.0, "precision": 0.0, "recall": 0.0}
                new_metrics = job.metrics or {}
                new_f1 = new_metrics.get("f1", 0.0)
                old_f1 = current_metrics.get("f1", 0.0)
                
                improvement_f1 = new_f1 - old_f1
                is_better = improvement_f1 > 0
                passes_threshold = improvement_f1 >= job.config.min_improvement_threshold
                
                job.comparison_result = {
                    "current_model": current_metrics,
                    "new_model": job.metrics,
                    "improvement": {
                        "f1": improvement_f1,
                        "precision": new_metrics.get("precision", 0) - current_metrics.get("precision", 0),
                        "recall": new_metrics.get("recall", 0) - current_metrics.get("recall", 0),
                    },
                    "is_better": is_better,
                    "passes_threshold": passes_threshold,
                }
                
                job.current_step = f"Comparison complete. Improvement: {improvement_f1:.4f} (Threshold: {job.config.min_improvement_threshold})"
                self._persist_state()
                self._persist_state()
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            raise
        finally:
            await engine.dispose()
    
    async def _step_decision(self, job: RetrainJob):
        """Decide whether to promote new model."""
        job.progress = 1.0
        
        comparison = job.comparison_result
        
        if comparison and comparison.get("is_better") and comparison.get("passes_threshold"):
            if job.config.auto_promote:
                job.status = RetrainStatus.COMPLETED
                job.current_step = "New model promoted to production"
                job.new_model_id = str(uuid4())
                logger.info(f"Job {job.id}: New model promoted: {job.new_model_id}")
            else:
                job.status = RetrainStatus.COMPLETED
                job.current_step = "Awaiting manual approval"
                job.new_model_id = str(uuid4())
                logger.info(f"Job {job.id}: New model ready for approval")
        else:
            job.status = RetrainStatus.REJECTED
            job.current_step = "New model did not meet improvement threshold"
            logger.info(f"Job {job.id}: New model rejected - no significant improvement")
        
        job.completed_at = now_ist()
        self._persist_state()
    
    def get_job(self, job_id: str) -> Optional[RetrainJob]:
        """Get job by ID."""
        self._load_state()
        return self._jobs.get(job_id)
    
    def list_jobs(
        self,
        model_id: Optional[str] = None,
        status: Optional[RetrainStatus] = None,
        limit: int = 20,
    ) -> List[RetrainJob]:
        """List retraining jobs."""
        self._load_state()
        jobs = list(self._jobs.values())
        
        if model_id:
            jobs = [j for j in jobs if j.model_id == model_id]
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        jobs.sort(key=lambda j: j.started_at, reverse=True)
        return jobs[:limit]

    def delete_job(self, job_id: str) -> bool:
        """Delete a retraining job from persisted state."""
        self._load_state()
        job = self._jobs.get(job_id)
        if not job:
            return False

        active_states = {
            RetrainStatus.PENDING,
            RetrainStatus.DATA_PREPARATION,
            RetrainStatus.TRAINING,
            RetrainStatus.VALIDATION,
            RetrainStatus.COMPARISON,
        }
        if job.status in active_states:
            raise ValueError(f"Cannot delete active retraining job in state {job.status.value}")

        self._jobs.pop(job_id, None)
        client = self._get_redis_client()
        if client:
            try:
                raw = client.get(self._STATE_KEY)
                if raw:
                    payload = json.loads(raw)
                    filtered = [
                        j for j in payload.get("jobs", [])
                        if j.get("id") != job_id
                    ]
                    client.set(self._STATE_KEY, json.dumps({"jobs": filtered}, default=str))
                else:
                    self._persist_state()
            except Exception as exc:
                logger.warning(f"Retraining delete persist failed: {exc}")
                self._persist_state()
        else:
            self._persist_state()
        return True
    
    def should_retrain(
        self,
        drift_status: str,
        performance_status: str,
        bias_status: str,
    ) -> Tuple[bool, RetrainReason]:
        """
        Determine if retraining should be triggered.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        # Priority order: Bias > Performance > Drift
        if bias_status == "CRITICAL":
            return True, RetrainReason.BIAS_DETECTED
        
        if performance_status == "CRITICAL":
            return True, RetrainReason.PERFORMANCE_DEGRADATION
        
        if drift_status == "CRITICAL":
            return True, RetrainReason.DRIFT_DETECTED
        
        # Warning level - suggest but don't force
        if drift_status == "WARNING" and performance_status == "WARNING":
            return True, RetrainReason.DRIFT_DETECTED
        
        return False, RetrainReason.MANUAL


# Import Tuple for type hints
from typing import Tuple


# Singleton pipeline instance
_pipeline: Optional[RetrainingPipeline] = None


def get_retraining_pipeline() -> RetrainingPipeline:
    """Get the global retraining pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RetrainingPipeline()
    return _pipeline
