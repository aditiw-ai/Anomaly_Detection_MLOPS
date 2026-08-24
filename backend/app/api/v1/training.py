"""
Training API Endpoints
Model training job management.
"""
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.core.dependencies import require_auth, can_train_models, can_manage_jobs
from app.core.auth import User
from app.services.training_service import TrainingService
from app.schemas.hyperparameter_validator import check_hyperparameters

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["Training"])


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid numeric hyperparameter")
    return float(value)


def _collect_tuning_contract_errors(
    algorithm: str,
    tuning_method: str,
    hyperparameters: Dict[str, Any],
) -> list[str]:
    """
    Strict pre-queue validation for tuning payload shape/ranges.
    Returns hard errors (prefixed with ❌).
    """
    errors: list[str] = []

    # XGBoost fractional params that must stay in (0, 1]
    xgb_unit_interval = {
        "subsample",
        "colsample_bytree",
        "colsample_bylevel",
        "colsample_bynode",
        "learning_rate",
    }

    for key, value in hyperparameters.items():
        # Range dict validation used by random/bayesian/grid UI
        if isinstance(value, dict) and ("min" in value or "max" in value):
            min_v = value.get("min")
            max_v = value.get("max")
            if min_v is None or max_v is None:
                errors.append(
                    f"❌ '{key}' range must include both min and max values."
                )
                continue
            try:
                min_f = _to_float(min_v)
                max_f = _to_float(max_v)
            except (TypeError, ValueError):
                errors.append(
                    f"❌ '{key}' range bounds must be numeric (received min={min_v}, max={max_v})."
                )
                continue
            if min_f > max_f:
                errors.append(
                    f"❌ '{key}' has invalid range: min ({min_v}) cannot be greater than max ({max_v})."
                )
            if tuning_method == "bayesian" and min_f == max_f:
                errors.append(
                    f"❌ Bayesian tuning requires a strict range for '{key}' (min and max cannot be equal)."
                )

        # Grid search should receive explicit lists (not min/max dict ranges)
        if tuning_method == "grid" and isinstance(value, dict) and ("min" in value or "max" in value):
            errors.append(
                f"❌ GridSearch requires explicit value lists for '{key}', not min/max range objects."
            )

        # XGBoost bounds
        if algorithm == "xgboost" and key in xgb_unit_interval:
            values_to_check: list[Any] = []
            if isinstance(value, list):
                values_to_check = value
            elif isinstance(value, dict) and ("min" in value or "max" in value):
                values_to_check = [value.get("min"), value.get("max")]
            else:
                values_to_check = [value]
            for v in values_to_check:
                if v is None:
                    continue
                try:
                    fv = _to_float(v)
                except (TypeError, ValueError):
                    errors.append(f"❌ '{key}' value '{v}' must be numeric.")
                    continue
                if not (0 < fv <= 1):
                    errors.append(
                        f"❌ '{key}' value '{v}' is out of bounds. Expected range is (0, 1]."
                    )

    return errors


class TrainingJobRequest(BaseModel):
    """Request body for creating a training job."""
    name: str
    dataset_id: str
    feature_set_id: Optional[str] = None
    feature_config: Dict[str, Any] = Field(default_factory=dict)
    algorithm: str = "xgboost"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    tuning_method: str = "manual"  # manual, grid, random, bayesian
    tuning_config: Dict[str, Any] = Field(default_factory=dict)
    imbalanced_strategy: str = "class_weight"  # class_weight, smote, undersample
    test_size: float = 0.2
    processing_only: bool = False  # If True, only prepares data (split/fe) and stops


@router.get("/jobs")
async def list_training_jobs(
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(require_auth),
):
    """List all training jobs."""
    service = TrainingService(db)
    jobs, total = await service.list_training_jobs(
        status=status,
        page=page,
        page_size=min(page_size, 100),
    )
    
    return {
        "data": jobs,
        "meta": {
            "page": page,
            "page_size": page_size,
            "total": total,
        }
    }


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_training_job(
    request: TrainingJobRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_train_models),
):
    """
    Create a new training job.

    For "Split & Save to Cloud", processing_only=True: prepares data (split + upload) and creates a job in DATA_PREPARED state.
    The job will be queued for async execution by Celery workers when processing_only=False.
    """
    service = TrainingService(db)

    # Get default hyperparameters if not provided
    if not request.hyperparameters:
        request.hyperparameters = await service.get_default_hyperparameters(request.algorithm)

    # ── Hyperparameter conflict check ─────────────────────────────────────
    # Runs BEFORE job creation. Collects all constraint violations as
    # plain-English warnings. Does NOT modify params or block job creation
    # (except for cases explicitly marked as hard-rejections below).
    #
    # Inject _tuning_method so per-algorithm checkers can react to the
    # tuning strategy (e.g. IsolationForest cannot use grid/random/bayesian).
    validation_params = {**request.hyperparameters, "_tuning_method": request.tuning_method}
    validation_warnings = check_hyperparameters(
        request.algorithm,
        validation_params
    )

    validation_warnings.extend(
        _collect_tuning_contract_errors(
            algorithm=request.algorithm,
            tuning_method=request.tuning_method,
            hyperparameters=request.hyperparameters,
        )
    )

    if validation_warnings:
        logger.warning(
            "Hyperparameter conflict(s) detected for algorithm='%s': %s",
            request.algorithm,
            validation_warnings
        )
        has_errors = any(w.startswith("❌") for w in validation_warnings)
        # Hard-reject for any invalid hyperparameter/tuning contract error
        if has_errors:
            return {
                "data": None,
                "message": "Job rejected due to invalid hyperparameters",
                "validation_warnings": validation_warnings,
                "is_rejected": True
            }
    try:
        job = await service.create_training_job(
            name=request.name,
            dataset_id=request.dataset_id,
            feature_set_id=request.feature_set_id,
            feature_config=request.feature_config,
            algorithm=request.algorithm,
            hyperparameters={
                **request.hyperparameters,
                "imbalanced_strategy": request.imbalanced_strategy,
                "test_size": request.test_size,
            },
            tuning_method=request.tuning_method,
            tuning_config=request.tuning_config,
            processing_only=request.processing_only,
            user_id=str(current_user.id),
            enforce_preprocessing_commit=False,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Create training job failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    return {
        "data": job,
        "message": "Training job created and queued",
        # Empty list = no issues found.
        # Non-empty = the job was still queued but these combinations
        # may cause training to fail — fix them before retrying.
        "validation_warnings": validation_warnings,
    }


@router.get("/jobs/{job_id}")
async def get_training_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(require_auth),
):
    """Get training job status and details."""
    service = TrainingService(db)
    job = await service.get_training_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return {"data": job}


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(can_manage_jobs),
):
    """Delete a training job."""
    service = TrainingService(db)
    success = await service.delete_training_job(job_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Training job not found")


@router.get("/algorithms")
async def list_algorithms(db: AsyncSession = Depends(get_db)):
    """
    List available ML algorithms with their hyperparameters.
    
    Each algorithm includes:
    - id: Algorithm identifier
    - name: Display name
    - description: Brief description
    - hyperparameters: Configurable parameters with defaults
    """
    service = TrainingService(db)
    algorithms = await service.list_algorithms()
    return {"data": algorithms}


@router.get("/algorithms/{algorithm_id}/defaults")
async def get_algorithm_defaults(
    algorithm_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get default hyperparameters for an algorithm."""
    service = TrainingService(db)
    defaults = await service.get_default_hyperparameters(algorithm_id)
    
    if not defaults:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    
    return {"data": defaults}
