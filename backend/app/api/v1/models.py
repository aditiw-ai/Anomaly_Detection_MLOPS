"""
Models API Endpoints
Model registry and promotion.
"""
from typing import Optional, List
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_db
from app.core.dependencies import (
    require_auth,
    can_deploy_models,
    can_configure_monitoring,
    require_permission,
)
from app.core.auth import User, Permission
from app.services.training_service import ModelService
from app.services.classification_model_service import ClassificationModelService

router = APIRouter(prefix="/models", tags=["Models"])


class BaselineRequest(BaseModel):
    """Request body for setting baselines."""
    metric: str
    threshold: float
    operator: str = "gte"  # gte, lte, eq


class PromoteRequest(BaseModel):
    """Request body for model promotion."""
    target_status: str  # STAGING, PRODUCTION, ARCHIVED




@router.get("")
async def list_models(
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(require_auth),
):
    """List all models in the registry."""
    service = ModelService(db)
    models, total = await service.list_models(
        status=status,
        page=page,
        page_size=min(page_size, 100),
    )
    
    return {
        "data": [
            {
                "id": str(m.id),
                "name": m.name,
                "version": m.version,
                "algorithm": m.algorithm,
                "status": m.status,
                "metrics": m.metrics,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "promoted_at": m.promoted_at.isoformat() if m.promoted_at else None,
            }
            for m in models
        ],
        "meta": {
            "page": page,
            "page_size": page_size,
            "total": total,
        }
    }


@router.get("/classification-types")
async def list_classification_models(
    active_only: bool = Query(True, description="Return only active classification model types"),
    model_type: Optional[str] = Query(None, description="Filter: supervised or unsupervised"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    List all classification model types (algorithms) available for training.
    Use this to show the user which algorithms they can choose (e.g. XGBoost, LightGBM).
    """
    service = ClassificationModelService(db)
    rows, total = await service.list_classification_models(
        active_only=active_only,
        model_type=model_type,
        page=page,
        page_size=page_size,
    )
    data = [
        {
            "id": str(r.id),
            "algorithm_id": r.algorithm_id,
            "name": r.name,
            "description": r.description,
            "model_type": r.model_type,
            "hyperparameters_schema": r.hyperparameters_schema,
            "is_active": r.is_active,
        }
        for r in rows
    ]
    return {
        "data": data,
        "meta": {"page": page, "page_size": page_size, "total": total},
    }


@router.post("/classification-types/seed")
async def seed_classification_models(db: AsyncSession = Depends(get_db)):
    """
    Seed the classification_models table from the canonical list.
    Call once after migration. No body required.
    """
    service = ClassificationModelService(db)
    inserted = await service.seed_from_registry()
    await db.commit()
    return {"message": "Classification models seeded", "inserted": inserted}



@router.get("/production")
async def get_production_model(db: AsyncSession = Depends(get_db)):
    """Get the current production model."""
    service = ModelService(db)
    model = await service.get_production_model()
    
    if not model:
        return {"data": None, "message": "No production model deployed"}
    
    return {
        "data": {
            "id": str(model.id),
            "name": model.name,
            "version": model.version,
            "algorithm": model.algorithm,
            "metrics": model.metrics,
            "feature_importance": model.feature_importance,
            "promoted_at": model.promoted_at.isoformat() if model.promoted_at else None,
        }
    }


@router.get("/{model_id}")
async def get_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(require_auth),
):
    """Get model details."""
    service = ModelService(db)
    model = await service.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "data": {
            "id": str(model.id),
            "name": model.name,
            "version": model.version,
            "description": model.description,
            "algorithm": model.algorithm,
            "hyperparameters": model.hyperparameters,
            "status": model.status,
            "metrics": model.metrics,
            "feature_names": model.feature_names,
            "feature_importance": model.feature_importance,
            "storage_path": model.storage_path,
            "onnx_path": model.onnx_path,
            "checksum": model.checksum,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "promoted_at": model.promoted_at.isoformat() if model.promoted_at else None,
        }
    }


@router.post("/{model_id}/promote")
async def promote_model(
    model_id: str,
    request: PromoteRequest,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(can_deploy_models),
):
    """Promote a model to a new status (STAGING, PRODUCTION, ARCHIVED)."""
    if request.target_status not in ["STAGING", "PRODUCTION", "ARCHIVED"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid target status. Must be STAGING, PRODUCTION, or ARCHIVED"
        )
    
    service = ModelService(db)
    model = await service.promote_model(model_id, request.target_status)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "data": {
            "id": str(model.id),
            "status": model.status,
            "promoted_at": model.promoted_at.isoformat() if model.promoted_at else None,
        },
        "message": f"Model promoted to {request.target_status}"
    }


@router.post("/{model_id}/baselines")
async def set_baselines(
    model_id: str,
    baselines: List[BaselineRequest],
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(can_configure_monitoring),
):
    """Set performance baseline thresholds for a model."""
    service = ModelService(db)
    
    try:
        created = await service.set_baselines(
            model_id=model_id,
            baselines=[b.model_dump() for b in baselines],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "data": [
            {"metric": b.metric_name, "threshold": b.threshold, "operator": b.operator}
            for b in created
        ],
        "message": f"Created {len(created)} baselines"
    }


@router.get("/{model_id}/compare/{other_model_id}")
async def compare_models(
    model_id: str,
    other_model_id: str,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(require_auth),
):
    """Compare two models by their metrics."""
    service = ModelService(db)
    
    model1 = await service.get_model(model_id)
    model2 = await service.get_model(other_model_id)
    
    if not model1 or not model2:
        raise HTTPException(status_code=404, detail="One or both models not found")
    
    # Compare metrics
    comparison = {}
    all_metrics = set(model1.metrics.keys()) | set(model2.metrics.keys())
    
    for metric in all_metrics:
        v1 = model1.metrics.get(metric)
        v2 = model2.metrics.get(metric)
        
        if v1 is not None and v2 is not None:
            diff = v1 - v2
            winner = model_id if diff > 0 else other_model_id
        else:
            diff = None
            winner = None
        
        comparison[metric] = {
            model_id: v1,
            other_model_id: v2,
            "difference": diff,
            "winner": winner,
        }
    
    return {
        "data": comparison,
        "models": {
            model_id: {"name": model1.name, "version": model1.version},
            other_model_id: {"name": model2.name, "version": model2.version},
        }
    }


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(require_permission(Permission.MODEL_DELETE)),
):
    """Delete a model (hard delete - removes from database and storage)."""
    service = ModelService(db)
    try:
        success = await service.delete_model(model_id, hard_delete=True)
    except Exception as e:
        logger.exception("Model delete failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    return None



