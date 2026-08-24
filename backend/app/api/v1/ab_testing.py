"""
A/B Testing API Endpoints
Manage champion-challenger model tests.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

import asyncio
from app.core.database import get_db, async_session_maker
from app.services.ab_testing_service import (
    get_ab_testing_service,
    ABTestStatus,
    ABTestResult,
    ABTestConfig,
)
from app.services.data_service import DataService
from app.services.inference_service import InferenceService


router = APIRouter(prefix="/ab-tests", tags=["A/B Testing"])


class CreateTestRequest(BaseModel):
    """Request to create an A/B test."""
    name: str
    champion_model_id: str
    challenger_model_id: str
    challenger_traffic_percent: float = 10.0
    min_samples: int = 1000
    max_duration_hours: int = 168
    primary_metric: str = "f1"
    secondary_metrics: List[str] = ["precision", "recall", "auc"]
    auto_promote_on_win: bool = False


class SimulateRequest(BaseModel):
    """Request to simulate traffic for an A/B test."""
    dataset_id: str
    rows: int = 1000
    reset_existing: bool = True


class ConcludeTestRequest(BaseModel):
    """Request payload to conclude an A/B test."""
    result: str
    promote_challenger: bool = False


class ABTestResponse(BaseModel):
    """A/B test response."""
    id: str
    name: str
    champion_model_id: str
    challenger_model_id: str
    status: str
    result: str
    champion_samples: int
    challenger_samples: int


@router.get("/{test_id}/simulation-progress")
async def get_simulation_progress(
    test_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get live simulation progress for a test."""
    service = get_ab_testing_service()
    test = service.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    
    return {"data": service.get_simulation_progress(test_id)}


@router.post("")
async def create_ab_test(
    request: CreateTestRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new A/B test."""
    service = get_ab_testing_service()
    
    config = ABTestConfig(
        challenger_traffic_percent=request.challenger_traffic_percent,
        min_samples=request.min_samples,
        max_duration_hours=request.max_duration_hours,
        primary_metric=request.primary_metric,
        secondary_metrics=request.secondary_metrics,
        auto_promote_on_win=request.auto_promote_on_win,
    )
    
    test = service.create_test(
        name=request.name,
        champion_model_id=request.champion_model_id,
        challenger_model_id=request.challenger_model_id,
        config=config,
    )
    
    return {
        "data": {
            "id": test.id,
            "name": test.name,
            "status": test.status.value,
            "created_at": test.created_at.isoformat(),
        },
        "message": "A/B test created"
    }


@router.post("/{test_id}/start")
async def start_ab_test(
    test_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Start an A/B test."""
    service = get_ab_testing_service()
    
    try:
        test = service.start_test(test_id)
        return {
            "data": {
                "id": test.id,
                "status": test.status.value,
                "started_at": test.started_at.isoformat(),
            },
            "message": "A/B test started"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{test_id}")
async def get_ab_test(
    test_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get A/B test details."""
    service = get_ab_testing_service()
    
    test = service.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    
    return {
        "data": {
            "id": test.id,
            "name": test.name,
            "champion_model_id": test.champion_model_id,
            "challenger_model_id": test.challenger_model_id,
            "status": test.status.value,
            "result": test.result.value,
            "champion_samples": test.champion_samples,
            "challenger_samples": test.challenger_samples,
            "champion_metrics": test.champion_metrics,
            "challenger_metrics": test.challenger_metrics,
            "created_at": test.created_at.isoformat(),
            "started_at": test.started_at.isoformat() if test.started_at else None,
            "ended_at": test.ended_at.isoformat() if test.ended_at else None,
            "config": {
                "challenger_traffic_percent": test.config.challenger_traffic_percent,
                "min_samples": test.config.min_samples,
                "primary_metric": test.config.primary_metric,
            },
        }
    }


@router.get("")
async def list_ab_tests(
    status: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """List A/B tests."""
    service = get_ab_testing_service()
    
    st = ABTestStatus(status) if status else None
    tests = service.list_tests(status=st, limit=limit)
    
    return {
        "data": [
            {
                "id": t.id,
                "name": t.name,
                "champion_model_id": t.champion_model_id,
                "challenger_model_id": t.challenger_model_id,
                "status": t.status.value,
                "result": t.result.value,
                "champion_samples": t.champion_samples,
                "challenger_samples": t.challenger_samples,
                "champion_metrics": t.champion_metrics,
                "challenger_metrics": t.challenger_metrics,
                "min_samples": t.config.min_samples,
                "challenger_traffic_percent": t.config.challenger_traffic_percent,
                "created_at": t.created_at.isoformat(),
            }
            for t in tests
        ],
        "meta": {
            "total": len(tests),
        }
    }


@router.post("/{test_id}/simulate")
async def simulate_ab_test(
    test_id: str,
    request: SimulateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Simulate live traffic for a running A/B test using a dataset.
    This runs real inference on both models to produce comparison metrics.
    """
    ab_service = get_ab_testing_service()
    test = ab_service.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")

    # Return immediately and perform heavy data loading + inference in background.
    ab_service.mark_simulation_initializing(test_id=test_id, requested_rows=request.rows)

    async def _run_bg():
        async with async_session_maker() as bg_db:
            try:
                data_service = DataService(bg_db)
                inference_service = InferenceService.get_instance()
                preview = await data_service.preview_dataset(request.dataset_id, rows=request.rows)
                if not preview or not preview.get("rows"):
                    ab_service._finalize_simulation_progress(
                        test_id,
                        status="FAILED",
                        message="Dataset not found or empty",
                    )
                    return

                transactions = preview["rows"]
                columns = preview.get("columns", [])

                inferred_label_key = ab_service._infer_label_key(transactions)
                labelled_preview_count = 0
                if inferred_label_key:
                    for row in transactions:
                        if ab_service._extract_actual_label(row, inferred_key=inferred_label_key) is not None:
                            labelled_preview_count += 1

                if (not inferred_label_key) or labelled_preview_count == 0:
                    compact_cols = ", ".join(columns[:12]) if columns else "unknown"
                    if len(columns) > 12:
                        compact_cols += ", ..."
                    ab_service._finalize_simulation_progress(
                        test_id,
                        status="FAILED",
                        message=(
                            "Selected dataset has no usable binary ground-truth label column in sampled rows. "
                            f"Detected label key: {inferred_label_key or 'none'}. "
                            f"Available columns: {compact_cols}. "
                            "Use labels like is_fraud / label / target / class with 0/1 values."
                        ),
                    )
                    return

                await ab_service.simulate_traffic_from_dataset(
                    test_id=test_id,
                    transactions=transactions,
                    inference_service=inference_service,
                    db=bg_db,
                    reset_existing=request.reset_existing,
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Background A/B simulation failed: {e}")

    asyncio.create_task(_run_bg())

    return {
        "message": "Simulation started in background",
        "data": ab_service.get_simulation_progress(test_id)
    }


@router.post("/{test_id}/evaluate")
async def evaluate_ab_test(
    test_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Evaluate A/B test results."""
    service = get_ab_testing_service()
    
    try:
        evaluation = service.evaluate_test(test_id)
        return {"data": evaluation}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{test_id}/conclude")
async def conclude_ab_test(
    test_id: str,
    request: Optional[ConcludeTestRequest] = None,
    result: Optional[str] = Query(default=None),
    promote_challenger: Optional[bool] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Conclude an A/B test."""
    service = get_ab_testing_service()
    
    resolved_result = request.result if request else result
    resolved_promote = request.promote_challenger if request else bool(promote_challenger)

    if not resolved_result:
        raise HTTPException(status_code=400, detail="Missing result")

    try:
        ab_result = ABTestResult(resolved_result)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid result")
    
    try:
        test = service.conclude_test(
            test_id=test_id,
            result=ab_result,
            promote_challenger=resolved_promote,
        )
        return {
            "data": {
                "id": test.id,
                "status": test.status.value,
                "result": test.result.value,
                "ended_at": test.ended_at.isoformat(),
            },
            "message": "A/B test concluded"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{test_id}/abort")
async def abort_ab_test(
    test_id: str,
    reason: str = "",
    db: AsyncSession = Depends(get_db),
):
    """Abort a running A/B test."""
    service = get_ab_testing_service()
    
    try:
        test = service.abort_test(test_id, reason)
        return {
            "data": {
                "id": test.id,
                "status": test.status.value,
            },
            "message": "A/B test aborted"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/active/current")
async def get_active_test(
    db: AsyncSession = Depends(get_db),
):
    """Get currently active A/B test."""
    service = get_ab_testing_service()
    
    test = service.get_active_test()
    if not test:
        return {"data": None, "message": "No active test"}
    
    return {
        "data": {
            "id": test.id,
            "name": test.name,
            "status": test.status.value,
            "champion_samples": test.champion_samples,
            "challenger_samples": test.challenger_samples,
            "traffic_split": test.config.challenger_traffic_percent,
        }
    }


@router.delete("/{test_id}")
async def delete_ab_test(
    test_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Hard-delete an A/B test from persisted state."""
    service = get_ab_testing_service()
    if service.delete_test(test_id):
        return {"message": "A/B test deleted"}
    raise HTTPException(status_code=404, detail="Test not found")
