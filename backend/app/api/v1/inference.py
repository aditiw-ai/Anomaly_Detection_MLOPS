"""
Inference API Endpoints
Real-time anomaly prediction using ONNX models.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.services.inference_service import InferenceService
from app.services.ab_testing_service import get_ab_testing_service
from app.services.inference_logger import log_prediction, log_batch_predictions

router = APIRouter(prefix="/inference", tags=["Inference"])


class PredictionRequest(BaseModel):
    """Request for single prediction."""
    transaction_id: Optional[str] = None
    features: dict = Field(..., description="Feature name to value mapping")


class BatchPredictionRequest(BaseModel):
    """Request for batch prediction."""
    transactions: List[dict] = Field(..., description="List of feature dicts")


class LoadModelRequest(BaseModel):
    """Request to load a specific model."""
    model_id: str


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    transaction_id: Optional[str] = None
    prediction: int
    anomaly_score: float
    confidence: float
    risk_level: str
    response_time_ms: float
    model_id: Optional[str] = None
    anomaly_type: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request to provide ground truth actual label."""
    transaction_id: str
    actual_label: int


@router.get("/models")
async def list_inference_models(db: AsyncSession = Depends(get_db)):
    """List models available for inference (have ONNX artifacts)."""
    service = InferenceService.get_instance()
    models = await service.list_available_models(db)
    return {"data": models, "total": len(models)}


@router.post("/load")
async def load_model(
    request: LoadModelRequest,
    db: AsyncSession = Depends(get_db),
):
    """Load a specific model for inference."""
    service = InferenceService.get_instance()
    try:
        info = await service.load_model(request.model_id, db)
        return {"data": info, "message": "Model loaded successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Make a single anomaly prediction using loaded ONNX model."""
    service = InferenceService.get_instance()
    
    if service.get_loaded_model_info() is None:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Call POST /inference/load first."
        )
    
    try:
        # A/B Testing: Check if an active test should route to a challenger model
        ab_service = get_ab_testing_service()
        routed_model_id = ab_service.route_request()
        
        actual_model_id = service.get_loaded_model_info().get("model_id") if service.get_loaded_model_info() else None

        if routed_model_id and routed_model_id != "default":
            try:
                await service.load_model(routed_model_id, db)
                actual_model_id = routed_model_id
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"A/B routing failed to load challenger model {routed_model_id}: {e}")
                # actual_model_id remains the previously loaded model ID
        elif routed_model_id == "default":
            # the router might return "default", we just use the loaded model
            pass
        
        result = service.predict_single(request.features)
        result["transaction_id"] = request.transaction_id
        
        # Record prediction for A/B metrics tracking
        active_test = ab_service.get_active_test()
        if active_test:
            ab_service.record_prediction(
                test_id=active_test.id,
                model_id=actual_model_id,
                prediction=result.get("prediction", 0),
                response_time_ms=result.get("response_time_ms", 0),
            )
        used_model_id = result.get("model_id") or actual_model_id

        background_tasks.add_task(
            log_prediction,
            transaction_id=request.transaction_id,
            model_id=used_model_id,
            features=request.features,
            prediction=result.get("prediction", 0),
            fraud_score=result.get("fraud_score", result.get("anomaly_score", 0.0)),
            response_time_ms=result.get("response_time_ms", 0.0)
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch")
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Make batch predictions for multiple transactions."""
    service = InferenceService.get_instance()
    
    if service.get_loaded_model_info() is None:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Call POST /inference/load first."
        )
    
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")
    
    try:
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(service.predict_batch, request.transactions)
        
        used_model_id = result["meta"].get("model_id") or service.get_loaded_model_info().get("model_id")
        
        background_tasks.add_task(
            log_batch_predictions,
            model_id=used_model_id,
            predictions_data=result["results"],
            transactions=request.transactions
        )
        
        return {"data": result["results"], "meta": result["meta"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/model/info")
async def get_model_info(db: AsyncSession = Depends(get_db)):
    """Get information about the currently loaded model."""
    service = InferenceService.get_instance()
    info = service.get_loaded_model_info()
    
    if info is None:
        return {"data": None, "message": "No model loaded"}
    
    return {"data": info}


@router.post("/feedback")
async def provide_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Provide ground truth actual label for a previous prediction."""
    from sqlalchemy import update
    from app.models.inference_log import InferenceLog
    
    result = await db.execute(
        update(InferenceLog)
        .where(InferenceLog.transaction_id == request.transaction_id)
        .values(actual_label=request.actual_label)
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Transaction not found in inference logs")
        
    return {"message": "Feedback recorded successfully"}