import logging
from typing import Dict, Any, List
from uuid import UUID

from app.core.database import async_session_maker
from app.models.inference_log import InferenceLog

logger = logging.getLogger(__name__)


async def log_prediction(
    transaction_id: str,
    model_id: str,
    features: Dict[str, Any],
    prediction: int,
    fraud_score: float,
    response_time_ms: float
):
    """Asynchronously logs a single prediction to the database."""
    try:
        async with async_session_maker() as session:
            log_entry = InferenceLog(
                transaction_id=transaction_id,
                model_id=UUID(model_id) if model_id else None,
                features=features,
                prediction=prediction,
                fraud_score=fraud_score,
                response_time_ms=response_time_ms
            )
            session.add(log_entry)
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to log single prediction: {e}")


async def log_batch_predictions(
    model_id: str,
    predictions_data: List[Dict[str, Any]],
    transactions: List[Dict[str, Any]]
):
    """Asynchronously logs a batch of predictions to the database."""
    try:
        async with async_session_maker() as session:
            logs = []
            for idx, result in enumerate(predictions_data):
                features = transactions[idx]
                logs.append(
                    InferenceLog(
                        transaction_id=features.get("transaction_id"),
                        model_id=UUID(model_id) if model_id else None,
                        features=features,
                        prediction=result["prediction"],
                        fraud_score=result["fraud_score"],
                        response_time_ms=result.get("response_time_ms", 0.0)
                    )
                )
            session.add_all(logs)
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to log batch predictions: {e}")
