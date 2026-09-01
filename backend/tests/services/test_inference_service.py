import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import pytest
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.inference_service import InferenceService
from app.models.ml_model import MLModel
from ml.inference.onnx_engine import InferenceResult


@pytest.fixture
def mock_db_session():
    return AsyncMock(spec=AsyncSession)


@pytest.fixture(autouse=True)
def configure_azure_for_tests(monkeypatch):
    from app.core import config as config_module

    monkeypatch.setattr(
        config_module.settings,
        "AZURE_STORAGE_CONNECTION_STRING",
        "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test;EndpointSuffix=core.windows.net",
        raising=False,
    )


@pytest.fixture
def inference_service():
    """Get a fresh instance of InferenceService, resetting the singleton."""
    # Reset singleton state
    InferenceService._instance = None
    service = InferenceService.get_instance()
    return service


def test_select_stage2_model_ignores_stale_binary_candidates():
    now = datetime(2026, 8, 26, 16, 58, 20)
    stale_binary = MLModel(
        id=uuid4(),
        name="Stale Binary Stage 2",
        status="STAGING",
        created_at=now - timedelta(days=2),
        hyperparameters={"training_stage": "anomaly_type", "objective": "binary:logistic"},
    )
    older_multiclass = MLModel(
        id=uuid4(),
        name="Older Multiclass Stage 2",
        status="STAGING",
        created_at=now - timedelta(hours=1),
        hyperparameters={"training_stage": "anomaly_type", "objective": "multi:softprob", "num_class": 7},
    )
    current_multiclass = MLModel(
        id=uuid4(),
        name="Current Multiclass Stage 2",
        status="STAGING",
        created_at=now,
        hyperparameters={"training_stage": "anomaly_type", "objective": "multi:softprob", "num_class": 7},
    )

    selected = InferenceService._select_stage2_model([
        stale_binary,
        older_multiclass,
        current_multiclass,
    ])

    assert selected == current_multiclass


@pytest.mark.asyncio
async def test_list_available_models(inference_service, mock_db_session):
    """Test that listing models correctly parses SQLAlchemy join output."""
    m_id1 = uuid4()
    m_id2 = uuid4()
    
    mock_model_1 = MLModel(
        id=m_id1, name="Model 1", algorithm="xgboost", 
        status="PRODUCTION", version="1.0.0", feature_names=["f1", "f2"], onnx_path="path/to.onnx"
    )
    mock_model_2 = MLModel(
        id=m_id2, name="Model 2", algorithm="random_forest", 
        status="STAGING", version="1.0.1", feature_names=["f1", "f2"], onnx_path=None
    )
    
    # Simulate DB returning list of tuples: (MLModel, job_name_str)
    mock_result = MagicMock()
    mock_result.all.return_value = [
        (mock_model_1, "Job_Prod_Run"),
        (mock_model_2, None)  # No job name linked
    ]
    mock_db_session.execute.return_value = mock_result
    
    models = await inference_service.list_available_models(mock_db_session)
    
    assert len(models) == 2
    assert models[0]["model_id"] == str(m_id1)
    assert models[0]["name"] == "Job_Prod_Run (xgboost)"
    assert models[0]["has_onnx"] is True
    
    assert models[1]["model_id"] == str(m_id2)
    assert models[1]["name"] == "Model 2"
    assert models[1]["has_onnx"] is False


@pytest.mark.asyncio
async def test_load_model_local_without_azure(monkeypatch, inference_service, mock_db_session, tmp_path):
    """Test loading model artifacts from backend/storage when Azure is absent."""
    from app.core import config as config_module

    model_id = str(uuid4())
    model_dir = tmp_path / "storage" / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    onnx_bytes = b"fake_onnx_bytes"
    (model_dir / "model.onnx").write_bytes(onnx_bytes)
    (model_dir / "model.pkl").write_bytes(pickle.dumps(DummyPipeline()))

    preprocessor = DummyPreprocessor()
    (model_dir / "preprocessor.pkl").write_bytes(pickle.dumps(preprocessor))
    manifest = {
        "training_job_id": model_id,
        "onnx_checksum": hashlib.sha256(onnx_bytes).hexdigest(),
        "optimal_threshold": 0.41,
        "is_calibrated": False,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(config_module.settings, "AZURE_STORAGE_CONNECTION_STRING", "", raising=False)
    monkeypatch.chdir(tmp_path)

    with patch("app.services.inference_service.storage_service") as mock_storage, \
         patch("app.services.inference_service.ONNXInferenceEngine") as mock_onnx_engine_cls:
        mock_storage.download_model = AsyncMock(side_effect=AssertionError("Azure storage should not be called when local artifacts exist"))

        mock_model = MLModel(
            id=model_id, name="Local Test Model", algorithm="xgboost",
            onnx_path=f"storage/models/{model_id}/model.onnx",
            storage_path=f"storage/models/{model_id}/model.pkl",
            feature_names=["f1", "f2"],
            status="STAGING", version="1.0.0"
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_model
        mock_db_session.execute.return_value = mock_result

        mock_engine_instance = MagicMock()
        mock_onnx_engine_cls.return_value = mock_engine_instance

        info = await inference_service.load_model(model_id, mock_db_session)

        assert info["inference_engine"] == "onnx"
        assert info["optimal_threshold"] == 0.41
        mock_onnx_engine_cls.assert_called_once_with(onnx_bytes=onnx_bytes)


class DummyPreprocessor:
    feature_names_in_ = ["f1", "f2"]

    def transform(self, df):
        return df


@pytest.mark.asyncio
@patch("app.services.inference_service.storage_service")
@patch("app.services.inference_service.ONNXInferenceEngine")
async def test_load_model_onnx_success(mock_onnx_engine_cls, mock_storage, inference_service, mock_db_session):
    """Test loading a model with an ONNX path successfully."""
    # Setup DB mock
    model_id = str(uuid4())
    mock_model = MLModel(
        id=model_id, name="Test ONNX Model", algorithm="lightgbm",
        onnx_path="models/123/model.onnx", feature_names=["f1", "f2"],
        status="PRODUCTION", version="1.0.0"
    )
    
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_model
    mock_db_session.execute.return_value = mock_result
    
    # Setup storage mock (returning empty/fake bytes)
    mock_storage.download_model = AsyncMock()
    # 1. onnx bytes, 2. preprocessor bytes, 3. manifest bytes
    # We will simulate missing manifest and preprocessor to test pure ONNX loading resilience
    mock_storage.download_model.side_effect = [
        b"fake_onnx_bytes",  
        FileNotFoundError("No preprocessor"), 
        FileNotFoundError("No manifest")
    ]
    
    # Setup ONNX engine mock
    mock_engine_instance = MagicMock()
    mock_onnx_engine_cls.return_value = mock_engine_instance
    
    # Act
    info = await inference_service.load_model(model_id, mock_db_session)
    
    # Assert
    assert info["model_id"] == model_id
    assert info["inference_engine"] == "onnx"
    assert inference_service._use_onnx is True
    
    mock_storage.download_model.assert_any_call("models/123/model.onnx")
    mock_onnx_engine_cls.assert_called_once_with(onnx_bytes=b"fake_onnx_bytes")


class DummyPipeline:
    def __init__(self):
        self.named_steps = {}
    def predict(self, X):
        return np.array([1])
    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])

@pytest.mark.asyncio
@patch("app.services.inference_service.storage_service")
async def test_load_model_pickle_fallback(mock_storage, inference_service, mock_db_session):
    """Test loading a model that only has a pickle path, bypassing ONNX."""
    model_id = str(uuid4())
    mock_model = MLModel(
        id=model_id, name="Test Pickle Model", algorithm="random_forest",
        onnx_path=None, storage_path="models/123/model.pkl", 
        feature_names=["f1", "f2"], status="STAGING"
    )
    
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_model
    mock_db_session.execute.return_value = mock_result
    
    # Use native Dummy class for pickle compatibility
    dummy_pipeline = DummyPipeline()
    
    mock_storage.download_model = AsyncMock(return_value=pickle.dumps(dummy_pipeline))
    
    info = await inference_service.load_model(model_id, mock_db_session)
    
    assert info["inference_engine"] == "pickle"
    assert inference_service._use_onnx is False


@pytest.mark.asyncio
async def test_predict_background_logging_uses_fraud_score(monkeypatch):
    """Regression: background logger must receive the project-standard fraud_score key."""
    from app.api.v1 import inference as inference_api
    from fastapi import BackgroundTasks

    fake_service = MagicMock()
    fake_service.get_loaded_model_info.return_value = {"model_id": "model-123"}
    fake_service.predict_single.return_value = {
        "prediction": 1,
        "fraud_score": 0.91,
        "anomaly_score": 0.91,
        "confidence": 0.82,
        "risk_level": "HIGH",
        "response_time_ms": 12.3,
        "model_id": "model-123",
    }

    monkeypatch.setattr(inference_api, "InferenceService", MagicMock())
    inference_api.InferenceService.get_instance.return_value = fake_service
    monkeypatch.setattr(inference_api, "get_ab_testing_service", lambda: MagicMock(route_request=lambda: None, get_active_test=lambda: None))

    bg_tasks = MagicMock(spec=BackgroundTasks)
    request = inference_api.PredictionRequest(transaction_id="txn-1", features={"amount": 100, "age": 30})

    result = await inference_api.predict(request=request, background_tasks=bg_tasks, db=MagicMock())

    assert result["fraud_score"] == 0.91
    assert result["anomaly_score"] == 0.91
    bg_tasks.add_task.assert_called_once()
    assert bg_tasks.add_task.call_args.kwargs["fraud_score"] == 0.91
    assert "anomaly_score" not in bg_tasks.add_task.call_args.kwargs


@pytest.mark.asyncio
@patch("app.services.inference_service.storage_service")
@patch("app.services.inference_service.ONNXInferenceEngine")
async def test_predict_single_and_batch_onnx(mock_onnx_cls, mock_storage, inference_service, mock_db_session):
    """Test both single and batch predictions using the ONNX engine."""
    # 1. Force the service to load a mock model
    model_id = str(uuid4())
    mock_model = MLModel(
        id=model_id, name="Test Predict", onnx_path="models/123/model.onnx"
    )
    
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_model
    mock_db_session.execute.return_value = mock_result
    
    # Storage side-effects
    mock_storage.download_model = AsyncMock(side_effect=[b"onnx", Exception(), Exception()])
    
    # Setup ONNX engine prediction outputs
    mock_engine = MagicMock()
    # predict() returns InferenceResult
    mock_engine.predict.return_value = InferenceResult(prediction=1, fraud_score=0.95, confidence=0.90, response_time_ms=1.5)
    # predict_batch() returns a list of InferenceResult
    mock_engine.predict_batch.return_value = [
        InferenceResult(prediction=1, fraud_score=0.95, confidence=0.90, response_time_ms=1.5),
        InferenceResult(prediction=0, fraud_score=0.10, confidence=0.80, response_time_ms=1.5)
    ]
    mock_onnx_cls.return_value = mock_engine
    
    # Load Model
    await inference_service.load_model(model_id, mock_db_session)
    
    # 2. Test predict_single
    single_res = inference_service.predict_single({"amount": 1000, "age": 30})
    assert single_res["prediction"] == 1
    assert single_res["fraud_score"] == 0.95
    assert single_res["risk_level"] == "CRITICAL"
    
    # 3. Test predict_batch
    batch_res = inference_service.predict_batch([
        {"amount": 1000, "age": 30},
        {"amount": 10, "age": 20}
    ])
    assert batch_res["meta"]["total_transactions"] == 2
    assert batch_res["meta"]["fraud_count"] == 1
    assert batch_res["meta"]["legit_count"] == 1
    assert batch_res["meta"]["total_amount"] == 1000.0
    assert batch_res["meta"]["all_transactions_amount"] == 1010.0
    
    results = batch_res["results"]
    assert results[0]["risk_level"] == "CRITICAL"
    assert results[1]["risk_level"] == "LOW"
    assert results[1]["fraud_score"] == 0.10
