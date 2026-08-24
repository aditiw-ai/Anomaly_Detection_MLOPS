import pickle

import app.workers.training_worker as training_worker
from app.core import config as config_module


class _DummyPipeline:
    def __init__(self):
        self.named_steps = {
            "model": object(),
            "fraud_features": object(),
        }


class _DummyResult:
    def __init__(self):
        self.pipeline = _DummyPipeline()
        self.feature_names = ["amount", "is_weekend"]
        self.optimal_threshold = 0.42
        self.calibrator = None
        self.metrics = {"precision": 0.81, "recall": 0.72, "f1": 0.76, "auc": 0.83, "accuracy": 0.79}


def test_local_model_artifacts_are_saved_without_azure(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module.settings, "AZURE_STORAGE_CONNECTION_STRING", "", raising=False)

    class DummyConverter:
        class Result:
            def __init__(self):
                self.onnx_model = b"onnx-bytes"
                self.checksum = "abc123"

        def convert(self, raw_model, feature_names):
            return self.Result()

    monkeypatch.setattr(training_worker, "ONNXConverter", DummyConverter)

    result = _DummyResult()
    artifact_info = training_worker._save_model_artifacts(
        job_id="9a85a2ad-d9f3-4489-9d4e-1be8a7242683",
        result=result,
        algorithm="xgboost",
        base_dir=tmp_path,
    )

    model_path = tmp_path / "storage" / "models" / "9a85a2ad-d9f3-4489-9d4e-1be8a7242683" / "model.pkl"
    assert model_path.exists()
    assert artifact_info["model_storage_path"] == "storage/models/9a85a2ad-d9f3-4489-9d4e-1be8a7242683/model.pkl"
    assert artifact_info["onnx_storage_path"] == "storage/models/9a85a2ad-d9f3-4489-9d4e-1be8a7242683/model.onnx"
    assert model_path.read_bytes() == pickle.dumps(result.pipeline)
