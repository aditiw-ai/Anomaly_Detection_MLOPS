import numpy as np
import pandas as pd

from ml.algorithms.trainer import FraudDetectionTrainer


class _DummyBinaryPipeline:
    def predict(self, X):
        return np.array([0])

    def predict_proba(self, X):
        return np.array([[0.7, 0.3]])


class _DummyCalibrator:
    def predict_proba(self, X):
        return np.column_stack([1 - np.full(len(X), 0.8), np.full(len(X), 0.8)])


def test_binary_predict_uses_calibrated_probability_and_threshold():
    trainer = FraudDetectionTrainer()
    trainer.pipeline = _DummyBinaryPipeline()
    trainer.calibrator = _DummyCalibrator()
    trainer.optimal_threshold = 0.6
    trainer._is_binary_classifier = True

    prediction = trainer.predict(pd.DataFrame([{"amount": 1.0}]))

    assert prediction.tolist() == [1]
