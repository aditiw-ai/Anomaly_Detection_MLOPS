"""
ONNX Inference Engine
High-performance model inference using ONNX Runtime.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of model inference."""
    prediction: int  # 0 or 1
    fraud_score: float  # Probability of fraud
    confidence: float  # Confidence in prediction
    response_time_ms: float
    feature_contributions: Optional[Dict[str, float]] = None


class ONNXInferenceEngine:
    """
    High-performance inference engine using ONNX Runtime.
    
    Features:
    - <10ms latency per prediction
    - Batch prediction support
    - Thread-safe
    - Automatic model warm-up
    """
    
    def __init__(self, onnx_bytes: bytes = None, onnx_path: str = None):
        """
        Initialize inference engine.
        
        Args:
            onnx_bytes: ONNX model as bytes
            onnx_path: Path to ONNX model file
        """
        import onnxruntime as ort
        
        # Configure ONNX Runtime for optimal performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2
        
        # Load model
        if onnx_bytes:
            self.session = ort.InferenceSession(onnx_bytes, sess_options)
        elif onnx_path:
            self.session = ort.InferenceSession(onnx_path, sess_options)
        else:
            raise ValueError("Must provide either onnx_bytes or onnx_path")
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Warm up the model
        self._warm_up()
        
        logger.info(f"ONNX Inference Engine initialized. Input: {self.input_name}")
    
    def _warm_up(self, n_warmup: int = 5):
        """Warm up the model to optimize JIT compilation."""
        input_shape = self.session.get_inputs()[0].shape
        feature_count = input_shape[1] if len(input_shape) > 1 else 30
        
        dummy_input = np.random.randn(1, feature_count).astype(np.float32)
        
        for _ in range(n_warmup):
            self.session.run(None, {self.input_name: dummy_input})
        
        logger.info(f"Model warmed up with {n_warmup} iterations")
    
    def predict(
        self,
        features: np.ndarray,
        return_contributions: bool = False,
    ) -> InferenceResult:
        """
        Make a single prediction.
        
        Args:
            features: Feature array of shape (n_features,) or (1, n_features)
            return_contributions: Whether to compute feature contributions
        
        Returns:
            InferenceResult with prediction and metadata
        """
        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        features = features.astype(np.float32)
        
        # Measure inference time
        start_time = time.perf_counter()
        
        outputs = self.session.run(None, {self.input_name: features})
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse outputs
        prediction = int(outputs[0][0])
        
        # Get probabilities
        if len(outputs) > 1 and outputs[1] is not None:
            probabilities = outputs[1][0]
            fraud_score = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        else:
            fraud_score = float(prediction)
        
        # Calculate confidence
        confidence = abs(fraud_score - 0.5) * 2  # Scale to 0-1
        
        # Feature contributions (placeholder - would need SHAP for real values)
        contributions = None
        if return_contributions:
            contributions = self._compute_contributions(features)
        
        return InferenceResult(
            prediction=prediction,
            fraud_score=fraud_score,
            confidence=confidence,
            response_time_ms=elapsed_ms,
            feature_contributions=contributions,
        )
    
    def predict_batch(
        self,
        features: np.ndarray,
    ) -> List[InferenceResult]:
        """
        Make batch predictions.
        
        Args:
            features: Feature array of shape (batch_size, n_features)
        
        Returns:
            List of InferenceResult objects
        """
        features = features.astype(np.float32)
        
        start_time = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: features})
        total_time_ms = (time.perf_counter() - start_time) * 1000
        
        predictions = outputs[0]
        probabilities = outputs[1] if len(outputs) > 1 else None
        
        results = []
        per_sample_time = total_time_ms / len(features)
        
        for i in range(len(features)):
            pred = int(predictions[i])
            
            if probabilities is not None:
                prob = probabilities[i]
                fraud_score = float(prob[1]) if len(prob) > 1 else float(prob[0])
            else:
                fraud_score = float(pred)
            
            confidence = abs(fraud_score - 0.5) * 2
            
            results.append(InferenceResult(
                prediction=pred,
                fraud_score=fraud_score,
                confidence=confidence,
                response_time_ms=per_sample_time,
            ))
        
        logger.info(f"Batch prediction: {len(features)} samples in {total_time_ms:.2f}ms "
                   f"({per_sample_time:.2f}ms per sample)")
        
        return results
    
    def _compute_contributions(
        self,
        features: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute feature contributions using SHAP (placeholder).
        
        In production, this would use TreeExplainer or equivalent.
        """
        # For now, return mock contributions
        # Real implementation would use SHAP
        n_features = features.shape[1]
        return {f"feature_{i}": float(features[0, i]) for i in range(min(10, n_features))}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        return {
            "inputs": [
                {
                    "name": i.name,
                    "shape": i.shape,
                    "type": i.type,
                }
                for i in inputs
            ],
            "outputs": [
                {
                    "name": o.name,
                    "shape": o.shape,
                    "type": o.type,
                }
                for o in outputs
            ],
            "providers": self.session.get_providers(),
        }
