"""
ONNX Model Conversion
Converts trained scikit-learn/XGBoost models to ONNX format for production inference.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
import io
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ONNXConversionResult:
    """Result of ONNX conversion."""
    onnx_model: bytes
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    model_size_bytes: int
    checksum: str
    metadata: Dict[str, Any]


class ONNXConverter:
    """
    Converts ML models to ONNX format for optimized inference.
    
    Supports:
    - XGBoost classifiers
    - LightGBM classifiers
    - Scikit-learn classifiers (RandomForest, LogisticRegression, etc.)
    """
    
    def __init__(self):
        self.supported_models = [
            "XGBClassifier",
            "LGBMClassifier", 
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "LogisticRegression",
        ]
    
    def convert(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str = "fraud_model",
    ) -> ONNXConversionResult:
        """
        Convert a trained model to ONNX format.
        
        Args:
            model: Trained sklearn/xgboost/lightgbm model
            feature_names: List of feature names
            model_name: Name for the ONNX model
        
        Returns:
            ONNXConversionResult with ONNX bytes and metadata
        """
        model_type = type(model).__name__
        logger.info(f"Converting {model_type} to ONNX...")
        
        if model_type not in self.supported_models:
            logger.warning(f"Model type {model_type} may not be fully supported")
        
        # Determine conversion method
        if model_type.startswith("XGB"):
            onnx_model = self._convert_xgboost(model, feature_names, model_name)
        elif model_type.startswith("LGBM"):
            onnx_model = self._convert_lightgbm(model, feature_names, model_name)
        else:
            onnx_model = self._convert_sklearn(model, feature_names, model_name)
        
        # Serialize to bytes
        onnx_bytes = self._serialize_onnx(onnx_model)
        
        # Compute checksum
        checksum = hashlib.sha256(onnx_bytes).hexdigest()
        
        # Build schemas
        input_schema = {
            "name": "input",
            "type": "float32",
            "shape": [None, len(feature_names)],
            "features": feature_names,
        }
        
        output_schema = {
            "prediction": {"type": "int64", "shape": [None]},
            "probabilities": {"type": "float32", "shape": [None, 2]},
        }
        
        logger.info(f"ONNX conversion complete. Size: {len(onnx_bytes)} bytes, Checksum: {checksum[:16]}...")
        
        return ONNXConversionResult(
            onnx_model=onnx_bytes,
            input_schema=input_schema,
            output_schema=output_schema,
            model_size_bytes=len(onnx_bytes),
            checksum=checksum,
            metadata={
                "model_type": model_type,
                "feature_count": len(feature_names),
                "model_name": model_name,
            }
        )
    
    def _convert_xgboost(
        self, 
        model: Any, 
        feature_names: List[str],
        model_name: str,
    ) -> Any:
        """Convert XGBoost model to ONNX."""
        try:
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            # Temporary override feature names in XGBoost booster to prevent ONNX regex failures
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
                if hasattr(booster, "feature_names"):
                    booster.feature_names = [f"f{i}" for i in range(len(feature_names))]
            
            initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_xgboost(
                model, 
                initial_types=initial_type,
                target_opset=12,
            )
            onnx_model.doc_string = model_name
            return onnx_model
            
        except ImportError as e:
            logger.error(f"onnxmltools not installed: {e}")
            raise RuntimeError("onnxmltools required for XGBoost conversion")
    
    def _convert_lightgbm(
        self, 
        model: Any, 
        feature_names: List[str],
        model_name: str,
    ) -> Any:
        """Convert LightGBM model to ONNX."""
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_lightgbm(
                model,
                initial_types=initial_type,
                target_opset=12,
            )
            onnx_model.doc_string = model_name
            return onnx_model
            
        except ImportError as e:
            logger.error(f"onnxmltools not installed: {e}")
            raise RuntimeError("onnxmltools required for LightGBM conversion")
    
    def _convert_sklearn(
        self, 
        model: Any, 
        feature_names: List[str],
        model_name: str,
    ) -> Any:
        """Convert sklearn model to ONNX."""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_type,
                target_opset=12,
                options={id(model): {"zipmap": False}},  # Disable zipmap for performance
            )
            onnx_model.doc_string = model_name
            return onnx_model
            
        except ImportError as e:
            logger.error(f"skl2onnx not installed: {e}")
            raise RuntimeError("skl2onnx required for sklearn conversion")
    
    def _serialize_onnx(self, onnx_model: Any) -> bytes:
        """Serialize ONNX model to bytes."""
        return onnx_model.SerializeToString()
    
    def validate_onnx(
        self,
        onnx_bytes: bytes,
        sample_input: np.ndarray,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate ONNX model with sample input.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and check model
            onnx_model = onnx.load_model_from_string(onnx_bytes)
            onnx.checker.check_model(onnx_model)
            
            # Run inference test
            session = ort.InferenceSession(onnx_bytes)
            input_name = session.get_inputs()[0].name
            
            output = session.run(None, {input_name: sample_input.astype(np.float32)})
            
            logger.info(f"ONNX validation passed. Output shape: {output[0].shape}")
            return True, None
            
        except Exception as e:
            error_msg = f"ONNX validation failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
