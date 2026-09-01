"""
Inference Service
Loads trained models (ONNX + preprocessor) and provides prediction methods.
"""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pickle
import logging
import time
from typing import Dict, Any, List, Optional
from uuid import UUID

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.ml_model import MLModel
from app.core.storage import storage_service
from ml.inference.onnx_engine import ONNXInferenceEngine, InferenceResult

logger = logging.getLogger(__name__)


STAGE2_MODEL_CLASSES = {
    0: "All Attributes",
    1: "Dimensions Only",
    2: "Dimensions and Weight",
    3: "Price Only",
    4: "Price and Dimensions",
    5: "Price and Weight",
    6: "Weight Only",
}


class InferenceService:
    """
    Manages ONNX model loading and real-time prediction.
    
    Uses a hybrid approach:
    - Pickle preprocessor (AnomalyFeatureEngineer) for feature transformation
    - ONNX engine for fast model inference
    """

    @staticmethod
    def _resolve_local_model_dir(model_id: str, storage_path: Optional[str] = None, onnx_path: Optional[str] = None) -> Optional[Path]:
        """Return the local artifact directory for a model when Azure is unavailable."""
        backend_root = Path(__file__).resolve().parents[2]
        cwd_root = Path.cwd()
        roots = [backend_root, cwd_root]
        candidates: List[Path] = []

        for root in roots:
            if model_id:
                candidates.append(root / "storage" / "models" / str(model_id))

        for value in (storage_path, onnx_path):
            if not value:
                continue
            normalized = str(value).replace("\\", "/")
            for root in roots:
                if normalized.startswith("storage/"):
                    candidates.append(root / normalized)
                elif normalized.startswith("models/"):
                    candidates.append(root / "storage" / normalized)
                elif normalized.startswith("/"):
                    candidates.append(Path(normalized))

                if "/" in normalized:
                    maybe_dir = normalized.rsplit("/", 1)[0]
                    if maybe_dir.startswith("models/"):
                        candidates.append(root / "storage" / maybe_dir)
                    elif maybe_dir.startswith("storage/"):
                        candidates.append(root / maybe_dir)

        seen = set()
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists() and resolved.is_dir():
                return resolved

        for root in roots:
            fallback_dir = root / "storage" / "models" / str(model_id)
            if fallback_dir.exists() and fallback_dir.is_dir():
                return fallback_dir

        return None

    @staticmethod
    def _read_local_artifact_bytes(model_id: str, filename: str, storage_path: Optional[str] = None, onnx_path: Optional[str] = None) -> Optional[bytes]:
        """Read a local artifact bytes payload if it exists in the backend storage tree."""
        local_dir = InferenceService._resolve_local_model_dir(model_id, storage_path=storage_path, onnx_path=onnx_path)
        if local_dir is None:
            return None
        artifact_path = local_dir / filename
        if not artifact_path.exists() or artifact_path.is_dir():
            return None
        return artifact_path.read_bytes()

    @staticmethod
    def _select_stage2_model(candidates: List[MLModel]) -> Optional[MLModel]:
        """Pick the current multiclass anomaly-type model deterministically."""
        stage2_candidates = []
        for candidate in candidates:
            hyperparameters = candidate.hyperparameters if isinstance(candidate.hyperparameters, dict) else {}
            is_stage2 = hyperparameters.get("training_stage") == "anomaly_type"
            is_multiclass = (
                str(hyperparameters.get("objective", "")).startswith("multi:")
                or int(hyperparameters.get("num_class") or 0) > 2
            )
            if is_stage2 and is_multiclass:
                stage2_candidates.append(candidate)

        if not stage2_candidates:
            return None

        status_rank = {"PRODUCTION": 0, "STAGING": 1, "TRAINED": 2}

        def sort_key(candidate: MLModel):
            created_at = candidate.created_at or datetime.min
            created_rank = (
                created_at.toordinal(),
                created_at.hour,
                created_at.minute,
                created_at.second,
                created_at.microsecond,
            )
            return (
                status_rank.get(candidate.status, 99),
                tuple(-part for part in created_rank),
            )

        return sorted(stage2_candidates, key=sort_key)[0]
    
    _instance: Optional["InferenceService"] = None
    
    def __init__(self):
        self._loaded_model_id: Optional[str] = None
        self._onnx_engine: Optional[ONNXInferenceEngine] = None
        self._pipeline = None  # Pickle pipeline fallback
        self._preprocessor = None  # AnomalyFeatureEngineer
        self._calibrator = None  # Platt scaler (LogisticRegression on raw probs)
        self._model_info: Optional[Dict[str, Any]] = None
        self._feature_names: Optional[List[str]] = None
        self._use_onnx: bool = False
        self._optimal_threshold: float = 0.5  # Loaded from manifest; corrects scale_pos_weight bias
        self._stage2_service: Optional["InferenceService"] = None
        self._training_stage: str = "binary"
    
    @classmethod
    def get_instance(cls) -> "InferenceService":
        """Singleton. No threading lock — safe for asyncio context."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def list_available_models(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """List all trained models available for inference."""
        from app.models.training_job import TrainingJob
        
        result = await db.execute(
            select(MLModel, TrainingJob.name.label("job_name"))
            .outerjoin(TrainingJob, TrainingJob.model_id == MLModel.id)
            .where(MLModel.status.in_(['STAGING', 'PRODUCTION', 'TRAINED']))
            .order_by(desc(MLModel.created_at))
        )
        rows = result.all()
        
        return [
            {
                "model_id": str(m.id),
                "name": f"{job_name} ({m.algorithm})" if job_name else m.name,
                "algorithm": m.algorithm,
                "version": m.version,
                "status": m.status,
                "metrics": m.metrics,
                "feature_names": m.feature_names,
                "has_onnx": m.onnx_path is not None,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "is_loaded": str(m.id) == self._loaded_model_id,
            }
            for m, job_name in rows
        ]
    
    async def load_model(self, model_id: str, db: AsyncSession, _load_stage2: bool = True) -> Dict[str, Any]:
        """
        Load a model for inference. Prefers ONNX, falls back to pickle pipeline.
        """
        # Skip if already loaded
        if self._loaded_model_id == model_id and (self._onnx_engine is not None or self._pipeline is not None):
            logger.info(f"Model {model_id} already loaded, skipping")
            return self._model_info
        
        # Fetch model record
        result = await db.execute(
            select(MLModel).where(MLModel.id == UUID(model_id))
        )
        model = result.scalar_one_or_none()
        
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        logger.info(f"Loading model {model_id}: {model.name}")

        # Reset state
        self._onnx_engine = None
        self._pipeline = None
        self._preprocessor = None
        self._calibrator = None
        self._use_onnx = False
        self._optimal_threshold = 0.5  # Default; overridden by manifest if present
        model_hyperparameters = model.hyperparameters if isinstance(model.hyperparameters, dict) else {}
        self._training_stage = model_hyperparameters.get("training_stage", "binary")

        azure_configured = bool((settings.AZURE_STORAGE_CONNECTION_STRING or "").strip())
        if not azure_configured:
            logger.info("Azure not configured; loading model artifacts from local backend storage.")

        # Try ONNX first
        if model.onnx_path:
            try:
                if azure_configured:
                    onnx_bytes = await storage_service.download_model(model.onnx_path)
                    logger.info(f"Downloaded ONNX model: {len(onnx_bytes)} bytes")
                else:
                    onnx_bytes = self._read_local_artifact_bytes(model_id, "model.onnx", storage_path=model.storage_path, onnx_path=model.onnx_path)
                    if onnx_bytes is None:
                        raise FileNotFoundError(f"Local ONNX artifact not found for model {model_id}")
                    logger.info(f"Loaded ONNX model from local storage: {len(onnx_bytes)} bytes")

                artifact_base = model.onnx_path.rsplit("/", 1)[0] if model.onnx_path else None
                if azure_configured:
                    preprocessor_path = artifact_base + "/preprocessor.pkl" if artifact_base else None
                    try:
                        preprocessor_bytes = await storage_service.download_model(preprocessor_path)
                        self._preprocessor = pickle.loads(preprocessor_bytes)
                        logger.info("Preprocessor loaded")
                    except Exception as e:
                        logger.warning(f"Could not load preprocessor: {e}")

                    manifest_path = artifact_base + "/manifest.json" if artifact_base else None
                    try:
                        manifest_bytes = await storage_service.download_model(manifest_path)
                        manifest = json.loads(manifest_bytes)
                        actual_checksum = hashlib.sha256(onnx_bytes).hexdigest()
                        if manifest.get("onnx_checksum") and manifest["onnx_checksum"] != actual_checksum:
                            logger.warning(
                                f"ONNX checksum mismatch! Manifest expects {manifest['onnx_checksum'][:16]}... "
                                f"but loaded {actual_checksum[:16]}... — artifacts may be from different training jobs"
                            )
                        else:
                            logger.info(f"Artifact version verified: job={manifest.get('training_job_id', 'unknown')}")
                        if "optimal_threshold" in manifest:
                            self._optimal_threshold = float(manifest["optimal_threshold"])
                            logger.info(f"Loaded optimal threshold from manifest: {self._optimal_threshold:.4f}")
                        if manifest.get("is_calibrated"):
                            artifact_base = model.onnx_path.rsplit("/", 1)[0]
                            try:
                                cal_bytes = await storage_service.download_model(artifact_base + "/calibrator.pkl")
                                self._calibrator = pickle.loads(cal_bytes)
                                logger.info("Platt calibrator loaded — probabilities will be calibrated at inference")
                            except Exception as cal_e:
                                logger.warning(f"Could not load calibrator: {cal_e}")
                    except Exception as e:
                        logger.info(f"No version manifest found (older model): {e}")
                else:
                    local_dir = self._resolve_local_model_dir(model_id, storage_path=model.storage_path, onnx_path=model.onnx_path)
                    if local_dir is not None:
                        preprocessor_bytes = self._read_local_artifact_bytes(model_id, "preprocessor.pkl", storage_path=model.storage_path, onnx_path=model.onnx_path)
                        if preprocessor_bytes is not None:
                            self._preprocessor = pickle.loads(preprocessor_bytes)
                            logger.info("Preprocessor loaded from local storage")

                        manifest_bytes = self._read_local_artifact_bytes(model_id, "manifest.json", storage_path=model.storage_path, onnx_path=model.onnx_path)
                        if manifest_bytes is not None:
                            manifest = json.loads(manifest_bytes)
                            actual_checksum = hashlib.sha256(onnx_bytes).hexdigest()
                            if manifest.get("onnx_checksum") and manifest["onnx_checksum"] != actual_checksum:
                                logger.warning(
                                    f"ONNX checksum mismatch! Manifest expects {manifest['onnx_checksum'][:16]}... "
                                    f"but loaded {actual_checksum[:16]}... — artifacts may be from different training jobs"
                                )
                            else:
                                logger.info(f"Artifact version verified: job={manifest.get('training_job_id', 'unknown')}")
                            if "optimal_threshold" in manifest:
                                self._optimal_threshold = float(manifest["optimal_threshold"])
                                logger.info(f"Loaded optimal threshold from manifest: {self._optimal_threshold:.4f}")
                            if manifest.get("is_calibrated"):
                                calibrator_bytes = self._read_local_artifact_bytes(model_id, "calibrator.pkl", storage_path=model.storage_path, onnx_path=model.onnx_path)
                                if calibrator_bytes is not None:
                                    self._calibrator = pickle.loads(calibrator_bytes)
                                    logger.info("Platt calibrator loaded from local storage — probabilities will be calibrated at inference")

                self._onnx_engine = ONNXInferenceEngine(onnx_bytes=onnx_bytes)
                self._use_onnx = True
                logger.info("Using ONNX engine for inference")
            except Exception as e:
                logger.warning(f"ONNX load failed, falling back to pickle: {e}")

        # Fallback to pickle pipeline
        if not self._use_onnx and model.storage_path:
            try:
                if azure_configured:
                    pipeline_bytes = await storage_service.download_model(model.storage_path)
                else:
                    pipeline_bytes = self._read_local_artifact_bytes(model_id, "model.pkl", storage_path=model.storage_path, onnx_path=model.onnx_path)
                    if pipeline_bytes is None:
                        raise FileNotFoundError(f"Local pickle artifact not found for model {model_id}")
                self._pipeline = pickle.loads(pipeline_bytes)
                logger.info("Loaded pickle pipeline for inference")
            except Exception as e:
                raise ValueError(f"Failed to load model: {e}")
        
        if not self._use_onnx and self._pipeline is None:
            raise ValueError(f"Model {model_id} has no loadable artifacts")
        
        self._loaded_model_id = model_id
        self._feature_names = model.feature_names
        
        # Extract raw input column names for dynamic form
        input_features = None
        if self._preprocessor and hasattr(self._preprocessor, 'feature_names_in_'):
            input_features = self._preprocessor.feature_names_in_
        elif self._pipeline and hasattr(self._pipeline, 'named_steps'):
            fe = self._pipeline.named_steps.get('anomaly_features')
            if fe and hasattr(fe, 'feature_names_in_'):
                input_features = fe.feature_names_in_
        
        self._model_info = {
            "model_id": str(model.id),
            "name": model.name,
            "algorithm": model.algorithm,
            "version": model.version,
            "status": model.status,
            "metrics": model.metrics,
            "feature_names": model.feature_names,
            "input_features": input_features,
            "inference_engine": "onnx" if self._use_onnx else "pickle",
            "optimal_threshold": self._optimal_threshold,
        }
        # Also check metrics JSON for threshold persisted by trainer (pickle path / older ONNX models)
        if model.metrics and "optimal_threshold" in model.metrics:
            threshold_from_metrics = float(model.metrics["optimal_threshold"])
            # Manifest wins if already set by manifest load above; otherwise use metrics
            if self._optimal_threshold == 0.5:
                self._optimal_threshold = threshold_from_metrics
                self._model_info["optimal_threshold"] = self._optimal_threshold
                logger.info(f"Loaded optimal threshold from model metrics: {self._optimal_threshold:.4f}")

        logger.info(f"Model {model_id} loaded via {'ONNX' if self._use_onnx else 'pickle'} | threshold={self._optimal_threshold:.4f} | calibrated={self._calibrator is not None}")

        if _load_stage2 and self._training_stage != "anomaly_type":
            stage2_result = await db.execute(
                select(MLModel).where(MLModel.status.in_(['STAGING', 'PRODUCTION', 'TRAINED']))
            )
            stage2_model = self._select_stage2_model(stage2_result.scalars().all())
            if stage2_model is not None:
                self._stage2_service = InferenceService()
                await self._stage2_service.load_model(str(stage2_model.id), db, _load_stage2=False)

        return self._model_info
    
    def _calibrate(self, anomaly_score: float) -> float:
        """Apply Platt calibrator to a single raw probability score."""
        if self._calibrator is None:
            return anomaly_score
        import numpy as np
        raw = np.array([[anomaly_score]], dtype=np.float64)
        return float(self._calibrator.predict_proba(raw)[0, 1])

    def _calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt calibrator to a batch of raw probability scores."""
        if self._calibrator is None:
            return scores
        raw = scores.reshape(-1, 1).astype(np.float64)
        return self._calibrator.predict_proba(raw)[:, 1]
    
    def _safe_float(self, val: Any, default: float = 0.0) -> float:
        import math
        try:
            f = float(val)
            return default if math.isnan(f) or math.isinf(f) else f
        except (ValueError, TypeError):
            return default

    def _predict_stage2_class(self, features: Dict[str, Any]) -> int:
        """Return the encoded Stage 2 class using the loaded model probabilities."""
        if self._onnx_engine is None and self._pipeline is None:
            raise RuntimeError("Stage 2 model is not loaded")

        df = pd.DataFrame([features])
        if self._use_onnx:
            if self._preprocessor is not None:
                transformed = self._preprocessor.transform(df)
                feature_array = (
                    transformed.values.astype(np.float32)
                    if isinstance(transformed, pd.DataFrame)
                    else np.array(transformed, dtype=np.float32)
                )
            else:
                feature_array = df.values.astype(np.float32)
            outputs = self._onnx_engine.session.run(None, {self._onnx_engine.input_name: feature_array})
            probabilities = outputs[1][0] if len(outputs) > 1 else None
        else:
            probabilities = self._pipeline.predict_proba(df)[0]

        if probabilities is None:
            return int(self._pipeline.predict(df)[0])
        return int(np.argmax(probabilities))

    def _predict_stage2_direct(self, features: Dict[str, Any], start: float) -> Dict[str, Any]:
        """Return a direct anomaly-type classification for a loaded Stage 2 model."""
        df = pd.DataFrame([features])
        if self._use_onnx:
            if self._preprocessor is not None:
                transformed = self._preprocessor.transform(df)
                feature_array = (
                    transformed.values.astype(np.float32)
                    if isinstance(transformed, pd.DataFrame)
                    else np.array(transformed, dtype=np.float32)
                )
            else:
                feature_array = df.values.astype(np.float32)
            outputs = self._onnx_engine.session.run(None, {self._onnx_engine.input_name: feature_array})
            probabilities = outputs[1][0] if len(outputs) > 1 else None
        else:
            probabilities = self._pipeline.predict_proba(df)[0]

        if probabilities is None:
            stage2_class = int(self._pipeline.predict(df)[0])
            confidence = 1.0
        else:
            stage2_class = int(np.argmax(probabilities))
            confidence = self._safe_float(np.max(probabilities))

        return {
            "prediction": stage2_class,
            "fraud_score": confidence,
            "anomaly_score": confidence,
            "confidence": confidence,
            "risk_level": self._get_risk_level(confidence),
            "response_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "model_id": self._loaded_model_id,
            "anomaly_type": STAGE2_MODEL_CLASSES.get(stage2_class),
        }
    
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single prediction using ONNX or pickle pipeline."""
        if self._onnx_engine is None and self._pipeline is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        start = time.perf_counter()

        if self._training_stage == "anomaly_type":
            return self._predict_stage2_direct(features, start)

        df = pd.DataFrame([features])

        if self._use_onnx:
            # ONNX path
            if self._preprocessor is not None:
                transformed = self._preprocessor.transform(df)

                if isinstance(transformed, pd.DataFrame):
                    feature_array = transformed.values.astype(np.float32)
                else:
                    feature_array = np.array(transformed, dtype=np.float32)
            else:
                feature_array = df.values.astype(np.float32)

            result = self._onnx_engine.predict(feature_array)

            raw_score = self._safe_float(result.fraud_score)
            anomaly_score = self._safe_float(self._calibrate(raw_score))

            prediction = 1 if anomaly_score >= self._optimal_threshold else 0
            confidence = self._safe_float(abs(anomaly_score - 0.5) * 2)

        else:
            # Pickle pipeline path
            proba = self._pipeline.predict_proba(df)[0]

            raw_score = self._safe_float(
                proba[1] if len(proba) > 1 else proba[0]
            )
            anomaly_score = self._safe_float(self._calibrate(raw_score))

            prediction = 1 if anomaly_score >= self._optimal_threshold else 0
            confidence = self._safe_float(abs(anomaly_score - 0.5) * 2)

        # Risk level is computed once here (moved above the Stage 2 branch below) so it is
        # available regardless of whether Stage 2 runs. Value/formula unchanged from before.
        risk_level = self._get_risk_level(anomaly_score)

        if prediction == 1 and self._stage2_service is not None:
            # Stage 2: only runs when Stage 1 flags an anomaly (prediction == 1) and a
            # Stage 2 ("anomaly_type") model was discovered/loaded via load_model()'s
            # existing hyperparameters-based discovery. No Stage 1 logic above is touched.
            stage2_class = self._stage2_service._predict_stage2_class(features)
            result = {
                "prediction": prediction,
                "fraud_score": anomaly_score,
                "anomaly_score": anomaly_score,
                "confidence": confidence,
                "risk_level": risk_level,
                "response_time_ms": round((time.perf_counter() - start) * 1000, 2),
                "model_id": self._loaded_model_id,
                "anomaly_type": STAGE2_MODEL_CLASSES.get(stage2_class),
            }
            return result

        total_ms = (time.perf_counter() - start) * 1000

        return {
            "prediction": prediction,
            "fraud_score": anomaly_score,
            "anomaly_score": anomaly_score,
            "confidence": confidence,
            "risk_level": risk_level,
            "response_time_ms": round(total_ms, 2),
            "model_id": self._loaded_model_id,
            "anomaly_type": None,
        }
    
    def predict_batch(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions using ONNX or pickle pipeline."""
        if self._onnx_engine is None and self._pipeline is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        start = time.perf_counter()
        df = pd.DataFrame(transactions)
        
        # Data quality check: warn about rows with excessive missing values
        if len(df) > 0:
            missing_pct = df.isnull().sum(axis=1) / len(df.columns)
            problematic_rows = missing_pct[missing_pct > 0.5]
            
            if len(problematic_rows) > 0:
                logger.warning(
                    f"Found {len(problematic_rows)} rows with >50% missing values. "
                    f"These may produce unreliable predictions. Row indices: {problematic_rows.index.tolist()[:10]}"
                )
        
        if len(df) == 0:
            return {
                "results": [],
                "meta": {
                    "total_transactions": 0,
                    "fraud_count": 0,
                    "legit_count": 0,
                    "risk_summary": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
                    "total_amount": 0.0,
                    "avg_amount": 0.0,
                    "fraud_total_amount": 0.0,
                    "fraud_avg_amount": 0.0,
                    "all_transactions_amount": 0.0,
                    "all_transactions_avg_amount": 0.0,
                    "has_amount": False,
                    "total_time_ms": 0.0,
                    "avg_time_per_transaction_ms": 0.0,
                    "model_id": self._loaded_model_id,
                }
            }

        # Normalize columns to match expected features (case-insensitive fallback)
        if self._model_info and self._model_info.get("input_features"):
            expected = set(self._model_info["input_features"])
            mapping = {}
            for col in df.columns:
                if col in expected:
                    continue
                # Try case insensitive match
                match = next((e for e in expected if e.lower() == col.lower()), None)
                if match:
                    mapping[col] = match
            
            if mapping:
                logger.info(f"Normalizing input columns: {mapping}")
                df.rename(columns=mapping, inplace=True)
        
        if self._use_onnx:
            if self._preprocessor is not None:
                transformed = self._preprocessor.transform(df)
                if isinstance(transformed, pd.DataFrame):
                    feature_array = transformed.values.astype(np.float32)
                else:
                    feature_array = np.array(transformed, dtype=np.float32)
            else:
                feature_array = df.values.astype(np.float32)
            
            results = self._onnx_engine.predict_batch(feature_array)
            raw_scores = np.array([self._safe_float(r.fraud_score) for r in results])
            cal_scores = self._calibrate_batch(raw_scores)  # Platt-corrected probabilities
            formatted = []
            for i, (r, fs) in enumerate(zip(results, cal_scores)):
                fs = self._safe_float(fs)
                pred = 1 if fs >= self._optimal_threshold else 0
                formatted.append({
                    "index": i,
                    "prediction": pred,
                    "fraud_score": round(fs, 4),
                    "anomaly_score": round(fs, 4),
                    "confidence": round(self._safe_float(abs(fs - 0.5) * 2), 4),
                    "risk_level": self._get_risk_level(fs),
                })
        else:
            probas = self._pipeline.predict_proba(df)
            raw_scores = np.array([
                self._safe_float(probas[i][1] if probas.shape[1] > 1 else probas[i][0])
                for i in range(len(probas))
            ])
            cal_scores = self._calibrate_batch(raw_scores)  # Platt-corrected probabilities
            formatted = []
            for i, fs in enumerate(cal_scores):
                fs = self._safe_float(fs)
                pred = 1 if fs >= self._optimal_threshold else 0
                formatted.append({
                    "index": i,
                    "prediction": pred,
                    "fraud_score": round(fs, 4),
                    "anomaly_score": round(fs, 4),
                    "confidence": round(self._safe_float(abs(fs - 0.5) * 2), 4),
                    "risk_level": self._get_risk_level(fs),
                })
        
        # Calculate extended metrics
        fraud_count = sum(1 for r in formatted if r['prediction'] == 1)
        legit_count = len(formatted) - fraud_count
        
        risk_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        scores = []
        for r in formatted:
            risk_counts[r['risk_level']] = risk_counts.get(r['risk_level'], 0) + 1
            scores.append(r['anomaly_score'])
            
        # Financial impact logic
        total_amount = 0.0
        avg_amount = 0.0
        fraud_total_amount = 0.0
        fraud_avg_amount = 0.0
        all_transactions_amount = 0.0
        all_transactions_avg_amount = 0.0
        amount_col = self._get_amount_column(df)
        
        if amount_col:
            try:
                # Ensure numeric
                amounts = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
                all_transactions_amount = self._safe_float(amounts.sum())
                all_transactions_avg_amount = self._safe_float(amounts.mean())

                # Sum and average amount only for predicted fraud transactions.
                fraud_indices = [r["index"] for r in formatted if r["prediction"] == 1]
                if fraud_indices:
                    fraud_amounts = amounts.iloc[fraud_indices]
                    fraud_total_amount = self._safe_float(fraud_amounts.sum())
                    fraud_avg_amount = self._safe_float(fraud_amounts.mean())

                # Keep existing API keys for UI backward compatibility, but with fraud-only semantics.
                total_amount = fraud_total_amount
                avg_amount = fraud_avg_amount
            except Exception as e:
                logger.warning(f"Failed to calculate amount stats: {e}")

        total_ms = (time.perf_counter() - start) * 1000
        
        return {
            "results": formatted,
            "meta": {
                "total_transactions": len(formatted),
                "fraud_count": fraud_count,
                "legit_count": legit_count,
                "risk_summary": risk_counts,
                "total_amount": round(total_amount, 2),
                "avg_amount": round(avg_amount, 2),
                "fraud_total_amount": round(fraud_total_amount, 2),
                "fraud_avg_amount": round(fraud_avg_amount, 2),
                "all_transactions_amount": round(all_transactions_amount, 2),
                "all_transactions_avg_amount": round(all_transactions_avg_amount, 2),
                "has_amount": amount_col is not None,
                "total_time_ms": round(total_ms, 2),
                "avg_time_per_transaction_ms": round(total_ms / max(len(formatted), 1), 2),
                "model_id": self._loaded_model_id,
            }
        }

    def _get_amount_column(self, df: pd.DataFrame) -> Optional[str]:
        """Heuristic to find transaction amount column."""
        candidates = ['amount', 'transaction_amount', 'txn_amt', 'value', 'total_amount']
        # Check exact matches first
        for col in df.columns:
            if col.lower() in candidates:
                return col
        # Check partial matches
        for col in df.columns:
            if 'amount' in col.lower():
                return col
        return None
    
    def get_loaded_model_info(self) -> Optional[Dict[str, Any]]:
        """Return info about currently loaded model, or None."""
        return self._model_info
    
    @staticmethod
    def _get_risk_level(anomaly_score: float) -> str:
        if anomaly_score > 0.9:
            return "CRITICAL"
        elif anomaly_score > 0.7:
            return "HIGH"
        elif anomaly_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
