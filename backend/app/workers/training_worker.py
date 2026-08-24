"""
Training Worker
Background tasks for model training using SYNCHRONOUS database operations.
"""
from celery import shared_task
from celery.signals import worker_ready
from datetime import datetime, timezone, timedelta
from uuid import UUID
import logging

from app.core.time import now_ist
import os
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

from app.core.database_sync import SyncSessionLocal
from app.models.training_job import TrainingJob
from app.models.ml_model import MLModel
from app.models.dataset import Dataset
from app.core.storage import storage_service
from ml.algorithms.trainer import FraudDetectionTrainer, TrainingConfig
from ml.inference.onnx_converter import ONNXConverter
from ml.logging.job_log_handler import attach_job_logger, detach_job_logger
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.services.alert_service import AlertService

from app.workers.feature_worker import _resolve_local_storage_path

logger = logging.getLogger(__name__)


def _save_model_artifacts(job_id: str, result, algorithm: str, *, base_dir: str | os.PathLike[str] | None = None):
    """Persist model artifacts locally when Azure is unavailable; otherwise use Azure storage."""
    import json
    import pickle
    from pathlib import Path

    from app.core.config import get_settings

    settings = get_settings()
    backend_root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[2]
    azure_configured = bool((settings.AZURE_STORAGE_CONNECTION_STRING or "").strip())

    model_bytes = pickle.dumps(result.pipeline)
    feature_names = list(result.feature_names)
    onnx_result = None
    onnx_storage_path = None

    if azure_configured:
        container_client = storage_service.client.get_container_client("models")
        if not container_client.exists():
            container_client.create_container()

        model_path = f"{job_id}/model.pkl"
        blob_client = container_client.get_blob_client(model_path)
        blob_client.upload_blob(model_bytes, overwrite=True)
        model_storage_path = f"models/{model_path}"

        _ONNX_SIZE_LIMIT_BYTES = 50 * 1024 * 1024
        model_size_bytes = len(model_bytes)
        if model_size_bytes <= _ONNX_SIZE_LIMIT_BYTES:
            raw_model = result.pipeline.named_steps["model"]
            converter = ONNXConverter()
            onnx_result = converter.convert(raw_model, feature_names)
            onnx_path = f"{job_id}/model.onnx"
            blob_client = container_client.get_blob_client(onnx_path)
            blob_client.upload_blob(onnx_result.onnx_model, overwrite=True)
            onnx_storage_path = f"models/{onnx_path}"

        preprocessor = result.pipeline.named_steps["fraud_features"]
        preprocessor_bytes = pickle.dumps(preprocessor)
        preprocessor_path = f"{job_id}/preprocessor.pkl"
        blob_client = container_client.get_blob_client(preprocessor_path)
        blob_client.upload_blob(preprocessor_bytes, overwrite=True)

        calibrator_storage_path = None
        if result.calibrator is not None:
            calibrator_bytes = pickle.dumps(result.calibrator)
            calibrator_path = f"{job_id}/calibrator.pkl"
            blob_client = container_client.get_blob_client(calibrator_path)
            blob_client.upload_blob(calibrator_bytes, overwrite=True)
            calibrator_storage_path = f"models/{calibrator_path}"

        manifest = {
            "training_job_id": job_id,
            "algorithm": algorithm,
            "feature_count": len(feature_names),
            "onnx_checksum": onnx_result.checksum if onnx_result is not None else None,
            "optimal_threshold": result.optimal_threshold,
            "is_calibrated": result.calibrator is not None,
            "created_at": now_ist().isoformat(),
        }
        manifest_path = f"{job_id}/manifest.json"
        blob_client = container_client.get_blob_client(manifest_path)
        blob_client.upload_blob(json.dumps(manifest), overwrite=True)
        manifest_storage_path = f"models/{manifest_path}"
        logger.info(f"Artifact manifest saved for job {job_id} (threshold={result.optimal_threshold:.4f})")
        return {
            "model_storage_path": model_storage_path,
            "onnx_storage_path": onnx_storage_path,
            "preprocessor_storage_path": f"models/{job_id}/preprocessor.pkl",
            "calibrator_storage_path": calibrator_storage_path,
            "manifest_storage_path": manifest_storage_path,
        }

    artifact_dir = backend_root / "storage" / "models" / str(job_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_file = artifact_dir / "model.pkl"
    model_file.write_bytes(model_bytes)
    model_storage_path = str(model_file.relative_to(backend_root)).replace("\\", "/")

    _ONNX_SIZE_LIMIT_BYTES = 50 * 1024 * 1024
    model_size_bytes = len(model_bytes)
    if model_size_bytes <= _ONNX_SIZE_LIMIT_BYTES:
        raw_model = result.pipeline.named_steps["model"]
        converter = ONNXConverter()
        onnx_result = converter.convert(raw_model, feature_names)
        onnx_file = artifact_dir / "model.onnx"
        onnx_file.write_bytes(onnx_result.onnx_model)
        onnx_storage_path = str(onnx_file.relative_to(backend_root)).replace("\\", "/")
    else:
        logger.warning(
            f"[ONNX] Skipping conversion — model is {model_size_bytes / 1e6:.0f} MB "
            f"(limit {_ONNX_SIZE_LIMIT_BYTES / 1e6:.0f} MB). Inference will use the .pkl model instead."
        )

    preprocessor = result.pipeline.named_steps["fraud_features"]
    preprocessor_file = artifact_dir / "preprocessor.pkl"
    preprocessor_file.write_bytes(pickle.dumps(preprocessor))
    preprocessor_storage_path = str(preprocessor_file.relative_to(backend_root)).replace("\\", "/")

    calibrator_storage_path = None
    if result.calibrator is not None:
        calibrator_file = artifact_dir / "calibrator.pkl"
        calibrator_file.write_bytes(pickle.dumps(result.calibrator))
        calibrator_storage_path = str(calibrator_file.relative_to(backend_root)).replace("\\", "/")

    manifest = {
        "training_job_id": job_id,
        "algorithm": algorithm,
        "feature_count": len(feature_names),
        "onnx_checksum": onnx_result.checksum if onnx_result is not None else None,
        "optimal_threshold": result.optimal_threshold,
        "is_calibrated": result.calibrator is not None,
        "created_at": now_ist().isoformat(),
    }
    manifest_file = artifact_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_storage_path = str(manifest_file.relative_to(backend_root)).replace("\\", "/")

    logger.info(f"Model saved locally to: {model_storage_path}")
    return {
        "model_storage_path": model_storage_path,
        "onnx_storage_path": onnx_storage_path,
        "preprocessor_storage_path": preprocessor_storage_path,
        "calibrator_storage_path": calibrator_storage_path,
        "manifest_storage_path": manifest_storage_path,
    }


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numerics to reduce memory usage by ~50%.
    float64 → float32, int64 → int32.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype(np.int32)
    return df


def _compute_dynamic_timeout(n_rows: int) -> int:
    """
    Dynamic soft time limit: 2700s base + 60s per 100K rows, capped at 3 hours.
    Prevents Celery killing large-dataset training jobs prematurely.
    """
    extra = int((n_rows / 100_000) * 60)
    return min(2700 + extra, 10800)


def _set_redis_status(job_id: str, status: str) -> None:
    """Write job terminal status to Redis so SSE stream can stop."""
    try:
        import redis
        from app.core.config import settings
        r = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=2)
        r.set(f"training:status:{job_id}", status, ex=86400)
        r.close()
    except Exception as exc:
        logger.warning(f"Could not set Redis status key for job {job_id}: {exc}")


def _safe_uuid_user_id(user_id: str | None) -> str | None:
    """Normalize user_id for UUID-backed created_by columns."""
    if not user_id:
        return None
    try:
        return str(UUID(str(user_id)))
    except (ValueError, TypeError):
        logger.warning("Invalid non-UUID created_by=%r; storing NULL for model row", user_id)
        return None


async def _send_training_success_notification(
    model_id: str,
    job_id: str,
    algorithm: str,
    duration: float,
    metrics: dict,
    event_time=None,
):
    """Async helper to send Slack notification via AlertService."""
    DATABASE_URL = os.environ.get("DATABASE_URL", "")
    if not DATABASE_URL:
        return

    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as db:
            service = AlertService(db)
            await service.create_training_success_alert(
                model_id=model_id,
                job_id=job_id,
                algorithm=algorithm,
                duration=duration,
                metrics=metrics,
                event_time=event_time,
            )
    except Exception as e:
        logger.error(f"Failed to send training success notification: {e}")
    finally:
        await engine.dispose()


@worker_ready.connect
def cleanup_stale_running_jobs(sender, **kwargs):
    """On worker startup: mark stale RUNNING jobs as FAILED."""
    try:
        db = SyncSessionLocal()
        try:
            stale = db.query(TrainingJob).filter(TrainingJob.status == "RUNNING").all()
            if stale:
                logger.warning(f"[Startup] Found {len(stale)} stale RUNNING job(s) — marking as FAILED.")
                for job in stale:
                    job.status = "FAILED"
                    job.error_message = (
                        "Worker process was restarted while this job was running. "
                        "Please resubmit the training job."
                    )
                    job.completed_at = now_ist().replace(tzinfo=None)
                db.commit()
                logger.info(f"[Startup] Marked {len(stale)} stale job(s) as FAILED.")
            else:
                logger.info("[Startup] No stale RUNNING jobs found.")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"[Startup] Failed to clean up stale jobs: {e}")


@shared_task(bind=True, max_retries=2, soft_time_limit=7200, time_limit=10800, name="app.workers.training_worker.train_model")
def train_model(self, job_id: str):
    """
    Train a machine learning model using pre-split data.

    Steps:
    1. Load training job from database
    2. Load pre-split train/test data from Azure Blob (Synchronously)
    3. Validate data quality
    4. Train model using FraudDetectionTrainer
    5. Save model artifacts to Azure Blob
    6. Register model in database
    7. Update job status to COMPLETED
    """
    # Attach per-job Redis log handler — captures all logger.* calls in this task
    _log_handler = attach_job_logger(job_id)
    try:
        logger.info(f"Starting training job {job_id}")

        # ========================================
        # STEP 1: Load Job from Database (SYNC)
        # ========================================
        db = SyncSessionLocal()
        try:
            # Retry logic for job loading (handling potential race conditions)
            import time
            for _ in range(3):
                job = db.query(TrainingJob).filter(TrainingJob.id == UUID(job_id)).first()
                if job:
                    break
                logger.warning(f"Training job {job_id} not found. Retrying in 1s...")
                time.sleep(1)

            if not job:
                raise ValueError(f"Training job {job_id} not found after 3 attempts")

            # Extract configuration
            algorithm = job.algorithm
            hyperparameters = job.hyperparameters
            feature_config = job.feature_config
            tuning_method = job.tuning_method or "manual"
            tuning_config = job.tuning_config or {}
            train_dataset_id = job.metrics.get("train_dataset_id")
            test_dataset_id = job.metrics.get("test_dataset_id")

            logger.info(f"Job config: algorithm={algorithm}, hyperparameters={hyperparameters}")

            # Update status to RUNNING
            job.status = "RUNNING"
            job.progress = 0.1
            job.started_at = now_ist().replace(tzinfo=None)
            db.commit()
        finally:
            db.close()

        # ========================================
        # STEP 2: Load Pre-Split Data from Azure Blob (SYNC)
        # ========================================
        logger.info(f"Loading train data: {train_dataset_id}")

        # Helper for sync download
        def download_blob_sync(storage_path_uri):
            """Download dataset from local storage or Azure Blob Storage."""
            try:
                from pathlib import Path

                # Local Windows/Linux storage path
                local_path = Path(storage_path_uri)
                if local_path.exists() and local_path.is_file():
                    logger.info(f"Loading dataset from local storage: {local_path}")
                    return local_path.read_bytes()

                # Azure Blob Storage path: container/blob
                parts = storage_path_uri.split("/", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid storage path: {storage_path_uri}")

                container_name, blob_name = parts

                # Use the underlying sync client directly
                blob_client = storage_service.client.get_blob_client(
                    container=container_name,
                    blob=blob_name
                )

                return blob_client.download_blob().readall()

            except Exception as e:
                logger.error(f"Failed to download {storage_path_uri}: {e}")
                raise

        # Load train dataset
        db = SyncSessionLocal()
        try:
            train_dataset = db.query(Dataset).filter(Dataset.id == UUID(train_dataset_id)).first()
            if not train_dataset:
                raise ValueError(f"Train dataset {train_dataset_id} not found")

            # Download train data synchronously
            train_blob = download_blob_sync(train_dataset.storage_path)
            train_df = pd.read_parquet(BytesIO(train_blob))
            logger.info(f"Loaded train data: {train_df.shape}")
            mem_before = train_df.memory_usage(deep=True).sum() / 1e6
            train_df = _optimise_dtypes(train_df)
            mem_after = train_df.memory_usage(deep=True).sum() / 1e6
            logger.info(f"Train memory: {mem_before:.0f} MB → {mem_after:.0f} MB after dtype optimisation")
        finally:
            db.close()

        logger.info(f"Loading test data: {test_dataset_id}")

        # Load test dataset
        db = SyncSessionLocal()
        try:
            test_dataset = db.query(Dataset).filter(Dataset.id == UUID(test_dataset_id)).first()
            if not test_dataset:
                raise ValueError(f"Test dataset {test_dataset_id} not found")

            # Download test data synchronously
            test_blob = download_blob_sync(test_dataset.storage_path)
            test_df = pd.read_parquet(BytesIO(test_blob))
            test_df = _optimise_dtypes(test_df)
            logger.info(f"Loaded test data: {test_df.shape}")
        finally:
            db.close()

        # ========================================
        # STEP 3: Data Validation & Preparation
        # ========================================

        # Detect target column dynamically
        from ml.transformers.column_role_detector import ColumnRoleDetector

        detector = ColumnRoleDetector()
        roles = detector.detect(train_df)

        if roles.target_col:
            target_col = roles.target_col
        else:
            # Fallback
            common_targets = ['is_fraud', 'fraud_label', 'target', 'label', 'class', 'is_fraudulent']
            target_col = next((c for c in common_targets if c in train_df.columns), None)

            if not target_col:
                raise ValueError(
                    f"No target column found. Columns: {list(train_df.columns)}"
                )

        logger.info(f"Using target column: '{target_col}'")

        # Drop rows where target is NaN
        train_df = train_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])

        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]

        # Determine training stage.
        training_stage = hyperparameters.get("training_stage", "binary")

        if training_stage == "anomaly_type":
            # Stage 2: anomaly-type classification.
            # Remove normal samples and keep the original anomaly labels.

            y_train = y_train.astype(str).str.strip()
            y_test = y_test.astype(str).str.strip()

            # Remove normal class: 3 / No Anomaly
            train_mask = ~y_train.isin(["3", "No Anomaly"])
            test_mask = ~y_test.isin(["3", "No Anomaly"])

            X_train = X_train.loc[train_mask].reset_index(drop=True)
            y_train = y_train.loc[train_mask].reset_index(drop=True)

            X_test = X_test.loc[test_mask].reset_index(drop=True)
            y_test = y_test.loc[test_mask].reset_index(drop=True)

            # Encode anomaly-type names for XGBoost
            from sklearn.preprocessing import LabelEncoder

            label_encoder = LabelEncoder()

            combined_labels = pd.concat([y_train, y_test]).astype(str).str.strip()
            label_encoder.fit(combined_labels)

            y_train = pd.Series(
                      label_encoder.transform(y_train.astype(str).str.strip()),
                      index=y_train.index
            )
            y_test = pd.Series(
                     label_encoder.transform(y_test.astype(str).str.strip()),
                     index=y_test.index
    )

            logger.info(
                f"Stage 2 LabelEncoder classes: {label_encoder.classes_.tolist()}"
            )

            logger.info(
                f"Stage 2 anomaly-type labels: "
                f"train={pd.Series(y_train).value_counts().sort_index().to_dict()}, "
                f"test={pd.Series(y_test).value_counts().sort_index().to_dict()}"
            )
           

            logger.info(
                f"Stage 2 anomaly-type labels: "
                f"train={y_train.value_counts().sort_index().to_dict()}, "
                f"test={y_test.value_counts().sort_index().to_dict()}"
            )

        else:
            # Stage 1 — existing binary behavior.
            def _coerce_labels(s: pd.Series) -> pd.Series:
                return s.astype(str).str.strip().apply(
                    lambda x: 0 if x in ("3", "No Anomaly") else 1
                ).astype(int)

            y_train = _coerce_labels(y_train)
            y_test = _coerce_labels(y_test)

            logger.info(
                f"Stage 1 binary labels: "
                f"train={y_train.value_counts().sort_index().to_dict()}, "
                f"test={y_test.value_counts().sort_index().to_dict()}"
            )
        # --- Keep all columns for feature engineering ---
        # The FraudFeatureEngineer handles encoding of categorical and datetime columns.
        # We no longer drop non-numeric columns here.
        logger.info("Retaining all columns for feature engineering; no pre-drop applied.")
        logger.info(f"Final Features: {X_train.shape}, Labels: {y_train.shape}")
        logger.info(f"Feature columns: {list(X_train.columns)}")

        # Compute dynamic timeout based on dataset size and update Celery soft limit
        _dynamic_timeout = _compute_dynamic_timeout(len(X_train))
        logger.info(f"Dataset: {len(X_train):,} train rows | Dynamic soft_time_limit: {_dynamic_timeout}s")

        # Update progress
        db = SyncSessionLocal()
        try:
            job = db.query(TrainingJob).filter(TrainingJob.id == UUID(job_id)).first()
            job.progress = 0.3
            db.commit()
            _job_started_at = job.started_at
        finally:
            db.close()

        # ========================================
        # STEP 4: Train Model
        # ========================================
        logger.info("Initializing trainer...")

        config = TrainingConfig(
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            imbalanced_strategy=hyperparameters.get("imbalanced_strategy", "class_weight"),
            feature_config=feature_config,
            tuning_method=tuning_method,
            tuning_config=tuning_config
        )

        trainer = FraudDetectionTrainer(config)

        print("========== DEBUG BEFORE TRAINING ==========")
        print("DEBUG y_train unique:", y_train.unique())
        print("DEBUG y_train counts:", y_train.value_counts().to_dict())
        print("DEBUG y_test unique:", y_test.unique())
        print("DEBUG y_test counts:", y_test.value_counts().to_dict())
        print("==========================================")

        logger.info(
            f"AFTER BINARY CONVERSION - Train labels: "
            f"{y_train.value_counts().to_dict()}"
        )
        logger.info(
            f"AFTER BINARY CONVERSION - Test labels: "
            f"{y_test.value_counts().to_dict()}"
        )

        logger.info("Training model with pre-split data...")
        result = trainer.train_with_splits(X_train, y_train, X_test, y_test)

        logger.info(f"Training complete! Metrics: {result.metrics}")

        # ── Extract ACTUAL model parameters via get_params() ──────────────────
        actual_model_params = {}
        try:
            trained_model = result.pipeline.named_steps.get("model")
            if trained_model is not None and hasattr(trained_model, "get_params"):
                raw = trained_model.get_params()
                # Reverse the reg_lambda / reg_alpha rename so the registry
                # stores the user-facing names ("lambda" / "alpha").
                if "reg_lambda" in raw:
                    raw["lambda"] = raw.pop("reg_lambda")
                if "reg_alpha" in raw:
                    raw["alpha"] = raw.pop("reg_alpha")

                # ── Sanitize for JSON: PostgreSQL rejects Python's NaN / Inf ──
                import math
                def _json_safe(v):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        return None
                    if hasattr(v, "tolist"):   # numpy scalars/arrays
                        v = v.tolist()
                        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                            return None
                    return v

                actual_model_params = {k: _json_safe(v) for k, v in raw.items()}
                logger.info(f"[Worker] Actual model params (get_params): {actual_model_params}")
        except Exception as gp_err:
            logger.warning(f"[Worker] Could not read model.get_params(): {gp_err}")
            actual_model_params = hyperparameters  # fallback to user input

        # Log duration (both naive for subtraction)
        started_at = _job_started_at
        if started_at:
            started_naive = started_at.replace(tzinfo=None) if started_at.tzinfo else started_at
            duration = (now_ist() - started_naive).total_seconds()
        else:
            duration = 0

        # Update progress
        db = SyncSessionLocal()
        try:
            job = db.query(TrainingJob).filter(TrainingJob.id == UUID(job_id)).first()
            job.progress = 0.7
            db.commit()
        finally:
            db.close()

        # ========================================
        # STEP 5: Save Model Artifacts
        # ========================================
        logger.info("Saving model artifacts...")

        artifact_info = _save_model_artifacts(job_id, result, algorithm)
        full_path = artifact_info["model_storage_path"]
        onnx_full_path = artifact_info["onnx_storage_path"]

        # Update progress
        db = SyncSessionLocal()
        try:
            job = db.query(TrainingJob).filter(TrainingJob.id == UUID(job_id)).first()
            job.progress = 0.85
            db.commit()
        finally:
            db.close()

        # ========================================
        # STEP 6: Register Model
        # ========================================
        logger.info("Registering model in database...")

        db = SyncSessionLocal()
        try:
            # Auto-increment version per algorithm: 1.0.0, 1.1.0, 1.2.0...
            existing_count = db.query(MLModel).filter(
                MLModel.algorithm == algorithm
            ).count()
            model_version = f"1.{existing_count}.0"
            logger.info(f"Assigning model version {model_version} (existing {algorithm} models: {existing_count})")

            ml_model = MLModel(
                name=f"{job.name} ({algorithm.upper()})",
                algorithm=algorithm,
                version=model_version,
                storage_path=full_path,
                onnx_path=onnx_full_path,
                feature_names=list(result.feature_names),
                metrics=result.metrics,
                hyperparameters=actual_model_params,
                feature_importance=result.feature_importance,
                status="STAGING",
                created_at=now_ist(),
                created_by=_safe_uuid_user_id(job.created_by)
            )
            db.add(ml_model)
            db.commit()
            db.refresh(ml_model)
            model_id = ml_model.id
        finally:
            db.close()

        # ========================================
        # STEP 7: Update Job Status
        # ========================================
        logger.info("Updating job status...")

        db = SyncSessionLocal()
        try:
            job = db.query(TrainingJob).filter(TrainingJob.id == UUID(job_id)).first()
            job.status = "COMPLETED"
            job.progress = 1.0
            job.completed_at = now_ist().replace(tzinfo=None)

            completed = job.completed_at
            started = job.started_at.replace(tzinfo=None) if job.started_at and job.started_at.tzinfo else job.started_at
            training_duration = (completed - started).total_seconds() if started else 0

            existing_metrics = job.metrics or {}
            job.metrics = {
                **existing_metrics,
                "precision": result.metrics["precision"],
                "recall": result.metrics["recall"],
                "f1": result.metrics["f1"],
                "auc": result.metrics["auc"],
                "accuracy": result.metrics["accuracy"],
                "training_duration_seconds": training_duration,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_count": X_train.shape[1],
                "input_hyperparameters": hyperparameters,
                "actual_model_params": actual_model_params,
            }
            job.model_id = model_id
            db.commit()
        finally:
            db.close()

        # ========================================
        # STEP 8: Auto-Apply Default Baselines
        # ========================================
        logger.info(f"Applying default baselines for model {model_id}...")
        try:
            from app.models.ml_model import Baseline
            DEFAULT_BASELINES = [
                {"metric_name": "precision", "threshold": 0.85, "operator": "gte"},
                {"metric_name": "recall",    "threshold": 0.80, "operator": "gte"},
                {"metric_name": "f1",        "threshold": 0.82, "operator": "gte"},
                {"metric_name": "auc",       "threshold": 0.90, "operator": "gte"},
                {"metric_name": "fpr",       "threshold": 0.10, "operator": "lte"},
            ]
            db = SyncSessionLocal()
            try:
                for cfg in DEFAULT_BASELINES:
                    db.add(Baseline(
                        model_id=model_id,
                        metric_name=cfg["metric_name"],
                        threshold=cfg["threshold"],
                        operator=cfg["operator"],
                    ))
                db.commit()
                logger.info(f"Applied {len(DEFAULT_BASELINES)} default baselines for model {model_id}")
            finally:
                db.close()
        except Exception as baseline_err:
            logger.warning(f"Failed to apply default baselines (non-blocking): {baseline_err}")

        logger.info(f"Training job {job_id} completed successfully!")
        _set_redis_status(job_id, "COMPLETED")

        # ========================================
        # STEP 9: Send Notification
        # ========================================
        try:
            notification_metrics = {
                "f1": result.metrics["f1"],
                "precision": result.metrics["precision"],
                "recall": result.metrics["recall"]
            }
            asyncio.run(_send_training_success_notification(
                model_id=str(model_id),
                job_id=job_id,
                algorithm=algorithm,
                duration=training_duration,
                metrics=notification_metrics,
                event_time=job.completed_at,
            ))
        except Exception as notify_err:
            logger.warning(f"Failed to trigger success notification (non-blocking): {notify_err}")

    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {e}", exc_info=True)

        retries_left = self.max_retries - self.request.retries
        if retries_left > 0:
            # Retries remaining — let Celery re-queue
            logger.warning(
                f"Job {job_id} will retry (attempt {self.request.retries + 1}/{self.max_retries}) "
                f"in 120s. Error: {e}"
            )
            # Use a plain Exception wrapper to avoid non-picklable DB/driver exceptions
            # causing Celery deserialization failures on retry.
            raise self.retry(exc=Exception(str(e)), countdown=120)

        # All retries exhausted — mark terminal FAILED and do not re-raise
        logger.error(f"Job {job_id} exhausted all retries. Marking internally as FAILED.")
        _set_redis_status(job_id, "FAILED")
        db = SyncSessionLocal()
        try:
            job = db.query(TrainingJob).filter(TrainingJob.id == UUID(job_id)).first()
            if job:
                job.status = "FAILED"
                job.error_message = str(e)
                job.completed_at = now_ist().replace(tzinfo=None)
                db.commit()
        finally:
            db.close()

        # We do NOT `raise` here. Re-raising uncaught exceptions when max_retries
        # is hit causes the worker to loop indefinitely under some configurations.

    finally:
        detach_job_logger(_log_handler)