"""
Feature Engineering Worker
Background tasks for feature computation and selection.
"""
from celery import shared_task
from datetime import datetime
import logging
import time
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_local_storage_path(storage_path: str | None, base_dir: str | os.PathLike[str] | None = None) -> Path | None:
    """Resolve a dataset path stored in the DB to a real local file.

    Local storage paths from the API are relative to the backend root, but may also be
    returned with Windows backslashes or a raw absolute path. This helper normalizes the
    common local-development variants without affecting Azure blob paths.
    """
    if not storage_path:
        return None

    normalized = str(storage_path).replace("\\", "/")
    if os.path.isabs(normalized):
        candidates = [Path(normalized)]
    else:
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
        base_dir = Path(base_dir)
        candidates = [
            base_dir / normalized,
            Path.cwd() / normalized,
            base_dir / normalized.lstrip("./"),
            Path.cwd() / normalized.lstrip("./"),
        ]

    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except Exception:
            resolved = candidate
        if resolved.exists() and resolved.is_file():
            return resolved

    return None


def _get_azure_connection_string() -> str:
    """Return a configured Azure connection string or raise before any Azure client is created."""
    from app.core.config import get_settings

    settings = get_settings()
    azure_conn = (settings.AZURE_STORAGE_CONNECTION_STRING or "").strip()
    if not azure_conn:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not configured")
    return azure_conn


def _read_dataset_bytes(storage_path: str | None, *, file_format: str | None = None, base_dir: str | os.PathLike[str] | None = None) -> bytes:
    """Read dataset bytes from a local path when present; otherwise require Azure config."""
    if not storage_path:
        raise ValueError("Dataset storage path is required")

    local_path = _resolve_local_storage_path(storage_path, base_dir=base_dir)
    if local_path is not None:
        return local_path.read_bytes()

    if "/" not in storage_path:
        raise ValueError(f"Invalid storage path: {storage_path}")

    from azure.storage.blob import BlobServiceClient

    azure_conn = _get_azure_connection_string()
    container_name, blob_path = storage_path.split("/", 1)
    blob_service = BlobServiceClient.from_connection_string(azure_conn)
    container_client = blob_service.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_path)
    return blob_client.download_blob().readall()


def _set_redis_status(job_id: str, status: str) -> None:
    """Write job terminal status to Redis so SSE stream can stop."""
    try:
        import redis
        url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        r = redis.from_url(url, decode_responses=True, socket_timeout=2)
        r.set(f"feature:status:{job_id}", status, ex=86400)
        r.close()
    except Exception as exc:
        logger.warning(f"Could not set Redis status key for job {job_id}: {exc}")



@shared_task(bind=True, max_retries=3, name="app.workers.feature_worker.compute_features")
def compute_features(self, job_id: str):
    """
    Compute features for a dataset.
    
    Steps:
    1. Load dataset from storage
    2. Apply feature engineering pipeline
    3. Apply feature selection (if enabled)
    4. Save features to storage
    5. Cache features in Redis
    6. Update job status
    """
    from app.core.database_sync import SyncSessionLocal
    from app.services.feature_service import FeatureService
    from app.services.data_service import DataService
    from ml.transformers.fraud_feature_engineer import FraudFeatureEngineer
    import pandas as pd
    import io
    from azure.storage.blob import BlobServiceClient
    from app.core.config import get_settings
    
    settings = get_settings()
    
    # Use sync database session
    with SyncSessionLocal() as db:
        # Create service instances with sync session
        feature_service = FeatureService(db)
        data_service = DataService(db)
        
        try:
            start_time = time.time()
            logger.info(f"Starting feature computation for job {job_id}")

            # Update status to Running
            feature_service.update_feature_set_status_sync(job_id, "RUNNING", progress=0.0)

            # 1. Load dataset from metadata
            feature_set = feature_service.get_feature_set_sync(job_id)
            if not feature_set:
                raise ValueError(f"Feature set {job_id} not found")
            
            dataset_id = str(feature_set.dataset_id)
            logger.info(f"Loading dataset {dataset_id} for feature computation")

            # 2. Resolve the dataset path from the project record and prefer the local file system
            # when the dataset is stored locally. Azure should only be used when a valid connection
            # string is configured and the dataset is not already present on disk.
            dataset = data_service.get_dataset_sync(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")

            dataset_storage_path = dataset.storage_path or ""
            local_dataset_path = _resolve_local_storage_path(dataset_storage_path)
            if local_dataset_path is not None:
                logger.info(f"Loading source dataset from local storage: {local_dataset_path}")
                blob_data = local_dataset_path.read_bytes()
                use_local_storage = True
            else:
                use_local_storage = False
                azure_conn = _get_azure_connection_string()
                blob_service = BlobServiceClient.from_connection_string(azure_conn)
                dataset_container = settings.AZURE_STORAGE_CONTAINER_DATASETS
                features_container = settings.AZURE_STORAGE_CONTAINER_FEATURES
                storage_path_parts = dataset_storage_path.split("/", 1)
                if len(storage_path_parts) == 2:
                    _, blob_path = storage_path_parts
                else:
                    blob_path = dataset_storage_path

                datasets_container_client = blob_service.get_container_client(dataset_container)
                features_container_client = blob_service.get_container_client(features_container)

                from azure.core.exceptions import ResourceNotFoundError
                try:
                    features_container_client.get_container_properties()
                except ResourceNotFoundError:
                    logger.info(f"Creating container: {features_container}")
                    features_container_client.create_container()
                blob_client = datasets_container_client.get_blob_client(blob_path)
                blob_data = blob_client.download_blob().readall()
            
            # Load based on file format
            if dataset.file_format == "parquet":
                df = pd.read_parquet(io.BytesIO(blob_data))
            elif dataset.file_format == "json":
                df = pd.read_json(io.BytesIO(blob_data))
            else:
                try:
                    df = pd.read_csv(io.BytesIO(blob_data))
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(io.BytesIO(blob_data), encoding='cp1252')
                    except UnicodeDecodeError:
                        df = pd.read_csv(io.BytesIO(blob_data), encoding='ISO-8859-1')

            logger.info(f"Dataset loaded successfully: {len(df)} rows, columns: {list(df.columns)}")
            feature_service.update_feature_set_status_sync(job_id, "RUNNING", progress=0.2)

            # 3. Detect target column using ColumnRoleDetector
            from ml.transformers.column_role_detector import ColumnRoleDetector
            detector = ColumnRoleDetector()
            
            # DIAGNOSTIC: Log all columns before detection
            logger.info(f"[DIAGNOSTIC] DataFrame columns before detection: {list(df.columns)}")
            logger.info(f"[DIAGNOSTIC] DataFrame shape: {df.shape}")
            logger.info(f"[DIAGNOSTIC] Column mapping config: {feature_set.config.get('column_mapping')}")
            
            roles = detector.detect(df, column_mapping=feature_set.config.get("column_mapping"))
            
            # DIAGNOSTIC: Log detection results
            logger.info(f"[DIAGNOSTIC] Detected target_col: '{roles.target_col}'")
            logger.info(f"[DIAGNOSTIC] Detected amount_col: '{roles.amount_col}'")
            logger.info(f"[DIAGNOSTIC] Detected timestamp_col: '{roles.timestamp_col}'")

            target_series = None
            if roles.target_col and roles.target_col in df.columns:
                logger.info(f"✓ Using target column '{roles.target_col}' for supervised feature engineering")
                target_series = df[roles.target_col]
                logger.info(f"[DIAGNOSTIC] target_series setup: name='{target_series.name}', length={len(target_series)}, dtype={target_series.dtype}")
            else:
                logger.warning(f"✗ No target column detected. Supervised features will use defaults.")
                logger.warning(f"[DIAGNOSTIC] Detection result: target_col='{roles.target_col}', exists_in_df={roles.target_col in df.columns if roles.target_col else 'N/A'}")

            # 4. Apply generic feature engineering (config-driven, auto-detection)
            logger.info("Applying FraudFeatureEngineer transformer (generic, role-based)")
            transformer = FraudFeatureEngineer(config=feature_set.config)

            # Fit and transform on raw DataFrame — no manual column renaming
            transformer.fit(df, y=target_series)
            df_transformed = transformer.transform(df)
            
            logger.info(f"[DIAGNOSTIC] After transformation: {df_transformed.shape}, columns={list(df_transformed.columns)[:10]}...")

            logger.info(f"Created {len(df_transformed.columns)} features")
            feature_service.update_feature_set_status_sync(job_id, "RUNNING", progress=0.6)

            # 5. Save Transformed features to blob storage (hierarchical structure)
            # Phase 2: Use dataset type and split info for hierarchical storage
            dataset_type = dataset.dataset_type or "raw"
            split_type = dataset.split_type
            
            # Build hierarchical path based on dataset type and split type
            if dataset_type == 'split' and split_type:
                # User requirement: "feature engineered data for both trained and test seperately"
                # Structure: features/{split_type}/{job_id}/features.parquet
                output_blob_path = f"features/{split_type}/{job_id}/features.parquet"
                transformer_blob_path = f"features/{split_type}/{job_id}/transformer.pkl"
                logger.info(f"Saving features for {split_type} split to hierarchical storage: {output_blob_path}")
            else:
                # For raw/merged datasets (no split), put in 'features/raw' or just 'features/{job_id}'
                # Defaulting to features/raw for consistency
                output_blob_path = f"features/raw/{job_id}/features.parquet"
                transformer_blob_path = f"features/raw/{job_id}/transformer.pkl"
                logger.info(f"Saving features for {dataset_type} dataset to hierarchical storage: {output_blob_path}")
            
            # Save Feature Data (Parquet)
            # Save Feature Data (Parquet)
            # IMPORTANT: Re-attach target column for downstream training
            logger.info(f"[DIAGNOSTIC] Checking target_series before attachment: Is None? {target_series is None}")

            if target_series is not None:
                # Ensure index alignment (transformer resets index)
                y_aligned = target_series.reset_index(drop=True)
                # Use original target name if available, otherwise 'target'
                target_col_name = roles.target_col or "target"
                df_transformed[target_col_name] = y_aligned
                logger.info(f"✓ SUCCESS: Attached target column '{target_col_name}' to features (shape now: {df_transformed.shape})")
                logger.info(f"[DIAGNOSTIC] Final columns in dataframe: {list(df_transformed.columns)}")

                # Double check content
                if target_col_name in df_transformed.columns:
                     logger.info(f"[DIAGNOSTIC] Verified: '{target_col_name}' exists in df_transformed.")
                else:
                     logger.error(f"[DIAGNOSTIC] CRITICAL ERROR: '{target_col_name}' NOT found in df_transformed after assignment!")
            else:
                logger.warning(f"✗ FAILURE: target_series is None - NOT attaching to features!")
                logger.warning(f"[DIAGNOSTIC] This means the target column was not detected in the input data.")


            output_buffer = io.BytesIO()
            for col in df_transformed.select_dtypes(include=['object']).columns:
                df_transformed[col] = df_transformed[col].astype(str)
            df_transformed.to_parquet(output_buffer, index=False, engine='pyarrow')
            output_buffer.seek(0)

            output_storage_path = output_blob_path
            transformer_storage_path = transformer_blob_path
            if use_local_storage:
                backend_root = Path(__file__).resolve().parents[2]
                local_feature_dir = backend_root / "storage" / "features" / dataset_type / str(job_id)
                local_feature_dir.mkdir(parents=True, exist_ok=True)
                local_feature_file = local_feature_dir / "features.parquet"
                local_feature_file.write_bytes(output_buffer.getvalue())
                output_storage_path = str(local_feature_file.relative_to(backend_root)).replace("\\", "/")
                logger.info(f"Saved features locally to path='{output_storage_path}'")
            else:
                output_blob_client = features_container_client.get_blob_client(output_blob_path)
                output_blob_client.upload_blob(output_buffer, overwrite=True)
                logger.info(f"Saved features to container='{settings.AZURE_STORAGE_CONTAINER_FEATURES}' path='{output_blob_path}'")

            # Save Fitted Transformer (Pickle) - REQUIRED FOR ONNX
            import pickle
            transformer_buffer = io.BytesIO()
            pickle.dump(transformer, transformer_buffer)
            transformer_buffer.seek(0)

            if use_local_storage:
                backend_root = Path(__file__).resolve().parents[2]
                local_transformer_path = backend_root / "storage" / "features" / dataset_type / str(job_id) / "transformer.pkl"
                local_transformer_path.write_bytes(transformer_buffer.getvalue())
                transformer_storage_path = str(local_transformer_path.relative_to(backend_root)).replace("\\", "/")
                logger.info(f"Saved fitted transformer locally to path='{transformer_storage_path}'")
            else:
                transformer_blob_client = features_container_client.get_blob_client(transformer_blob_path)
                transformer_blob_client.upload_blob(transformer_buffer, overwrite=True)
                logger.info(f"Saved fitted transformer to container='{settings.AZURE_STORAGE_CONTAINER_FEATURES}' path='{transformer_blob_path}'")

            feature_service.update_feature_set_status_sync(job_id, "RUNNING", progress=0.9)

            # 6. Update Feature set metadata with storage path
            selected_features = list(df_transformed.columns)
            feature_service.update_feature_set_status_sync(
                job_id, "COMPLETED",
                storage_path=output_storage_path,
                selected_features=selected_features,
                all_features=selected_features,  # All generated features before analysis
                feature_count=len(df_transformed.columns),
                input_rows=len(df),
                processing_time_seconds=int(time.time() - start_time),
                selection_report={
                    "total_features": len(selected_features),
                    "feature_names": selected_features,
                    "transformer": "FraudFeatureEngineer",
                    "input_rows": len(df),
                    "output_rows": len(df_transformed),
                    "dataset_type": dataset_type,
                    "split_type": split_type,
                    "detected_roles": transformer.roles_.summary() if hasattr(transformer, 'roles_') else {},
                })

            logger.info(f"Feature computation completed for job {job_id} (dataset_type={dataset_type}, split_type={split_type})")
            
            # Commit the transaction
            db.commit()
            _set_redis_status(job_id, "COMPLETED")

        except Exception as e:
            logger.error(f"Feature computation failed for job {job_id}: {e}", exc_info=True)
            db.rollback()
            try:
                feature_service.update_feature_set_status_sync(job_id, "FAILED", error_message=str(e))
                db.commit()
            except Exception as inner_e:
                logger.error(f"Failed to update feature set status: {inner_e}")
                db.rollback()
            
            retries_left = self.max_retries - self.request.retries
            if retries_left <= 0:
                _set_redis_status(job_id, "FAILED")
            
            raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=3, name="app.workers.feature_worker.analyze_features")
def analyze_features(self, feature_set_id: str):
    """
    Run feature selection analysis on an existing feature set.
    
    Steps:
    1. Load features.parquet from storage
    2. Run FeatureSelector to calculate importance (Mutual Info, XGBoost)
    3. Update feature_set.selection_report with detailed metrics
    """
    from app.core.database_sync import SyncSessionLocal
    from app.services.feature_service import FeatureService
    from ml.features.feature_selector import FeatureSelector, FeatureSelectionConfig
    import pandas as pd
    import io
    from azure.storage.blob import BlobServiceClient
    from app.core.config import get_settings
    
    settings = get_settings()
    
    with SyncSessionLocal() as db:
        feature_service = FeatureService(db)
        logger.info(f"Starting feature analysis for set {feature_set_id}")
        
        try:
            # 1. Get metadata
            feature_set = feature_service.get_feature_set_sync(feature_set_id)
            if not feature_set or not feature_set.storage_path:
                raise ValueError(f"Feature set {feature_set_id} not found or missing storage path")
                
            # 2. Download features.parquet (engineered features — target NOT included)
            blob_path = feature_set.storage_path
            local_feature_path = _resolve_local_storage_path(blob_path)
            if local_feature_path is not None:
                logger.info(f"Loading feature set from local storage: {local_feature_path}")
                blob_data = local_feature_path.read_bytes()
            else:
                azure_conn = _get_azure_connection_string()
                blob_service = BlobServiceClient.from_connection_string(azure_conn)
                # Features are saved to FEATURES container by compute_features, not DATASETS
                container_client = blob_service.get_container_client(settings.AZURE_STORAGE_CONTAINER_FEATURES)
                blob_client = container_client.get_blob_client(blob_path)
                blob_data = blob_client.download_blob().readall()
            df = pd.read_parquet(io.BytesIO(blob_data))
            
            # 3. Resolve the target column.
            #    compute_features attaches the target column to features.parquet,
            #    so check there FIRST before falling back to original dataset.
            target_col = None
            existing_report = feature_set.selection_report or {}
            detected_roles = existing_report.get("detected_roles", {})
            original_target = detected_roles.get("target_col")
            
            if original_target:
                logger.info(f"Target column from compute step: '{original_target}'")
            else:
                logger.info("No target in detected_roles, will try auto-detection")
            
            # Step A: Check if target already exists in features.parquet
            if original_target and original_target in df.columns:
                target_col = original_target
                logger.info(f"Target column '{original_target}' found directly in features.parquet (NaN count: {df[target_col].isna().sum()})")
            else:
                # Step B: Auto-detect target from features.parquet columns
                if not original_target:
                    from ml.transformers.column_role_detector import ColumnRoleDetector
                    detector = ColumnRoleDetector()
                    roles = detector.detect(df)
                    original_target = roles.target_col
                    if original_target and original_target in df.columns:
                        target_col = original_target
                        logger.info(f"Auto-detected target '{original_target}' in features.parquet")
                
                # Step C: Fall back to loading from original dataset only if needed
                if not target_col:
                    logger.info("Target not in features.parquet, loading from original dataset")
                    from app.services.data_service import DataService
                    data_service = DataService(db)
                    dataset = data_service.get_dataset_sync(str(feature_set.dataset_id))
                    
                    if dataset and dataset.storage_path:
                        storage_path_parts = dataset.storage_path.split("/", 1)
                        if len(storage_path_parts) == 2:
                            _, orig_blob_path = storage_path_parts
                        else:
                            orig_blob_path = dataset.storage_path
                        
                        datasets_container_client = blob_service.get_container_client(settings.AZURE_STORAGE_CONTAINER_DATASETS)
                        orig_blob_client = datasets_container_client.get_blob_client(orig_blob_path)
                        orig_blob_data = orig_blob_client.download_blob().readall()
                        
                        if dataset.file_format == "parquet":
                            df_orig = pd.read_parquet(io.BytesIO(orig_blob_data))
                        elif dataset.file_format == "json":
                            df_orig = pd.read_json(io.BytesIO(orig_blob_data))
                        else:
                            try:
                                df_orig = pd.read_csv(io.BytesIO(orig_blob_data))
                            except UnicodeDecodeError:
                                try:
                                    df_orig = pd.read_csv(io.BytesIO(orig_blob_data), encoding='cp1252')
                                except UnicodeDecodeError:
                                    df_orig = pd.read_csv(io.BytesIO(orig_blob_data), encoding='ISO-8859-1')
                        
                        if not original_target:
                            from ml.transformers.column_role_detector import ColumnRoleDetector
                            detector = ColumnRoleDetector()
                            roles = detector.detect(df_orig, column_mapping=feature_set.config.get("column_mapping") if feature_set.config else None)
                            original_target = roles.target_col
                        
                        if original_target and original_target in df_orig.columns:
                            if len(df_orig) == len(df):
                                target_series = df_orig[original_target].reset_index(drop=True)
                                target_col = "__target__"
                                df[target_col] = target_series.values
                                logger.info(f"Loaded target '{original_target}' from original dataset ({target_series.nunique()} unique values)")
                            else:
                                logger.warning(f"Row count mismatch: features={len(df)}, original={len(df_orig)}. Cannot align target.")
                        else:
                            logger.warning(f"Target column '{original_target}' not found in original dataset")
                    else:
                        logger.warning("Could not load original dataset to extract target column")
            
            # Drop NaN from target if present
            if target_col and target_col in df.columns:
                nan_count = df[target_col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Target column '{target_col}' has {nan_count} NaN values, dropping those rows")
                    df = df.dropna(subset=[target_col]).reset_index(drop=True)
            
            if not target_col:
                logger.error("Cannot analyze features: no target column detected")
                selection_report = {
                    "error": "Target column not found. Cannot calculate feature importance.",
                    "detected_roles": detected_roles,
                }
                # Early return with error report
                feature_service.update_feature_set_status_sync(
                    feature_set_id,
                    status=feature_set.status,
                    selection_report=selection_report
                )
                db.commit()
                return selection_report
            else:
                # 4. Run Feature Selector
                logger.info(f"Running FeatureSelector with target column: {target_col} (original: '{original_target}')")
                
                # Configure for analysis (we want scores for ALL features)
                config = FeatureSelectionConfig(
                    max_features=len(df.columns) - 1,  # exclude target
                    variance_threshold=0.01,
                    correlation_threshold=0.99 
                )
                selector = FeatureSelector(config)
                
                # fit_transform calculates the scores
                _, report = selector.fit_transform(df, target_column=target_col)
                
                # Remove __target__ from any report entries (it's not a real feature)
                if "scores" in report:
                    report["scores"].pop("__target__", None)
                    report["scores"].pop(target_col, None)
                
                selection_report = report
                selection_report["original_target_column"] = original_target
            
            # 5. Save Report to DB
            # We update the selection_report field. 
            # Note: We do NOT change status to 'COMPLETED' if it's already completed.
            feature_service.update_feature_set_status_sync(
                feature_set_id, 
                status=feature_set.status,
                selection_report=selection_report
            )
            
            db.commit()
            logger.info(f"Feature analysis completed for {feature_set_id}")
            _set_redis_status(feature_set_id, "COMPLETED")
            return selection_report

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            db.rollback()
            # Log error in report
            try:
                feature_service.update_feature_set_status_sync(
                    feature_set_id,
                    status="FAILED",
                    selection_report={"error": str(e)}
                )
                db.commit()
            except Exception as inner_e:
                logger.error(f"Failed to update feature analyze status: {inner_e}")
                db.rollback()
                
            retries_left = self.max_retries - self.request.retries
            if retries_left <= 0:
                _set_redis_status(feature_set_id, "FAILED")
                
            raise self.retry(exc=e, countdown=60)