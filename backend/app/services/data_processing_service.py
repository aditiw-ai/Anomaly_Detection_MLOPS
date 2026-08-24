"""
Data Processing Service
Handles data preparation, splitting, and feature engineering for training jobs.
"""
import io
import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.data_service import DataService
from app.core.storage import storage_service
from app.core.naming import generate_split_job_artifact_id

logger = logging.getLogger(__name__)


class DataProcessingService:
    """Service for data processing and splitting."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.data_service = DataService(db)
        self.storage = storage_service

    def _save_split_locally(self, split_job_artifact_id: str, split_type: str, data: bytes) -> str:
        """
        Save split parquet bytes to the local backend storage directory when
        Azure Blob Storage is not configured. Mirrors the existing local
        dataset storage convention used elsewhere in the project
        (storage/datasets/...), so training_worker.py's local-path detection
        can read it back directly.
        """
        local_dir = Path("storage") / "datasets" / split_job_artifact_id / split_type
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / "data.parquet"
        local_file.write_bytes(data)
        return str(local_file)

    async def prepare_training_data(
        self,
        dataset_id: str,
        feature_config: Dict[str, bool],
        test_size: float = 0.2,
        seed: int = 42
    ) -> Dict[str, str]:
        """
        Loads dataset, selects features, splits into train/test (preserving raw values),
        saves to 'datasets/processed/' container, returns artifact paths.
        
        Note: No scaling is applied here. Raw values are preserved for feature 
        engineering transformations that expect original value ranges.
        
        Returns:
            Dict containing 'train_path', 'test_path', 'validation_path' (optional)
        """
        logger.info(f"Preparing data for dataset {dataset_id} with test_size={test_size}")
        
        # 1. Load Dataset
        dataset = await self.data_service.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Download content
        try:
            from pathlib import Path

            local_path = Path(dataset.storage_path)

            if local_path.exists() and local_path.is_file():
                logger.info(
                    f"Loading dataset from local storage: {local_path}"
                )
                content = local_path.read_bytes()
            else:
                logger.info(
                    f"Loading dataset from Azure storage: {dataset.storage_path}"
                )
                content = await self.storage.download_dataset(
                    dataset.storage_path
                )

            logger.info(
                f"Downloaded dataset {dataset.name} size={len(content)} bytes"
            )

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise ValueError(f"Could not download dataset file: {e}")

        # Load into Pandas
        try:
            if dataset.file_format == "parquet":
                df = pd.read_parquet(io.BytesIO(content))
            elif dataset.file_format == 'csv':
                try:
                    df = pd.read_csv(io.BytesIO(content))
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(io.BytesIO(content), encoding='cp1252')
                    except UnicodeDecodeError:
                        df = pd.read_csv(io.BytesIO(content), encoding='ISO-8859-1')
            elif dataset.file_format == "json":
                df = pd.read_json(io.BytesIO(content))
            else:
                df = pd.read_csv(io.BytesIO(content))
            
            logger.info(f"Loaded DataFrame: {df.shape}")
            logger.info(f"[DIAGNOSTIC] POST-LOAD: rows={len(df)}, cols={len(df.columns)}, columns={list(df.columns)}")
            
            # Validation: Check for empty dataset
            if df.empty or df.shape[0] == 0:
                raise ValueError("Input dataset is empty (0 rows)")
                
        except Exception as e:
            if "Input dataset is empty" in str(e):
                raise e
            raise ValueError(f"Failed to parse dataset file: {e}")

        # 2. Train-Test Split (MOVED BEFORE feature engineering to avoid leakage)
        # Validate test_size
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")

        logger.info(f"[DIAGNOSTIC] PRE-SPLIT: df.shape={df.shape}, test_size={test_size}")

        # Stratified split: preserve fraud/non-fraud ratio in both sets
        _target_candidates = ['is_fraud', 'fraud_label', 'target', 'label', 'class', 'is_fraudulent']
        _strat_col = next((c for c in _target_candidates if c in df.columns), None)
        if _strat_col and df[_strat_col].nunique() >= 2:
            logger.info(f"Stratified split on '{_strat_col}' — class distribution preserved in train/test")
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=seed, stratify=df[_strat_col]
            )
        else:
            logger.warning("No suitable target column for stratification — using random split")
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

        logger.info(f"[DIAGNOSTIC] POST-SPLIT: train={len(train_df)} rows, test={len(test_df)} rows")
        
        # Validation: Prevent saving empty splits
        if train_df.empty or test_df.empty:
            raise ValueError(f"Split resulted in empty dataframe: Train={len(train_df)}, Test={len(test_df)}")
            
        logger.info(f"Split complete: Train={train_df.shape}, Test={test_df.shape}")

        # 3a. Feature Config Handling
        # IMPORTANT: feature_config contains feature GROUP toggles
        # (e.g., transaction_features: True, behavioral_features: True)
        # NOT column names. These toggles are passed to FraudFeatureEngineer
        # during compute_features — do NOT filter DataFrame columns here.
        if feature_config:
            logger.info(f"Feature config (group toggles for transformer): {feature_config}")
            # No column filtering — all columns must be preserved for feature engineering

        
        
        # 3b. No pre-scaling applied here
        # DESIGN DECISION: Feature engineering transformations (log, sqrt, ratios) 
        # require raw values, not z-scores. Scaling (if needed) will be applied 
        # later in the training pipeline after feature engineering.
        # This prevents:
        # - log1p() on z-scores producing meaningless values
        # - Ratio features (amount_vs_user_avg) becoming invalid comparisons
        # - Double-scaling when training pipeline applies StandardScaler
        logger.info(f"Preserving raw values for {len(train_df.columns)} columns (no pre-scaling)")

        # 4. Save Artifacts: processed/{split_job_artifact_id}/train|test/data.parquet (enterprise naming)
        split_job_artifact_id = generate_split_job_artifact_id()
        train_path = f"processed/{split_job_artifact_id}/train/data.parquet"
        test_path = f"processed/{split_job_artifact_id}/test/data.parquet"
        
        # Determine format for saving
        logger.info(f"[DIAGNOSTIC] PRE-WRITE: train={len(train_df)} rows, test={len(test_df)} rows")
        
        # Resolve PyArrow mixed-type serialization errors by casting object columns to string
        for col in train_df.select_dtypes(include=['object']).columns:
            train_df[col] = train_df[col].astype(str)
        for col in test_df.select_dtypes(include=['object']).columns:
            test_df[col] = test_df[col].astype(str)
        train_buffer = io.BytesIO()
        train_df.to_parquet(train_buffer, index=False, engine='pyarrow')
        train_bytes = train_buffer.getvalue()
        logger.info(f"[DIAGNOSTIC] POST-WRITE-TRAIN: buffer_size={len(train_bytes)} bytes")
        
        test_buffer = io.BytesIO()
        test_df.to_parquet(test_buffer, index=False, engine='pyarrow')
        test_bytes = test_buffer.getvalue()
        logger.info(f"[DIAGNOSTIC] POST-WRITE-TEST: buffer_size={len(test_bytes)} bytes")
        
        # 5. Register Output Datasets (path: processed/split_YYYYMMDDTHHMMSSZ_xxx/train|test)
        from app.models.dataset import Dataset as DatasetModel
        import uuid

        version = getattr(dataset, "version", None) or "1.0"

        # Whether Azure Blob Storage is configured for this environment. When it
        # isn't, split artifacts are written to local backend storage instead so
        # local development (e.g. on Windows without Azure creds) keeps working.
        azure_configured = bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING"))

        # Upload Train Split
        if azure_configured:
            try:
                train_storage_path = await self.storage.upload_dataset(
                    name=f"{split_job_artifact_id}/train",
                    version=version,
                    data=train_bytes,
                    file_format="parquet",
                    dataset_type="split",
                    metadata={
                        "row_count": str(len(train_df)),
                        "source_dataset_id": str(dataset.id),
                        "split_type": "train",
                        "split_job_artifact_id": split_job_artifact_id,
                    }
                )
            except Exception as e:
                logger.exception("Upload train split failed: %s", e)
                raise ValueError(f"Failed to upload train split to cloud: {e}") from e
        else:
            try:
                train_storage_path = self._save_split_locally(
                    split_job_artifact_id, "train", train_bytes
                )
                logger.info(f"Azure not configured — saved train split locally: {train_storage_path}")
            except Exception as e:
                logger.exception("Local save of train split failed: %s", e)
                raise ValueError(f"Failed to save train split locally: {e}") from e

        # Train Dataset DB Record
        train_dataset_id = uuid.uuid4()
        train_dataset = DatasetModel(
            id=train_dataset_id,
            name=f"{dataset.name} (Train Split)",
            description=f"Training split ({(1-test_size):.0%}) from {dataset.name}",
            version=version,
            storage_path=train_storage_path,
            file_format="parquet",
            file_size_bytes=len(train_bytes),
            row_count=len(train_df),
            column_count=len(train_df.columns),
            schema={"columns": [{"name": c, "type": str(train_df[c].dtype)} for c in train_df.columns]},
            parent_id=dataset.id,
            status="ACTIVE",
            created_by=dataset.created_by,
            dataset_type="split",
            split_type="train",
        )
        self.db.add(train_dataset)
        
        # Upload Test Split
        if azure_configured:
            try:
                test_storage_path = await self.storage.upload_dataset(
                    name=f"{split_job_artifact_id}/test",
                    version=version,
                    data=test_bytes,
                    file_format="parquet",
                    dataset_type="split",
                    metadata={
                        "row_count": str(len(test_df)),
                        "source_dataset_id": str(dataset.id),
                        "split_type": "test",
                        "split_job_artifact_id": split_job_artifact_id,
                    }
                )
            except Exception as e:
                logger.exception("Upload test split failed: %s", e)
                raise ValueError(f"Failed to upload test split to cloud: {e}") from e
        else:
            try:
                test_storage_path = self._save_split_locally(
                    split_job_artifact_id, "test", test_bytes
                )
                logger.info(f"Azure not configured — saved test split locally: {test_storage_path}")
            except Exception as e:
                logger.exception("Local save of test split failed: %s", e)
                raise ValueError(f"Failed to save test split locally: {e}") from e

        # Test Dataset DB Record
        test_dataset = DatasetModel(
            id=uuid.uuid4(),
            name=f"{dataset.name} (Test Split)",
            description=f"Testing split ({test_size:.0%}) from {dataset.name}",
            version=version,
            storage_path=test_storage_path,
            file_format="parquet",
            file_size_bytes=len(test_bytes),
            row_count=len(test_df),
            column_count=len(test_df.columns),
            schema={"columns": [{"name": c, "type": str(test_df[c].dtype)} for c in test_df.columns]},
            parent_id=dataset.id,
            status="ACTIVE",
            created_by=dataset.created_by,
            dataset_type="split",
            split_type="test",
        )
        self.db.add(test_dataset)

        try:
            await self.db.commit()
        except Exception as e:
            logger.exception("Failed to save split dataset records: %s", e)
            raise ValueError(f"Failed to save split records to database: {e}") from e

        logger.info(f"[DIAGNOSTIC] COMPLETE: Returning train_rows={len(train_df)}, test_rows={len(test_df)}")
        return {
            "train_path": train_path,
            "test_path": test_path,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_dataset_id": str(train_dataset_id),
            "test_dataset_id": str(test_dataset.id)
        }