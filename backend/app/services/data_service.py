"""
Data Service
Business logic for dataset operations with Azure Blob Storage integration,
with a local filesystem fallback for local development.
"""
from typing import Optional, Tuple, List, Dict, Any
from uuid import UUID
from pathlib import Path
import io
import logging
import shutil

from fastapi import UploadFile
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from app.models.dataset import Dataset
from app.core.storage import storage_service
from app.core.naming import generate_raw_artifact_id, generate_merge_artifact_id

logger = logging.getLogger(__name__)

# Local storage fallback root (used when Azure is not configured, e.g. local dev)
LOCAL_STORAGE_ROOT = Path("storage/datasets")


def _is_local_storage_path(storage_path: Optional[str]) -> bool:
    """Determine whether a dataset's storage_path refers to a local filesystem
    location (as opposed to an Azure Blob Storage path)."""
    if not storage_path:
        return False
    if storage_path.startswith(str(LOCAL_STORAGE_ROOT)):
        return True
    # Absolute or relative local paths that actually exist on disk are treated
    # as local even if they don't start with the expected prefix.
    try:
        return Path(storage_path).exists()
    except (OSError, ValueError):
        return False


class DataService:
    """Service for dataset operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.storage = storage_service
    
    async def list_datasets(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        include_merged: bool = False,
        dataset_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Dataset], int]:
        """List datasets with pagination."""
        query = select(Dataset).order_by(Dataset.created_at.desc())
        count_query = select(func.count(Dataset.id))

        # Filter by dataset_type when provided
        if dataset_type:
            query = query.where(Dataset.dataset_type == dataset_type)
            count_query = count_query.where(Dataset.dataset_type == dataset_type)
            # Safety net: when caller asks for 'raw', also exclude any rows with a
            # parent_id set — those are derived datasets (split/merged) that may have
            # dataset_type='raw' due to a migration-time server_default backfill gap.
            if dataset_type == "raw":
                query = query.where(Dataset.parent_id == None)  # noqa: E711
                count_query = count_query.where(Dataset.parent_id == None)  # noqa: E711
        elif not include_merged:
            # Default: exclude non-raw types (merged, split) unless caller opts in
            query = query.where(Dataset.dataset_type == "raw")
            count_query = count_query.where(Dataset.dataset_type == "raw")
            query = query.where(Dataset.parent_id == None)  # noqa: E711
            count_query = count_query.where(Dataset.parent_id == None)  # noqa: E711
        else:
            # Exclude splits when include_merged is true so they don't consume the page limit
            query = query.where(~Dataset.name.like("%(Train Split)%"), ~Dataset.name.like("%(Test Split)%"))
            count_query = count_query.where(~Dataset.name.like("%(Train Split)%"), ~Dataset.name.like("%(Test Split)%"))

        # Filter by status
        if status:
            query = query.where(Dataset.status == status)
            count_query = count_query.where(Dataset.status == status)

        # Shared registry: no per-user ownership filter

        # Get total count
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()

        # Paginate
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        # Execute
        result = await self.db.execute(query)
        datasets = result.scalars().all()

        return list(datasets), total
    
    async def get_dataset(self, dataset_id: str, user_id: Optional[str] = None) -> Optional[Dataset]:
        """Get a single dataset by ID."""
        try:
            uuid_id = UUID(dataset_id)
        except ValueError:
            return None
        
        stmt = select(Dataset).where(Dataset.id == uuid_id)
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def create_dataset(
        self,
        name: str,
        file: UploadFile,
        description: Optional[str] = None,
        version: str = "1.0",
        user_id: Optional[str] = None,
    ) -> Dataset:
        """Create a new dataset from uploaded file."""
        # Auto-version if a dataset with the same name already exists (scoped to same user)
        base_name = name
        counter = 2
        while True:
            dup_query = select(Dataset).where(Dataset.name == name, Dataset.status != "ARCHIVED")
            if user_id:
                dup_query = dup_query.where(Dataset.created_by == str(user_id))
            result = await self.db.execute(dup_query)
            if not result.scalars().first():
                break
            name = f"{base_name} ({counter})"
            counter += 1

        # Read file content
        content = await file.read()
        
        # Determine format and parse
        file_format = "csv"
        if file.filename and file.filename.endswith(".parquet"):
            file_format = "parquet"
            df = pd.read_parquet(io.BytesIO(content))
        elif file.filename and file.filename.endswith(".json"):
            file_format = "json"
            df = pd.read_json(io.BytesIO(content))
        else:
            try:
                df = pd.read_csv(io.BytesIO(content))
            except UnicodeDecodeError:
                # Handle files saved by Excel or legacy tools with non-UTF-8 encodings
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding='cp1252')
                except UnicodeDecodeError:
                    df = pd.read_csv(io.BytesIO(content), encoding='ISO-8859-1')
        
        # Extract schema
        schema = {
            "columns": [
                {
                    "name": col,
                    "type": str(df[col].dtype),
                    "nullable": bool(df[col].isna().any()),
                }
                for col in df.columns
            ]
        }
        
        # Compute basic statistics
        statistics = {}
        for col in df.columns:
            col_stats = {"type": str(df[col].dtype)}
            if df[col].dtype in ["int64", "float64"]:
                col_stats.update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                })
            elif df[col].dtype == "object":
                col_stats["unique_count"] = int(df[col].nunique())
            statistics[col] = col_stats
        
        # Create database record first so it's always visible in the registry
        raw_artifact_id = generate_raw_artifact_id()
        placeholder_path = f"pending/{raw_artifact_id}/data.{file_format}"
        dataset = Dataset(
            name=name,
            description=description,
            version=version,
            storage_path=placeholder_path,
            file_format=file_format,
            file_size_bytes=len(content),
            row_count=len(df),
            column_count=len(df.columns),
            schema=schema,
            statistics=statistics,
            status="PROCESSING",
            dataset_type="raw",
            created_by=user_id,
        )
        self.db.add(dataset)
        await self.db.commit()
        await self.db.refresh(dataset)

        # Upload to Azure Blob Storage (enterprise naming: raw/{artifact_id}/data.{format})
        # If Azure is not configured (e.g. local development), fall back to saving the
        # file on local disk under storage/datasets/<raw_artifact_id>/data.<format> so
        # dataset creation still works without AZURE_STORAGE_CONNECTION_STRING.
        try:
            storage_path = await self.storage.upload_raw_dataset(
                dataset_name=name,
                version=version,
                data=content,
                file_format=file_format,
                metadata={
                    "row_count": str(len(df)),
                    "column_count": str(len(df.columns)),
                    "description": description or "",
                },
                artifact_id=raw_artifact_id,
            )
            logger.info(f"Uploaded raw dataset to storage: {storage_path}")
            dataset.storage_path = storage_path
            dataset.status = "ACTIVE"
            await self.db.commit()
            await self.db.refresh(dataset)
        except Exception as e:
            logger.warning(f"Azure upload unavailable, falling back to local storage: {e}")
            try:
                local_dir = LOCAL_STORAGE_ROOT / raw_artifact_id
                local_dir.mkdir(parents=True, exist_ok=True)
                local_file_path = local_dir / f"data.{file_format}"
                local_file_path.write_bytes(content)

                local_storage_path = str(local_file_path)
                logger.info(f"Saved raw dataset to local storage: {local_storage_path}")
                dataset.storage_path = local_storage_path
                dataset.status = "ACTIVE"
                await self.db.commit()
                await self.db.refresh(dataset)
            except Exception as local_e:
                logger.error(f"Failed to save dataset to local storage: {local_e}")
                # Dataset record remains in PROCESSING state — visible in UI with pending path

        return dataset
    
    async def preview_dataset(
        self,
        dataset_id: str,
        rows: int = 10,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get preview of dataset rows."""
        dataset = await self.get_dataset(dataset_id, user_id=user_id)
        if not dataset:
            return None
        
        # Load preview efficiently without downloading the entire file when possible
        try:
            if _is_local_storage_path(dataset.storage_path):
                # Read directly from the local filesystem
                local_path = Path(dataset.storage_path)
                if not local_path.exists():
                    raise FileNotFoundError(dataset.storage_path)

                if dataset.file_format == "csv":
                    try:
                        df = pd.read_csv(local_path, nrows=rows)
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(local_path, nrows=rows, encoding='cp1252')
                        except UnicodeDecodeError:
                            df = pd.read_csv(local_path, nrows=rows, encoding='ISO-8859-1')
                    preview_df = df
                elif dataset.file_format == "parquet":
                    df = pd.read_parquet(local_path)
                    preview_df = df.head(rows)
                else:
                    df = pd.read_json(local_path)
                    preview_df = df.head(rows)
            elif dataset.file_format == "csv":
                # For CSV, streaming via SAS URL using pandas `nrows` drastically reduces latency
                sas_url = self.storage.generate_sas_url(dataset.storage_path, permission="r")
                try:
                    df = pd.read_csv(sas_url, nrows=rows)
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(sas_url, nrows=rows, encoding='cp1252')
                    except UnicodeDecodeError:
                        df = pd.read_csv(sas_url, nrows=rows, encoding='ISO-8859-1')
                preview_df = df
            else:
                content = await self.storage.download_dataset(dataset.storage_path)
                
                # Parse based on format
                if dataset.file_format == "parquet":
                    df = pd.read_parquet(io.BytesIO(content))
                else:
                    df = pd.read_json(io.BytesIO(content))
                
                preview_df = df.head(rows)
            
            return {
                "columns": list(df.columns),
                "rows": preview_df.to_dict(orient="records"),
                "total_rows": dataset.row_count,
                "preview_rows": len(preview_df),
            }
        except FileNotFoundError:
            logger.warning(f"Dataset file not found in storage: {dataset.storage_path}")
            return {
                "columns": [c["name"] for c in dataset.schema.get("columns", [])] if dataset.schema else [],
                "rows": [],
                "total_rows": dataset.row_count,
                "preview_rows": 0,
                "error": "File not found in storage",
            }
        except Exception as e:
            logger.error(f"Failed to load dataset preview: {e}")
            return {
                "columns": [c["name"] for c in dataset.schema.get("columns", [])] if dataset.schema else [],
                "rows": [],
                "total_rows": dataset.row_count,
                "preview_rows": 0,
                "error": str(e),
            }
    
    async def delete_dataset(self, dataset_id: str, hard_delete: bool = False, user_id: Optional[str] = None) -> bool:
        """Delete a dataset (soft delete by default)."""
        dataset = await self.get_dataset(dataset_id, user_id=user_id)
        if not dataset:
            return False
        
        if hard_delete:
            # Delete from storage first when present. If the file/blob is missing or any
            # error occurs, we still remove the DB record so orphaned rows can be cleaned
            # from the UI.
            if dataset.storage_path and _is_local_storage_path(dataset.storage_path):
                try:
                    local_path = Path(dataset.storage_path)
                    if local_path.exists():
                        # Remove the artifact file and, if it becomes empty, its parent
                        # artifact directory (storage/datasets/<artifact_id>/).
                        local_path.unlink()
                        parent_dir = local_path.parent
                        if parent_dir.exists() and parent_dir != LOCAL_STORAGE_ROOT:
                            try:
                                if not any(parent_dir.iterdir()):
                                    parent_dir.rmdir()
                                else:
                                    shutil.rmtree(parent_dir, ignore_errors=True)
                            except OSError:
                                pass
                        logger.info(f"Deleted dataset from local storage: {dataset.storage_path}")
                    else:
                        logger.warning(
                            f"Local file not found (may have been deleted manually): {dataset.storage_path}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Local storage delete failed (continuing to remove DB record): {dataset.storage_path} - {e}"
                    )
            elif dataset.storage_path and "/" in dataset.storage_path:
                try:
                    success = await self.storage.delete_dataset(dataset.storage_path)
                    if success:
                        logger.info(f"Deleted dataset from storage: {dataset.storage_path}")
                    else:
                        logger.warning(
                            f"Blob not found in storage (may have been deleted manually): {dataset.storage_path}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Storage delete failed (continuing to remove DB record): {dataset.storage_path} - {e}"
                    )
            else:
                logger.warning(f"Dataset has no valid storage_path, skipping blob delete: {dataset.storage_path!r}")
            # Remove dependent rows in FK order. Note: DB has training_jobs.feature_set_id (no dataset_id).
            from sqlalchemy import delete, update, select
            from app.models.training_job import TrainingJob
            from app.models.feature_set import FeatureSet
            from app.models.dataset_lineage import DatasetLineage
            from app.models.ml_model import MLModel
            ds_id = dataset.id
            try:
                # First delete training jobs that directly reference this dataset
                await self.db.execute(delete(TrainingJob).where(TrainingJob.dataset_id == ds_id))
                # Then handle feature sets
                fs_result = await self.db.execute(select(FeatureSet.id).where(FeatureSet.dataset_id == ds_id))
                fs_ids = list(fs_result.scalars().all())
                if fs_ids:
                    await self.db.execute(delete(TrainingJob).where(TrainingJob.feature_set_id.in_(fs_ids)))
                    await self.db.execute(update(MLModel).values(feature_set_id=None).where(MLModel.feature_set_id.in_(fs_ids)))
                await self.db.execute(delete(FeatureSet).where(FeatureSet.dataset_id == ds_id))
            except Exception as e:
                logger.warning(f"Delete FeatureSet / TrainingJob / MLModel for dataset {ds_id}: {e}")
            try:
                await self.db.execute(
                    delete(DatasetLineage).where(
                        (DatasetLineage.source_dataset_id == ds_id) | (DatasetLineage.target_dataset_id == ds_id)
                    )
                )
            except Exception as e:
                logger.warning(f"Delete DatasetLineage for dataset {ds_id}: {e}")
            # Delete by id to avoid session/relationship issues with the loaded object
            await self.db.execute(delete(Dataset).where(Dataset.id == ds_id))
        else:
            # Soft delete
            dataset.status = "ARCHIVED"
        
        await self.db.commit()
        return True
    
    async def get_dataset_download_url(
        self,
        dataset_id: str,
        expiry_hours: int = 1,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get a temporary download URL for a dataset.

        For Azure-backed datasets this returns a SAS URL. For locally-stored
        datasets (local development without Azure configured) this returns a
        `file://` URI pointing at the dataset on disk, since there is no blob
        storage to generate a signed URL against.
        """
        dataset = await self.get_dataset(dataset_id, user_id=user_id)
        if not dataset:
            return None
        
        try:
            if _is_local_storage_path(dataset.storage_path):
                local_path = Path(dataset.storage_path)
                if not local_path.exists():
                    logger.warning(f"Local dataset file not found: {dataset.storage_path}")
                    return None
                return local_path.resolve().as_uri()

            return self.storage.generate_sas_url(
                storage_path=dataset.storage_path,
                expiry_hours=expiry_hours,
                permission="r",
            )
        except Exception as e:
            logger.error(f"Failed to generate download URL: {e}")
            return None

    async def merge_datasets(
        self,
        dataset_ids: List[str],
        new_name: str,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dataset:
        """Merge multiple datasets into a new one."""
        # 1. Fetch all datasets
        datasets = []
        source_names = []
        for ds_id in dataset_ids:
            ds = await self.get_dataset(ds_id, user_id=user_id)
            if not ds:
                raise ValueError(f"Dataset {ds_id} not found")
            datasets.append(ds)
            source_names.append(ds.name)
            
        # 2. Validate Schemas
        dataset_schemas = []
        for ds in datasets:
             dataset_schemas.append({
                 "name": ds.name,
                 "columns": sorted([col["name"] for col in ds.schema["columns"]])
             })
        
        # Check lengths
        first_len = len(dataset_schemas[0]["columns"])
        length_mismatch = False
        for ds_schema in dataset_schemas[1:]:
            if len(ds_schema["columns"]) != first_len:
                length_mismatch = True
                break
        
        if length_mismatch:
            # Construct detailed message
            msg_parts = ["Column count mismatch found:"]
            for ds_schema in dataset_schemas:
                msg_parts.append(f"- {ds_schema['name']}: {len(ds_schema['columns'])} columns")
            raise ValueError("\n".join(msg_parts))
            
        # Check column names equality (lengths are now known to be equal)
        base_columns = dataset_schemas[0]["columns"]
        base_name = dataset_schemas[0]["name"]
        
        for ds_schema in dataset_schemas[1:]:
            if ds_schema["columns"] != base_columns:
                # Find diffs
                diff_added = set(ds_schema["columns"]) - set(base_columns)
                diff_missing = set(base_columns) - set(ds_schema["columns"])
                
                raise ValueError(
                    f"Column name mismatch between '{base_name}' and '{ds_schema['name']}':\n"
                    f"Extra in {ds_schema['name']}: {list(diff_added) if diff_added else 'None'}\n"
                    f"Missing in {ds_schema['name']}: {list(diff_missing) if diff_missing else 'None'}"
                )

        # 3. Download and Load DataFrames
        dfs = []
        for ds in datasets:
            try:
                if _is_local_storage_path(ds.storage_path):
                    local_path = Path(ds.storage_path)
                    if not local_path.exists():
                        raise FileNotFoundError(ds.storage_path)
                    if ds.file_format == "parquet":
                        df = pd.read_parquet(local_path)
                    elif ds.file_format == "json":
                        df = pd.read_json(local_path)
                    else:
                        try:
                            df = pd.read_csv(local_path)
                        except UnicodeDecodeError:
                            try:
                                df = pd.read_csv(local_path, encoding='cp1252')
                            except UnicodeDecodeError:
                                df = pd.read_csv(local_path, encoding='ISO-8859-1')
                else:
                    content = await self.storage.download_dataset(ds.storage_path)
                    if ds.file_format == "parquet":
                        df = pd.read_parquet(io.BytesIO(content))
                    elif ds.file_format == "json":
                        df = pd.read_json(io.BytesIO(content))
                    else:
                        df = pd.read_csv(io.BytesIO(content))
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load dataset {ds.id}: {e}")
                raise ValueError(f"Failed to load dataset {ds.name}: {str(e)}")
                
        # 3. Merge DataFrames
        try:
            if not dfs:
                raise ValueError("No data found in datasets")
                
            merged_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        except Exception as e:
            logger.error(f"Failed to concat datasets: {e}")
            raise ValueError(f"Failed to merge datasets: {str(e)}. Ensure columns match.")
            
        # 4. Save New Dataset
        # Use the same format as the source datasets
        output = io.BytesIO()
        base_format = datasets[0].file_format  # Preserve original format
        
        if base_format == "parquet":
            merged_df.to_parquet(output, index=False)
        elif base_format == "json":
            merged_df.to_json(output, orient="records", indent=2)
        else:  # csv
            merged_df.to_csv(output, index=False)
        
        data = output.getvalue()
        
        # Auto-generate description if not provided
        if not description:
            description = f"Merged from: {', '.join(source_names)}"
        
        version = "1.0"
        
        # Upload using hierarchical storage with enterprise artifact ID and lineage.
        # Falls back to local disk storage when Azure is not configured, mirroring
        # the fallback used in create_dataset().
        merged_id = generate_merge_artifact_id()
        try:
            storage_path = await self.storage.upload_merged_dataset(
                merged_dataset_id=merged_id,
                version=version,
                data=data,
                source_dataset_ids=[str(ds.id) for ds in datasets],
                merge_config={
                    "strategy": "concat",
                    "source_datasets": source_names,
                    "row_count": len(merged_df),
                    "column_count": len(merged_df.columns),
                },
                file_format=base_format,
                metadata={
                    "row_count": str(len(merged_df)),
                    "column_count": str(len(merged_df.columns)),
                    "description": description,
                }
            )
        except Exception as e:
            logger.warning(f"Azure upload unavailable for merged dataset, falling back to local storage: {e}")
            try:
                local_dir = LOCAL_STORAGE_ROOT / merged_id
                local_dir.mkdir(parents=True, exist_ok=True)
                local_file_path = local_dir / f"data.{base_format}"
                local_file_path.write_bytes(data)
                storage_path = str(local_file_path)
                logger.info(f"Saved merged dataset to local storage: {storage_path}")
            except Exception as local_e:
                raise ValueError(f"Failed to save merged dataset locally: {str(local_e)}")
            
        # 5. Create DB Record with dataset_type and lineage
        schema = {
            "columns": [
                {
                    "name": col,
                    "type": str(merged_df[col].dtype),
                    "nullable": bool(merged_df[col].isna().any()),
                }
                for col in merged_df.columns
            ]
        }
        
        # Compute basic stats
        statistics = {}
        for col in merged_df.columns:
            col_stats = {"type": str(merged_df[col].dtype)}
            statistics[col] = col_stats

        new_dataset = Dataset(
            name=new_name,
            description=description,
            version=version,
            storage_path=storage_path,
            file_format=base_format,
            created_by=user_id,
            file_size_bytes=len(data),
            row_count=len(merged_df),
            column_count=len(merged_df.columns),
            schema=schema,
            statistics=statistics,
            status="ACTIVE",
            parent_id=datasets[0].id,
            dataset_type="merged",
        )
        
        self.db.add(new_dataset)
        await self.db.commit()
        await self.db.refresh(new_dataset)
        
        # Create lineage records for each source dataset
        from app.models.dataset_lineage import DatasetLineage

        for source_ds in datasets:
            lineage = DatasetLineage(
                target_dataset_id=new_dataset.id,
                source_dataset_id=source_ds.id,
                relationship_type="merged_from",
                lineage_metadata={
                    "source_name": source_ds.name,
                    "source_rows": source_ds.row_count,
                    "merge_strategy": "concat",
                }
            )
            self.db.add(lineage)

        await self.db.commit()

        logger.info(f"Created merged dataset '{new_name}' from {len(datasets)} sources")
        return new_dataset
    
    # ===== Synchronous methods for Celery workers =====
    
    def get_dataset_sync(self, dataset_id: str) -> Optional[Dataset]:
        """Get a single dataset by ID (sync version for Celery)."""
        try:
            uuid_id = UUID(dataset_id)
        except ValueError:
            return None
        
        result = self.db.execute(
            select(Dataset).where(Dataset.id == uuid_id)
        )
        return result.scalar_one_or_none()