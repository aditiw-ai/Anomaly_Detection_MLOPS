"""
Azure Blob Storage Service
Handles all blob storage operations for datasets and model artifacts.
"""
from app.core.time import IST, now_ist
from typing import Optional, BinaryIO, List, Dict, Any
from datetime import datetime, timedelta
import io
import logging

from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError


from app.core.config import settings

logger = logging.getLogger(__name__)


# ============= Dataset Type Constants =============

class DatasetType:
    """Dataset type constants for hierarchical storage."""
    RAW = "raw"
    MERGED = "merged"
    SPLIT = "split"


class SplitType:
    """Split type constants."""
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class StorageService:
    """Service for Azure Blob Storage operations."""
    
    def __init__(self):
        """Initialize storage service with Azure credentials."""
        self._blob_service_client: Optional[BlobServiceClient] = None
        self._connection_string = settings.AZURE_STORAGE_CONNECTION_STRING
        
    @property
    def client(self) -> BlobServiceClient:
        """Lazy-load the blob service client."""
        if self._blob_service_client is None:
            if not self._connection_string:
                raise ValueError(
                    "AZURE_STORAGE_CONNECTION_STRING not configured. "
                    "Please set it in your .env file."
                )
            self._blob_service_client = BlobServiceClient.from_connection_string(
                self._connection_string,
                api_version="2024-11-04"
            )
        return self._blob_service_client
    
    def _get_container_client(self, container_name: str) -> ContainerClient:
        """Get a container client, creating the container if it doesn't exist."""
        container_client = self.client.get_container_client(container_name)
        try:
            container_client.get_container_properties()
        except ResourceNotFoundError:
            logger.info(f"Creating container: {container_name}")
            container_client.create_container()
        return container_client
    
    # ============= Dataset Operations =============
    
    async def upload_dataset(
        self,
        name: str,
        version: str,
        data: bytes,
        file_format: str = "parquet",
        dataset_type: str = DatasetType.RAW,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload a dataset to blob storage with hierarchical organization.
        
        Args:
            name: Dataset name
            version: Dataset version
            data: File content as bytes
            file_format: File format (parquet, csv, json)
            dataset_type: Type of dataset ('raw', 'merged', or 'split')
            metadata: Optional metadata dict
            
        Returns:
            Blob path (storage_path)
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_DATASETS
        
        # Build hierarchical path based on dataset type
        # Build hierarchical path based on dataset type
        if dataset_type == DatasetType.RAW:
            # Support enterprise artifact_id: raw/{artifact_id}/data.{format}
            # metadata may carry artifact_id when caller uses new naming
            artifact_id = (metadata or {}).get("artifact_id")
            if artifact_id:
                blob_path = f"raw/{artifact_id}/data.{file_format}"
            else:
                blob_path = f"raw/{name}/v{version}/data.{file_format}"
        elif dataset_type == DatasetType.MERGED:
            blob_path = f"merged/{name}/v{version}/data.{file_format}"
        elif dataset_type == DatasetType.SPLIT:
            # User requirement: "processed data seperate folder"
            # For splits, we use 'processed' folder
            # Structure: processed/{split_job_id}/{split_type}/data.{file_format} or processed/{name}/...
            # The 'name' argument here is typically {split_job_id} or {dataset_name}
            blob_path = f"processed/{name}/data.{file_format}"
        elif dataset_type == "features":
            # User requirement: "feature engineered data for both trained and test seperately"
            # Structure: features/{split_type}/{job_id}/features.{file_format}
            # expecting name to contain split info if available, or just job_id
            blob_path = f"features/{name}/features.{file_format}"
        else:
            # Fallback to old structure for backward compatibility
            blob_path = f"{name}/v{version}/data.{file_format}"
            logger.warning(f"Unknown dataset_type '{dataset_type}', using legacy path structure")
        
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        # Upload with metadata
        blob_metadata = metadata or {}
        blob_metadata.update({
            "dataset_name": name,
            "version": version,
            "format": file_format,
            "dataset_type": dataset_type,
            "uploaded_at": now_ist().isoformat(),
        })
        
        blob_client.upload_blob(
            data,
            overwrite=True,
            metadata=blob_metadata,
        )
        
        logger.info(f"Uploaded {dataset_type} dataset: {container_name}/{blob_path}")
        return f"{container_name}/{blob_path}"
    
    async def upload_raw_dataset(
        self,
        dataset_name: str,
        version: str,
        data: bytes,
        file_format: str = "parquet",
        metadata: Optional[Dict[str, str]] = None,
        artifact_id: Optional[str] = None,
    ) -> str:
        """
        Upload a raw dataset to datasets/raw/.
        
        Args:
            dataset_name: Name of the dataset (used for metadata/display; path if no artifact_id)
            version: Dataset version
            data: File content as bytes
            file_format: File format
            metadata: Optional metadata
            artifact_id: Optional enterprise artifact ID (e.g. raw_20260210T143022Z_a1b2c3d4).
                         When set, path is raw/{artifact_id}/data.{format}; otherwise legacy path.
            
        Returns:
            Storage path
        """
        meta = dict(metadata or {})
        if artifact_id:
            meta["artifact_id"] = artifact_id
        return await self.upload_dataset(
            name=dataset_name,
            version=version,
            data=data,
            file_format=file_format,
            dataset_type=DatasetType.RAW,
            metadata=meta,
        )
    
    async def upload_merged_dataset(
        self,
        merged_dataset_id: str,
        version: str,
        data: bytes,
        source_dataset_ids: List[str],
        merge_config: Dict[str, Any],
        file_format: str = "parquet",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload a merged dataset with lineage tracking.
        
        Args:
            merged_dataset_id: Unique ID for merged dataset
            version: Dataset version
            data: Merged dataset bytes
            source_dataset_ids: List of source dataset IDs
            merge_config: Merge configuration (strategy, etc.)
            file_format: File format
            metadata: Optional metadata
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_DATASETS
        base_path = f"merged/{merged_dataset_id}/v{version}"
        
        # Upload data
        data_path = f"{base_path}/data.{file_format}"
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(data_path)
        
        merge_metadata = metadata or {}
        merge_metadata.update({
            "dataset_type": DatasetType.MERGED,
            "merged_dataset_id": merged_dataset_id,
            "version": version,
        })
        blob_client.upload_blob(data, overwrite=True, metadata=merge_metadata)
        
        # Upload lineage.json
        lineage = {
            "merged_dataset_id": merged_dataset_id,
            "version": version,
            "created_at": now_ist().isoformat(),
            "source_dataset_ids": source_dataset_ids,
            "merge_config": merge_config,
        }
        lineage_path = f"{base_path}/lineage.json"
        await self._upload_json_metadata(container_name, lineage_path, lineage)
        
        # Upload merge_config.json
        config_path = f"{base_path}/merge_config.json"
        await self._upload_json_metadata(container_name, config_path, merge_config)
        
        logger.info(f"Uploaded merged dataset with lineage: {container_name}/{base_path}")
        return f"{container_name}/{data_path}"
    
    async def upload_split_dataset(
        self,
        split_job_id: str,
        split_type: str,  # 'train', 'test', 'validation'
        data: bytes,
        source_dataset_id: str,
        split_config: Dict[str, Any],
        file_format: str = "parquet",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload a train/test/validation split with proper organization.
        
        Args:
            split_job_id: Unique ID for this split job (groups train/test/val)
            split_type: 'train', 'test', or 'validation'
            data: Split dataset bytes
            source_dataset_id: ID of source dataset
            split_config: Split configuration
            file_format: File format
            metadata: Optional metadata
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_DATASETS
        # User requirement: "processed data seperate folder"
        base_path = f"processed/{split_job_id}"
        
        # Upload split data
        data_path = f"{base_path}/{split_type}/data.{file_format}"
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(data_path)
        
        split_metadata = metadata or {}
        split_metadata.update({
            "dataset_type": DatasetType.SPLIT,
            "split_job_id": split_job_id,
            "split_type": split_type,
            "source_dataset_id": source_dataset_id,
        })
        blob_client.upload_blob(data, overwrite=True, metadata=split_metadata)
        
        # Upload split metadata
        split_meta_path = f"{base_path}/{split_type}/metadata.json"
        split_meta = {
            "split_type": split_type,
            "rows": split_config.get(f"{split_type}_rows", 0),
            "ratio": split_config.get(f"{split_type}_ratio", 0),
            "created_at": now_ist().isoformat(),
        }
        await self._upload_json_metadata(container_name, split_meta_path, split_meta)
        
        # Upload split_config.json (only once, not per split)
        config_path = f"{base_path}/split_config.json"
        try:
            # Check if config already exists
            await self._download_json_metadata(f"{container_name}/{config_path}")
        except:
            # Upload if doesn't exist
            full_config = {
                "split_job_id": split_job_id,
                "source_dataset_id": source_dataset_id,
                "created_at": now_ist().isoformat(),
                **split_config,
            }
            await self._upload_json_metadata(container_name, config_path, full_config)
        
        # Upload source_dataset.json (reference to source)
        source_path = f"{base_path}/source_dataset.json"
        try:
            await self._download_json_metadata(f"{container_name}/{source_path}")
        except:
            source_info = {
                "source_dataset_id": source_dataset_id,
                "split_job_id": split_job_id,
            }
            await self._upload_json_metadata(container_name, source_path, source_info)
        
        logger.info(f"Uploaded {split_type} split: {container_name}/{data_path}")
        return f"{container_name}/{data_path}"
    
    async def download_dataset(
        self,
        storage_path: str,
    ) -> bytes:
        """
        Download a dataset from blob storage.
        
        Args:
            storage_path: Full storage path (container/blob_path)
            
        Returns:
            File content as bytes
        """
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid storage path: {storage_path}")

        container_name, blob_path = parts
        logger.debug(f"download_dataset: container={container_name!r} blob_path={blob_path!r}")

        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)

        try:
            import asyncio
            download_stream = await asyncio.to_thread(blob_client.download_blob)
            return await asyncio.to_thread(download_stream.readall)
        except ResourceNotFoundError:
            logger.warning(f"Blob not found: {container_name}/{blob_path}")
            raise FileNotFoundError(f"Dataset not found: {storage_path}")
    
    async def delete_dataset(self, storage_path: str) -> bool:
        """
        Delete a dataset from blob storage.
        
        Args:
            storage_path: Full storage path
            
        Returns:
            True if deleted, False if not found
        """
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            return False
        
        container_name, blob_path = parts
        
        # Check if this is a merged dataset (path: merged/{id}/v{version}/data.{ext})
        # If so, delete the entire directory to remove data + lineage + merge_config
        if blob_path.startswith("merged/"):
            # Extract directory: merged/{id}/v{version}/
            path_parts = blob_path.split("/")
            if len(path_parts) >= 3:  # merged/{id}/v{version}/...
                directory_prefix = "/".join(path_parts[:3]) + "/"  # merged/{id}/v{version}/
                
                container_client = self._get_container_client(container_name)
                deleted_count = 0
                
                try:
                    # List all blobs in the directory
                    blobs = container_client.list_blobs(name_starts_with=directory_prefix)
                    
                    # Delete each blob
                    for blob in blobs:
                        blob_client = container_client.get_blob_client(blob.name)
                        blob_client.delete_blob(delete_snapshots="include")
                        deleted_count += 1
                        logger.info(f"Deleted blob: {container_name}/{blob.name}")
                    
                    if deleted_count > 0:
                        logger.info(f"Deleted {deleted_count} files from merged dataset directory: {container_name}/{directory_prefix}")
                    else:
                        logger.warning(f"No files found in directory (already removed): {container_name}/{directory_prefix}")
                    # Return True either way so caller can clean DB (idempotent delete)
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete merged dataset directory: {e}")
                    return False
        # Standard single-file deletion for raw/processed datasets
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        try:
            blob_client.delete_blob(delete_snapshots="include")
            logger.info(f"Deleted dataset: {storage_path}")
            return True
        except ResourceNotFoundError:
            # Already gone - treat as success so caller can still remove DB record
            logger.warning(f"Blob already missing: {storage_path}")
            return True
    
    async def list_datasets(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all datasets in storage.
        
        Args:
            prefix: Optional prefix to filter by dataset name
            
        Returns:
            List of blob info dicts
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_DATASETS
        container_client = self._get_container_client(container_name)
        
        blobs = []
        for blob in container_client.list_blobs(name_starts_with=prefix):
            blobs.append({
                "name": blob.name,
                "size": blob.size,
                "created_on": blob.creation_time.isoformat() if blob.creation_time else None,
                "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                "metadata": blob.metadata,
            })
        
        return blobs
    
    # ============= Model Operations =============
    
    async def upload_model(
        self,
        model_name: str,
        version: str,
        model_data: bytes,
        format: str = "onnx",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload a model artifact to blob storage.
        
        Args:
            model_name: Model name
            version: Model version
            model_data: Serialized model bytes
            format: Model format (onnx, pkl, joblib)
            metadata: Optional metadata
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_MODELS
        blob_path = f"{model_name}/v{version}/model.{format}"
        
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        blob_metadata = metadata or {}
        blob_metadata.update({
            "model_name": model_name,
            "version": version,
            "format": format,
            "uploaded_at": now_ist().isoformat(),
        })
        
        blob_client.upload_blob(
            model_data,
            overwrite=True,
            metadata=blob_metadata,
        )
        
        logger.info(f"Uploaded model: {container_name}/{blob_path}")
        return f"{container_name}/{blob_path}"
    
    async def download_model(self, storage_path: str) -> bytes:
        """
        Download a model from blob storage.
        
        Args:
            storage_path: Full storage path
            
        Returns:
            Model bytes
        """
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid storage path: {storage_path}")
        
        container_name, blob_path = parts
        
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        try:
            import asyncio
            download_stream = await asyncio.to_thread(blob_client.download_blob)
            return await asyncio.to_thread(download_stream.readall)
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Model not found: {storage_path}")
    
    async def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version info
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_MODELS
        container_client = self._get_container_client(container_name)
        
        versions = []
        prefix = f"{model_name}/"
        
        for blob in container_client.list_blobs(name_starts_with=prefix):
            # Extract version from path: model_name/vX.Y/model.onnx
            parts = blob.name.split("/")
            if len(parts) >= 2:
                version = parts[1].replace("v", "")
                versions.append({
                    "version": version,
                    "path": blob.name,
                    "size": blob.size,
                    "created_on": blob.creation_time.isoformat() if blob.creation_time else None,
                    "metadata": blob.metadata,
                })
        
        return versions
    
    async def delete_model(self, storage_path: str) -> bool:
        """
        Delete a model from blob storage.
        
        Args:
            storage_path: Full storage path (container/blob_path)
            
        Returns:
            True if deleted, False if not found
        """
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            return False
        
        container_name, blob_path = parts
        
        # Check if this is a versioned model (path: model_name/vX.Y/model.ext)
        # Delete the entire version directory to remove all artifacts
        if "/v" in blob_path:
            # Extract directory: model_name/vX.Y/
            path_parts = blob_path.split("/")
            if len(path_parts) >= 2:
                directory_prefix = "/".join(path_parts[:2]) + "/"
                
                container_client = self._get_container_client(container_name)
                deleted_count = 0
                
                try:
                    # List all blobs in the version directory
                    blobs = container_client.list_blobs(name_starts_with=directory_prefix)
                    
                    # Delete each blob
                    for blob in blobs:
                        blob_client = container_client.get_blob_client(blob.name)
                        blob_client.delete_blob(delete_snapshots="include")
                        deleted_count += 1
                        logger.info(f"Deleted blob: {container_name}/{blob.name}")
                    
                    if deleted_count > 0:
                        logger.info(f"Deleted {deleted_count} files from model directory: {container_name}/{directory_prefix}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete model directory: {e}")
                    return False
        
        # Standard single-file deletion
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        try:
            blob_client.delete_blob(delete_snapshots="include")
            logger.info(f"Deleted model: {storage_path}")
            return True
        except ResourceNotFoundError:
            logger.warning(f"Blob already missing: {storage_path}")
            return True
    
    # ============= Utility Methods =============
    
    def generate_sas_url(
        self,
        storage_path: str,
        expiry_hours: int = 1,
        permission: str = "r",
    ) -> str:
        """
        Generate a SAS URL for temporary access to a blob.
        
        Args:
            storage_path: Full storage path
            expiry_hours: Hours until expiry
            permission: 'r' for read, 'w' for write
            
        Returns:
            SAS URL
        """
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid storage path: {storage_path}")
        
        container_name, blob_path = parts
        
        # Parse connection string for account info
        account_name = None
        account_key = None
        for part in self._connection_string.split(";"):
            if part.startswith("AccountName="):
                account_name = part.split("=", 1)[1]
            elif part.startswith("AccountKey="):
                account_key = part.split("=", 1)[1]
        
        if not account_name or not account_key:
            raise ValueError("Could not parse storage account info from connection string")
        
        # Generate SAS token
        sas_permissions = BlobSasPermissions(read="r" in permission, write="w" in permission)
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_path,
            account_key=account_key,
            permission=sas_permissions,
            expiry=now_ist() + timedelta(hours=expiry_hours),
        )
        
        blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_path}?{sas_token}"
        return blob_url
    
    async def get_blob_metadata(self, storage_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a blob.
        
        Args:
            storage_path: Full storage path
            
        Returns:
            Blob properties and metadata
        """
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            return None
        
        container_name, blob_path = parts
        
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        try:
            props = blob_client.get_blob_properties()
            return {
                "size": props.size,
                "content_type": props.content_settings.content_type,
                "created_on": props.creation_time.isoformat() if props.creation_time else None,
                "last_modified": props.last_modified.isoformat() if props.last_modified else None,
                "metadata": props.metadata,
            }
        except ResourceNotFoundError:
            return None

    # ============= Helper Methods =============
    
    async def _upload_json_metadata(
        self,
        container_name: str,
        blob_path: str,
        metadata_dict: Dict[str, Any],
    ) -> str:
        """
        Upload a JSON metadata file to blob storage.
        
        Args:
            container_name: Container name
            blob_path: Blob path (should end with .json)
            metadata_dict: Dictionary to serialize as JSON
            
        Returns:
            Full storage path
        """
        import json
        from azure.storage.blob import ContentSettings
        
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        json_content = json.dumps(metadata_dict, indent=2)
        blob_client.upload_blob(
            json_content,
            overwrite=True,
            content_settings=ContentSettings(content_type='application/json')
        )
        
        logger.info(f"Uploaded JSON metadata: {container_name}/{blob_path}")
        return f"{container_name}/{blob_path}"
    
    async def _download_json_metadata(
        self,
        storage_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Download and parse a JSON metadata file.
        
        Args:
            storage_path: Full storage path
            
        Returns:
            Parsed JSON as dictionary, or None if not found
        """
        import json
        
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            return None
        
        container_name, blob_path = parts
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        try:
            download_stream = blob_client.download_blob()
            content = download_stream.readall()
            return json.loads(content)
        except ResourceNotFoundError:
            return None
    
    def _build_blob_path(self, *parts: str) -> str:
        """
        Build a standardized blob path from parts.
        
        Args:
            *parts: Path components
            
        Returns:
            Joined path with forward slashes
        """
        return "/".join(str(p) for p in parts if p)
    
    def _ensure_container_exists(self, container_name: str) -> ContainerClient:
        """
        Ensure a container exists, creating it if necessary.
        This is an alias for _get_container_client for clarity.
        
        Args:
            container_name: Container name
            
        Returns:
            Container client
        """
        return self._get_container_client(container_name)
    
    # ============= Feature Storage Operations =============
    
    async def upload_feature_definition(
        self,
        feature_set_id: str,
        version: str,
        feature_config: Dict[str, Any],
        feature_schema: Dict[str, Any],
        transformations_code: Optional[str] = None,
    ) -> str:
        """
        Upload feature set definition files.
        
        Args:
            feature_set_id: Feature set ID (e.g., 'fs-20260122-001')
            version: Version (e.g., 'v1.0.0')
            feature_config: Feature configuration dictionary
            feature_schema: Feature schema dictionary
            transformations_code: Optional Python transformation code
            
        Returns:
            Base storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_FEATURES
        base_path = self._build_blob_path("definitions", feature_set_id, version)
        
        # Upload feature_config.json
        config_path = f"{base_path}/feature_config.json"
        await self._upload_json_metadata(container_name, config_path, feature_config)
        
        # Upload feature_schema.json
        schema_path = f"{base_path}/feature_schema.json"
        await self._upload_json_metadata(container_name, schema_path, feature_schema)
        
        # Upload transformations.py if provided
        if transformations_code:
            from azure.storage.blob import ContentSettings
            container_client = self._get_container_client(container_name)
            trans_path = f"{base_path}/transformations.py"
            blob_client = container_client.get_blob_client(trans_path)
            blob_client.upload_blob(
                transformations_code.encode('utf-8'),
                overwrite=True,
                content_settings=ContentSettings(content_type='text/x-python')
            )
        
        logger.info(f"Uploaded feature definition: {container_name}/{base_path}")
        return f"{container_name}/{base_path}"
    
    async def download_feature_definition(
        self,
        feature_set_id: str,
        version: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Download feature set definition.
        
        Args:
            feature_set_id: Feature set ID
            version: Version
            
        Returns:
            Dictionary with 'config' and 'schema' keys
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_FEATURES
        base_path = self._build_blob_path("definitions", feature_set_id, version)
        
        config_path = f"{container_name}/{base_path}/feature_config.json"
        schema_path = f"{container_name}/{base_path}/feature_schema.json"
        
        config = await self._download_json_metadata(config_path)
        schema = await self._download_json_metadata(schema_path)
        
        if config and schema:
            return {"config": config, "schema": schema}
        return None
    
    async def upload_computed_features(
        self,
        feature_set_id: str,
        version: str,
        dataset_id: str,
        features_data: bytes,
        statistics: Dict[str, Any],
    ) -> str:
        """
        Upload computed features for a dataset.
        
        Args:
            feature_set_id: Feature set ID
            version: Feature set version
            dataset_id: Dataset ID
            features_data: Parquet file bytes
            statistics: Feature statistics dictionary
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_FEATURES
        base_path = self._build_blob_path("computed", feature_set_id, version, dataset_id)
        
        # Upload features.parquet
        features_path = f"{base_path}/features.parquet"
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(features_path)
        blob_client.upload_blob(features_data, overwrite=True)
        
        # Upload statistics
        stats_path = f"{base_path}/feature_statistics.json"
        await self._upload_json_metadata(container_name, stats_path, statistics)
        
        logger.info(f"Uploaded computed features: {container_name}/{base_path}")
        return f"{container_name}/{base_path}"
    
    async def download_computed_features(
        self,
        feature_set_id: str,
        version: str,
        dataset_id: str,
    ) -> Optional[bytes]:
        """
        Download computed features parquet file.
        
        Args:
            feature_set_id: Feature set ID
            version: Version
            dataset_id: Dataset ID
            
        Returns:
            Parquet file bytes
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_FEATURES
        blob_path = self._build_blob_path("computed", feature_set_id, version, dataset_id, "features.parquet")
        storage_path = f"{container_name}/{blob_path}"
        
        try:
            return await self.download_dataset(storage_path)
        except FileNotFoundError:
            return None
    
    async def upload_feature_validation(
        self,
        feature_set_id: str,
        version: str,
        validation_report: Dict[str, Any],
        drift_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Upload feature validation reports.
        
        Args:
            feature_set_id: Feature set ID
            version: Version
            validation_report: Validation report dictionary
            drift_analysis: Optional drift analysis
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_FEATURES
        base_path = self._build_blob_path("validation", feature_set_id, version)
        
        # Upload validation report
        report_path = f"{base_path}/validation_report.json"
        await self._upload_json_metadata(container_name, report_path, validation_report)
        
        # Upload drift analysis if provided
        if drift_analysis:
            drift_path = f"{base_path}/drift_analysis.json"
            await self._upload_json_metadata(container_name, drift_path, drift_analysis)
        
        logger.info(f"Uploaded feature validation: {container_name}/{base_path}")
        return f"{container_name}/{base_path}"

    # ============= Monitoring Storage Operations =============
    
    async def upload_drift_report(
        self,
        model_id: str,
        report_date: str,  # Format: YYYY-MM-DD
        drift_report: Dict[str, Any],
        drift_html: Optional[str] = None,
        visualizations: Optional[Dict[str, bytes]] = None,
    ) -> str:
        """
        Upload drift detection report.
        
        Args:
            model_id: Model ID
            report_date: Report date (YYYY-MM-DD)
            drift_report: Drift report dictionary
            drift_html: Optional HTML report
            visualizations: Optional dict of {filename: image_bytes}
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_MONITORING
        base_path = self._build_blob_path("drift", "data-drift", model_id, report_date)
        
        # Upload JSON report
        json_path = f"{base_path}/drift_report.json"
        await self._upload_json_metadata(container_name, json_path, drift_report)
        
        # Upload HTML report if provided
        if drift_html:
            from azure.storage.blob import ContentSettings
            container_client = self._get_container_client(container_name)
            html_path = f"{base_path}/drift_report.html"
            blob_client = container_client.get_blob_client(html_path)
            blob_client.upload_blob(
                drift_html.encode('utf-8'),
                overwrite=True,
                content_settings=ContentSettings(content_type='text/html')
            )
        
        # Upload visualizations if provided
        if visualizations:
            container_client = self._get_container_client(container_name)
            viz_base = f"{base_path}/visualizations"
            for filename, image_bytes in visualizations.items():
                viz_path = f"{viz_base}/{filename}"
                blob_client = container_client.get_blob_client(viz_path)
                blob_client.upload_blob(image_bytes, overwrite=True)
        
        logger.info(f"Uploaded drift report: {container_name}/{base_path}")
        return f"{container_name}/{base_path}"
    
    async def download_drift_report(
        self,
        model_id: str,
        report_date: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Download drift report JSON.
        
        Args:
            model_id: Model ID
            report_date: Report date
            
        Returns:
            Drift report dictionary
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_MONITORING
        blob_path = self._build_blob_path("drift", "data-drift", model_id, report_date, "drift_report.json")
        storage_path = f"{container_name}/{blob_path}"
        
        return await self._download_json_metadata(storage_path)
    
    async def upload_bias_report(
        self,
        model_id: str,
        report_date: str,
        bias_report: Dict[str, Any],
        bias_html: Optional[str] = None,
    ) -> str:
        """
        Upload bias analysis report.
        
        Args:
            model_id: Model ID
            report_date: Report date
            bias_report: Bias report dictionary
            bias_html: Optional HTML report
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_MONITORING
        base_path = self._build_blob_path("bias", model_id, report_date)
        
        # Upload JSON report
        json_path = f"{base_path}/bias_report.json"
        await self._upload_json_metadata(container_name, json_path, bias_report)
        
        # Upload HTML if provided
        if bias_html:
            from azure.storage.blob import ContentSettings
            container_client = self._get_container_client(container_name)
            html_path = f"{base_path}/bias_report.html"
            blob_client = container_client.get_blob_client(html_path)
            blob_client.upload_blob(
                bias_html.encode('utf-8'),
                overwrite=True,
                content_settings=ContentSettings(content_type='text/html')
            )
        
        logger.info(f"Uploaded bias report: {container_name}/{base_path}")
        return f"{container_name}/{base_path}"
    
    async def upload_performance_metrics(
        self,
        model_id: str,
        metrics_date: str,
        daily_metrics: Dict[str, Any],
    ) -> str:
        """
        Upload daily performance metrics.
        
        Args:
            model_id: Model ID
            metrics_date: Metrics date
            daily_metrics: Metrics dictionary
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_MONITORING
        blob_path = self._build_blob_path("performance", model_id, metrics_date, "daily_metrics.json")
        
        await self._upload_json_metadata(container_name, blob_path, daily_metrics)
        
        logger.info(f"Uploaded performance metrics: {container_name}/{blob_path}")
        return f"{container_name}/{blob_path}"
    
    async def upload_alert(
        self,
        alert_id: str,
        alert_date: str,  # YYYY-MM-DD
        alert_data: Dict[str, Any],
        status: str = "triggered",
    ) -> str:
        """
        Upload monitoring alert.
        
        Args:
            alert_id: Alert ID
            alert_date: Alert date
            alert_data: Alert data dictionary
            status: 'triggered' or 'resolved'
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_MONITORING
        
        # Parse date for directory structure
        year, month, day = alert_date.split("-")
        base_path = self._build_blob_path("alerts", status, year, month, day)
        blob_path = f"{base_path}/{alert_id}.json"
        
        await self._upload_json_metadata(container_name, blob_path, alert_data)
        
        logger.info(f"Uploaded alert: {container_name}/{blob_path}")
        return f"{container_name}/{blob_path}"
    
    async def list_drift_reports(
        self,
        model_id: str,
        start_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List drift reports for a model.
        
        Args:
            model_id: Model ID
            start_date: Optional start date filter
            
        Returns:
            List of drift report info
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_MONITORING
        prefix = self._build_blob_path("drift", "data-drift", model_id)
        
        container_client = self._get_container_client(container_name)
        
        reports = []
        for blob in container_client.list_blobs(name_starts_with=prefix):
            if blob.name.endswith("drift_report.json"):
                # Extract date from path
                parts = blob.name.split("/")
                if len(parts) >= 4:
                    report_date = parts[3]
                    if not start_date or report_date >= start_date:
                        reports.append({
                            "date": report_date,
                            "path": blob.name,
                            "size": blob.size,
                            "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                        })
        
        return sorted(reports, key=lambda x: x["date"], reverse=True)
    # ============= Audit Log Operations =============
    
    async def append_prediction_log(
        self,
        predictions: List[Dict[str, Any]],
        timestamp: datetime,
    ) -> str:
        """
        Append predictions to JSONL log file (batch operation).
        
        Args:
            predictions: List of prediction dictionaries
            timestamp: Timestamp for organizing logs
            
        Returns:
            Storage path
        """
        import json
        
        container_name = settings.AZURE_STORAGE_CONTAINER_AUDIT_LOGS
        
        # Build path: predictions/YYYY/MM/DD/HH/predictions_TIMESTAMP.jsonl
        year = timestamp.strftime("%Y")
        month = timestamp.strftime("%m")
        day = timestamp.strftime("%d")
        hour = timestamp.strftime("%H")
        ts_str = timestamp.strftime("%Y%m%dT%H%M%S")
        
        base_path = self._build_blob_path("predictions", year, month, day, hour)
        blob_path = f"{base_path}/predictions_{ts_str}.jsonl"
        
        # Convert predictions to JSONL format
        jsonl_content = "\n".join(json.dumps(pred) for pred in predictions)
        
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        # Append mode (or create if doesn't exist)
        blob_client.upload_blob(
            jsonl_content.encode('utf-8'),
            overwrite=False,  # Don't overwrite existing
        )
        
        logger.info(f"Appended {len(predictions)} predictions: {container_name}/{blob_path}")
        return f"{container_name}/{blob_path}"
    
    async def upload_model_lineage(
        self,
        model_id: str,
        lineage_graph: Dict[str, Any],
        training_lineage: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Upload model lineage information.
        
        Args:
            model_id: Model ID
            lineage_graph: Complete lineage graph
            training_lineage: Optional training lineage details
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_AUDIT_LOGS
        base_path = self._build_blob_path("model-lineage", model_id)
        
        # Upload lineage graph
        graph_path = f"{base_path}/lineage_graph.json"
        await self._upload_json_metadata(container_name, graph_path, lineage_graph)
        
        # Upload training lineage if provided
        if training_lineage:
            training

    # ============= Experiment Operations =============
    
    async def upload_experiment_run(
        self,
        experiment_id: str,
        run_id: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        artifacts: Optional[Dict[str, bytes]] = None,
    ) -> str:
        """
        Upload experiment run data.
        
        Args:
            experiment_id: Experiment ID
            run_id: Run ID
            hyperparameters: Hyperparameters dictionary
            metrics: Metrics dictionary
            artifacts: Optional artifacts {filename: bytes}
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_EXPERIMENTS
        base_path = self._build_blob_path("training-experiments", experiment_id, "runs", run_id)
        
        # Upload hyperparameters
        hyper_path = f"{base_path}/hyperparameters.json"
        await self._upload_json_metadata(container_name, hyper_path, hyperparameters)
        
        # Upload metrics
        metrics_path = f"{base_path}/metrics.json"
        await self._upload_json_metadata(container_name, metrics_path, metrics)
        
        # Upload artifacts if provided
        if artifacts:
            container_client = self._get_container_client(container_name)
            artifacts_base = f"{base_path}/artifacts"
            for filename, file_bytes in artifacts.items():
                artifact_path = f"{artifacts_base}/{filename}"
                blob_client = container_client.get_blob_client(artifact_path)
                blob_client.upload_blob(file_bytes, overwrite=True)
        
        logger.info(f"Uploaded experiment run: {container_name}/{base_path}")
        return f"{container_name}/{base_path}"
    
    async def upload_ab_test_results(
        self,
        test_id: str,
        test_config: Dict[str, Any],
        champion_results: Dict[str, Any],
        challenger_results: Dict[str, Any],
        statistical_analysis: Dict[str, Any],
    ) -> str:
        """
        Upload A/B test results.
        
        Args:
            test_id: A/B test ID
            test_config: Test configuration
            champion_results: Champion model results
            challenger_results: Challenger model results
            statistical_analysis: Statistical analysis results
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_EXPERIMENTS
        base_path = self._build_blob_path("ab-tests", test_id)
        
        # Upload all components
        await self._upload_json_metadata(container_name, f"{base_path}/test_config.json", test_config)
        await self._upload_json_metadata(container_name, f"{base_path}/champion_results.json", champion_results)
        await self._upload_json_metadata(container_name, f"{base_path}/challenger_results.json", challenger_results)
        await self._upload_json_metadata(container_name, f"{base_path}/statistical_analysis.json", statistical_analysis)
        
        logger.info(f"Uploaded A/B test results: {container_name}/{base_path}")
        return f"{container_name}/{base_path}"
    
    async def list_experiment_runs(
        self,
        experiment_id: str,
    ) -> List[Dict[str, Any]]:
        """
        List all runs for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            List of run info
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_EXPERIMENTS
        prefix = self._build_blob_path("training-experiments", experiment_id, "runs")
        
        container_client = self._get_container_client(container_name)
        
        runs = []
        seen_runs = set()
        
        for blob in container_client.list_blobs(name_starts_with=prefix):
            # Extract run_id from path
            parts = blob.name.split("/")
            if len(parts) >= 4:
                run_id = parts[3]
                if run_id not in seen_runs:
                    seen_runs.add(run_id)
                    runs.append({
                        "run_id": run_id,
                        "path": f"{prefix}/{run_id}",
                    })
        
        return runs

    # ============= Backup Operations =============
    
    async def upload_backup(
        self,
        backup_type: str,  # 'database', 'models', 'configurations'
        backup_data: bytes,
        backup_date: str,  # YYYY-MM-DD
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload backup file.
        
        Args:
            backup_type: Type of backup
            backup_data: Backup file bytes
            backup_date: Backup date
            metadata: Optional metadata
            
        Returns:
            Storage path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_BACKUPS
        
        # Parse date for directory structure
        year, month, day = backup_date.split("-")
        timestamp = now_ist().strftime("%Y%m%dT%H%M%S")
        
        # Determine file extension based on type
        ext = "sql.gz" if backup_type == "database" else "tar.gz"
        filename = f"{backup_type}_backup_{timestamp}.{ext}"
        
        blob_path = self._build_blob_path(backup_type, year, month, day, filename)
        
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        blob_metadata = metadata or {}
        blob_metadata.update({
            "backup_type": backup_type,
            "backup_date": backup_date,
            "created_at": now_ist().isoformat(),
        })
        
        blob_client.upload_blob(
            backup_data,
            overwrite=True,
            metadata=blob_metadata,
        )
        
        logger.info(f"Uploaded backup: {container_name}/{blob_path}")
        return f"{container_name}/{blob_path}"
    
    async def download_backup(
        self,
        storage_path: str,
    ) -> bytes:
        """
        Download backup file.
        
        Args:
            storage_path: Full storage path
            
        Returns:
            Backup file bytes
        """
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid storage path: {storage_path}")
        
        container_name, blob_path = parts
        container_client = self._get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        try:
            download_stream = blob_client.download_blob()
            return download_stream.readall()
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Backup not found: {storage_path}")
    
    async def list_backups(
        self,
        backup_type: str,
        start_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List available backups by type.
        
        Args:
            backup_type: Type of backup
            start_date: Optional start date filter (YYYY-MM-DD)
            
        Returns:
            List of backup info
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_BACKUPS
        prefix = f"{backup_type}/"
        
        container_client = self._get_container_client(container_name)
        
        backups = []
        for blob in container_client.list_blobs(name_starts_with=prefix):
            # Extract date from path
            parts = blob.name.split("/")
            if len(parts) >= 4:
                backup_date = f"{parts[1]}-{parts[2]}-{parts[3]}"
                if not start_date or backup_date >= start_date:
                    backups.append({
                        "date": backup_date,
                        "path": blob.name,
                        "size": blob.size,
                        "created": blob.creation_time.isoformat() if blob.creation_time else None,
                        "metadata": blob.metadata,
                    })
        
        return sorted(backups, key=lambda x: x["date"], reverse=True)

    # ============= Temp Processing Operations =============
    
    async def create_temp_workspace(
        self,
        job_id: str,
        job_type: str,  # 'feature-computation', 'model-training', 'data-validation'
    ) -> str:
        """
        Create a temporary workspace for a processing job.
        
        Args:
            job_id: Job ID
            job_type: Type of job
            
        Returns:
            Workspace base path
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_TEMP_PROCESSING
        base_path = self._build_blob_path(job_type, job_id)
        
        # Create a placeholder file to establish the directory
        container_client = self._get_container_client(container_name)
        placeholder_path = f"{base_path}/.workspace"
        blob_client = container_client.get_blob_client(placeholder_path)
        
        workspace_info = {
            "job_id": job_id,
            "job_type": job_type,
            "created_at": now_ist().isoformat(),
        }
        
        import json
        blob_client.upload_blob(
            json.dumps(workspace_info).encode('utf-8'),
            overwrite=True,
        )
        
        logger.info(f"Created temp workspace: {container_name}/{base_path}")
        return f"{container_name}/{base_path}"
    
    async def cleanup_temp_workspace(
        self,
        workspace_path: str,
    ) -> bool:
        """
        Delete a temporary workspace and all its contents.
        
        Args:
            workspace_path: Workspace path (container/job_type/job_id)
            
        Returns:
            True if deleted successfully
        """
        parts = workspace_path.split("/", 1)
        if len(parts) != 2:
            return False
        
        container_name, base_path = parts
        container_client = self._get_container_client(container_name)
        
        # Delete all blobs with this prefix
        deleted_count = 0
        for blob in container_client.list_blobs(name_starts_with=base_path):
            blob_client = container_client.get_blob_client(blob.name)
            blob_client.delete_blob()
            deleted_count += 1
        
        logger.info(f"Cleaned up temp workspace: {workspace_path} ({deleted_count} files)")
        return True
    
    async def cleanup_old_temp_files(
        self,
        days_old: int = 7,
    ) -> int:
        """
        Auto-cleanup temporary files older than specified days.
        
        Args:
            days_old: Delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        container_name = settings.AZURE_STORAGE_CONTAINER_TEMP_PROCESSING
        container_client = self._get_container_client(container_name)
        
        cutoff_date = now_ist() - timedelta(days=days_old)
        deleted_count = 0
        
        for blob in container_client.list_blobs():
            if blob.last_modified and blob.last_modified < cutoff_date:
                blob_client = container_client.get_blob_client(blob.name)
                blob_client.delete_blob()
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} temp files older than {days_old} days")
        return deleted_count

# Singleton instance
storage_service = StorageService()