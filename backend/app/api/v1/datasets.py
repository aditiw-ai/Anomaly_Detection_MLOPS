"""
Datasets API Endpoints
CRUD operations for dataset management.
"""
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import require_auth, require_data_write
from app.core.auth import User, Role
from app.schemas.dataset import (
    DatasetResponse,
    DatasetListResponse,
    DatasetPreviewResponse,
    DatasetMergeRequest,
)
from app.services.data_service import DataService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["Datasets"])


def _get_user_scope_id(current_user: User) -> Optional[str]:
    """
    Return dataset ownership scope.
    - Admins can access all datasets (no user filter).
    - Non-admin users are limited to their own datasets.
    """
    return None if Role.ADMIN in current_user.roles else str(current_user.id)


@router.post("/merge", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def merge_datasets(
    request: DatasetMergeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_data_write),
):
    """
    Merge multiple datasets into a new one.

    - **dataset_ids**: List of dataset IDs to merge.
    - **new_name**: Name for the new merged dataset.
    """
    service = DataService(db)
    user_scope_id = _get_user_scope_id(current_user)
    try:
        dataset = await service.merge_datasets(
            dataset_ids=request.dataset_ids,
            new_name=request.new_name,
            description=request.description,
            user_id=user_scope_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return DatasetResponse(data=dataset)


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    include_merged: bool = False,
    dataset_type: Optional[str] = None,  # Phase 2: Filter by type
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    List all datasets with pagination and optional filtering.

    - **page**: Page number (1-indexed)
    - **page_size**: Number of items per page (max 100)
    - **status**: Filter by status (ACTIVE, ARCHIVED, PROCESSING)
    - **include_merged**: Whether to include merged datasets (default: False)
    - **dataset_type**: Filter by dataset type (raw, merged, split)
    """
    service = DataService(db)
    user_scope_id = _get_user_scope_id(current_user)
    datasets, total = await service.list_datasets(
        page=page,
        page_size=min(page_size, 100),
        status=status,
        include_merged=include_merged,
        dataset_type=dataset_type,  # Phase 2: Pass to service
        user_id=user_scope_id
    )
    return DatasetListResponse(
        data=datasets,
        meta={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_data_write),
):
    """
    Upload a new dataset.

    - **name**: Dataset name (required)
    - **description**: Optional description
    - **file**: Dataset file (CSV, Parquet, JSON)

    The file will be validated for schema and uploaded to Azure Blob Storage.
    """
    # Validate file type
    allowed_types = ["text/csv", "application/octet-stream", "application/json"]
    if file.content_type not in allowed_types and not file.filename.endswith(('.csv', '.parquet', '.json')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Allowed: CSV, Parquet, JSON"
        )

    service = DataService(db)
    try:
        dataset = await service.create_dataset(
            name=name,
            description=description,
            file=file,
            user_id=str(current_user.id)
        )
    except ValueError as e:
        logger.error(f"Validation error creating dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Unexpected error creating dataset")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dataset: {str(e)}"
        )
    return DatasetResponse(data=dataset)


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """Get dataset details by ID."""
    service = DataService(db)
    user_scope_id = _get_user_scope_id(current_user)
    dataset = await service.get_dataset(str(dataset_id), user_id=user_scope_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    return DatasetResponse(data=dataset)


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def preview_dataset(
    dataset_id: UUID,
    rows: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    Preview dataset rows.

    - **rows**: Number of rows to preview (max 100)
    """
    logger.info(f"Previewing dataset: {dataset_id}")
    service = DataService(db)
    user_scope_id = _get_user_scope_id(current_user)
    preview = await service.preview_dataset(str(dataset_id), rows=min(rows, 100), user_id=user_scope_id)
    if not preview:
        logger.error(f"Preview failed (None returned) for dataset: {dataset_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    if "error" in preview and preview["error"]:
        logger.error(f"Preview error for dataset {dataset_id}: {preview['error']}")
    else:
        logger.info(f"Preview success for dataset {dataset_id}, rows: {preview.get('preview_rows')}")
    return DatasetPreviewResponse(data=preview)


@router.get("/{dataset_id}/schema")
async def get_dataset_schema(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """Get dataset schema with column types and statistics."""
    service = DataService(db)
    user_scope_id = _get_user_scope_id(current_user)
    dataset = await service.get_dataset(str(dataset_id), user_id=user_scope_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    return {
        "data": {
            "columns": dataset.schema.get("columns", []) if dataset.schema else [],
            "row_count": dataset.row_count,
            "statistics": dataset.statistics
        }
    }


@router.get("/{dataset_id}/download")
async def get_dataset_download_url(
    dataset_id: UUID,
    expiry_hours: int = 1,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    Generate a temporary download URL for a dataset.

    - **expiry_hours**: Hours until the URL expires (default: 1, max: 24)
    """
    service = DataService(db)
    user_scope_id = _get_user_scope_id(current_user)
    download_url = await service.get_dataset_download_url(
        str(dataset_id),
        expiry_hours=min(expiry_hours, 24),
        user_id=user_scope_id
    )
    if not download_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    return {"data": {"download_url": download_url, "expires_in_hours": expiry_hours}}


@router.get("/{dataset_id}/lineage")
async def get_dataset_lineage(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    Get lineage records for a dataset.

    Returns all lineage relationships where this dataset is either
    the source (e.g. a raw dataset that was merged) or the target
    (e.g. a merged dataset derived from sources).
    """
    from sqlalchemy import select, or_
    from app.models.dataset_lineage import DatasetLineage
    from app.models.dataset import Dataset
    user_scope_id = _get_user_scope_id(current_user)

    # Confirm dataset exists
    ds_stmt = select(Dataset).where(Dataset.id == dataset_id)
    if user_scope_id:
        ds_stmt = ds_stmt.where(Dataset.created_by == user_scope_id)
    ds_result = await db.execute(ds_stmt)
    if not ds_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found",
        )

    # Fetch all lineage rows for this dataset
    lineage_result = await db.execute(
        select(DatasetLineage).where(
            or_(
                DatasetLineage.source_dataset_id == dataset_id,
                DatasetLineage.target_dataset_id == dataset_id,
            )
        )
    )
    records = lineage_result.scalars().all()

    # Collect all related dataset IDs to resolve names in one query
    related_ids = set()
    for r in records:
        related_ids.add(r.source_dataset_id)
        related_ids.add(r.target_dataset_id)

    name_map: dict = {}
    if related_ids:
        names_result = await db.execute(
            select(Dataset.id, Dataset.name).where(Dataset.id.in_(related_ids))
        )
        name_map = {row.id: row.name for row in names_result}

    lineage_list = [
        {
            "id": str(r.id),
            "source_dataset_id": str(r.source_dataset_id),
            "source_dataset_name": name_map.get(r.source_dataset_id, "Unknown"),
            "target_dataset_id": str(r.target_dataset_id),
            "target_dataset_name": name_map.get(r.target_dataset_id, "Unknown"),
            "relationship_type": r.relationship_type,
            "lineage_metadata": r.lineage_metadata,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]

    return {"data": lineage_list, "total": len(lineage_list)}


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_data_write),
):
    """Delete a dataset (hard delete - removes from database and storage)."""
    service = DataService(db)
    user_scope_id = _get_user_scope_id(current_user)
    try:
        success = await service.delete_dataset(str(dataset_id), hard_delete=True, user_id=user_scope_id)
    except Exception as e:
        logger.exception("Dataset delete failed: %s", e)
        await db.rollback()
        # Fallback: force-remove via raw SQL (bypasses ORM/session issues)
        try:
            await _force_remove_dataset_by_id(str(dataset_id))
        except Exception as e2:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e2),
            )
        return None
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    return None


async def _force_remove_dataset_by_id(dataset_id: str) -> None:
    """Remove dataset and dependents using raw SQL. Use when ORM delete fails."""
    from sqlalchemy import text
    from app.core.database import engine
    async with engine.begin() as conn:
        # Delete training_jobs that directly reference this dataset
        await conn.execute(
            text("DELETE FROM training_jobs WHERE dataset_id = :id"),
            {"id": dataset_id},
        )
        # Delete training_jobs that reference feature_sets for this dataset
        await conn.execute(
            text("DELETE FROM training_jobs WHERE feature_set_id IN (SELECT id FROM feature_sets WHERE dataset_id = :id)"),
            {"id": dataset_id},
        )
        await conn.execute(
            text("UPDATE ml_models SET feature_set_id = NULL WHERE feature_set_id IN (SELECT id FROM feature_sets WHERE dataset_id = :id)"),
            {"id": dataset_id},
        )
        await conn.execute(text("DELETE FROM feature_sets WHERE dataset_id = :id"), {"id": dataset_id})
        await conn.execute(
            text("DELETE FROM dataset_lineage WHERE source_dataset_id = :id OR target_dataset_id = :id"),
            {"id": dataset_id},
        )
        result = await conn.execute(text("DELETE FROM datasets WHERE id = :id RETURNING id"), {"id": dataset_id})
        if result.fetchone() is None:
            raise ValueError(f"Dataset {dataset_id} not found")