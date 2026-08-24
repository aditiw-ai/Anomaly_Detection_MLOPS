"""
Dataset Pydantic Schemas
Request/Response models for Dataset API.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class DatasetBase(BaseModel):
    """Base dataset schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class DatasetCreate(DatasetBase):
    """Schema for creating a dataset."""
    pass


class DatasetSchema(DatasetBase):
    """Full dataset schema for responses."""
    id: UUID
    version: str
    storage_path: str
    file_format: str
    file_size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    schema: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None
    
    
    # Hierarchical storage (Phase 2)
    # dataset_type: str = "raw"  # raw, merged, split
    
    # Expose parent_id for frontend filtering of merged datasets
    parent_id: Optional[UUID] = None
    
    # split_type: Optional[str] = None  # train, test, validation
    # split_job_id: Optional[UUID] = None
    
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DatasetResponse(BaseModel):
    """Single dataset response."""
    data: DatasetSchema


class DatasetListResponse(BaseModel):
    """Paginated dataset list response."""
    data: List[DatasetSchema]
    meta: Dict[str, Any]


class DatasetPreviewResponse(BaseModel):
    """Dataset preview response."""
    data: Dict[str, Any]  # Contains columns and rows


class DatasetMergeRequest(BaseModel):
    """Request schema for merging datasets."""
    dataset_ids: List[str]
    new_name: str
    description: Optional[str] = None
