"""
Dataset SQLAlchemy Model
Database model for dataset management.
"""
from typing import Optional
import uuid

from sqlalchemy import Column, String, Integer, BigInteger, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.core.time import now_ist


class Dataset(Base):
    """Dataset model for storing uploaded datasets metadata."""
    
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=False, default="1.0")
    
    # Storage
    storage_path = Column(String(500), nullable=False)
    file_format = Column(String(50), default="parquet")
    file_size_bytes = Column(BigInteger, nullable=True)
    
    # Data info
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    schema = Column(JSON, nullable=True)  # Column definitions
    statistics = Column(JSON, nullable=True)  # Column statistics
    
    # Dataset organization and lineage
    dataset_type = Column(String(50), nullable=False, default="raw", server_default="raw", index=True)
    parent_dataset_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    split_type = Column(String(50), nullable=True, index=True)
    split_job_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    # Status and lifecycle
    status = Column(String(50), default="ACTIVE", index=True)  # ACTIVE, ARCHIVED, PROCESSING
    parent_id = Column(UUID(as_uuid=True), nullable=True)  # Legacy: use parent_dataset_id instead

    # Audit
    # Auth user IDs are string subjects (local UUID strings or Azure B2C subject IDs)
    created_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=now_ist, index=True)
    updated_at = Column(DateTime, default=now_ist, onupdate=now_ist)
    
    # Relationships
    feature_sets = relationship("FeatureSet", back_populates="dataset", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Dataset {self.name} v{self.version}>"
