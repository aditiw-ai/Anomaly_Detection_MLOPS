"""
DatasetLineage SQLAlchemy Model
Tracks lineage relationships between datasets (merged from, split from).
"""
from app.core.time import IST, now_ist
from datetime import datetime
import uuid

from sqlalchemy import Column, String, ForeignKey, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class DatasetLineage(Base):
    """Dataset lineage model for tracking data provenance."""
    
    __tablename__ = "dataset_lineage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Lineage relationship
    target_dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )  # The derived dataset (merged or split)
    
    source_dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )  # The source dataset
    
    # Relationship type
    relationship_type = Column(String(50), nullable=False, index=True)  # 'merged_from', 'split_from'
    
    # Additional metadata (renamed to avoid SQLAlchemy reserved word)
    lineage_metadata = Column(JSON, nullable=True)  # Additional lineage info (merge config, split ratio, etc.)
    
    # Audit
    created_at = Column(DateTime, default=lambda: now_ist().replace(tzinfo=None), index=True)
    
    def __repr__(self):
        return f"<DatasetLineage {self.relationship_type}: {self.source_dataset_id} -> {self.target_dataset_id}>"
