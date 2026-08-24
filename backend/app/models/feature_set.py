"""
Feature Set SQLAlchemy Model
Database model for computed feature sets.
"""
from app.core.time import IST, now_ist
from datetime import datetime
import uuid

from sqlalchemy import Column, String, Integer, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class FeatureSet(Base):
    """Feature set model for computed features."""
    
    __tablename__ = "feature_sets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=False, default="1.0")
    
    # Configuration
    config = Column(JSON, nullable=False)  # Feature engineering config
    
    # Results
    all_features = Column(JSON, nullable=True)  # All generated features
    selected_features = Column(JSON, nullable=True)  # After selection
    selection_report = Column(JSON, nullable=True)  # MI scores, rankings
    preprocessing_profile = Column(JSON, nullable=True)
    preprocessing_recommendations = Column(JSON, nullable=True)
    preprocessing_decisions = Column(JSON, nullable=True)
    preprocessing_audit = Column(JSON, nullable=True)
    preprocessing_plan_version = Column(Integer, nullable=False, default=0)
    preprocessing_committed_at = Column(DateTime, nullable=True)
    preprocessing_committed_by = Column(String(255), nullable=True)
    
    # Storage
    storage_path = Column(String(500), nullable=True)
    
    # Stats
    input_rows = Column(Integer, nullable=True)
    feature_count = Column(Integer, nullable=True)
    selected_feature_count = Column(Integer, nullable=True)
    
    # Status
    status = Column(String(50), default="PENDING", index=True)  # PENDING, PROCESSING, COMPLETED, FAILED
    error_message = Column(Text, nullable=True)
    processing_time_seconds = Column(Integer, nullable=True)
    
    # Audit
    # Store auth subject/user ID as text
    created_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: now_ist().replace(tzinfo=None))
    completed_at = Column(DateTime, nullable=True)

    # Registry / Lifecycle
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    config_hash = Column(String(64), nullable=True, index=True)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="feature_sets")
    
    def __repr__(self):
        return f"<FeatureSet {self.name} v{self.version}>"
