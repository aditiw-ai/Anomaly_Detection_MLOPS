"""
Training Job SQLAlchemy Model
Database model for tracking training job execution.
"""
from app.core.time import IST, now_ist
from datetime import datetime
import uuid

from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, Boolean, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class TrainingJob(Base):
    """Training Job for tracking execution."""
    
    __tablename__ = "training_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    
    # Configuration
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    feature_set_id = Column(UUID(as_uuid=True), ForeignKey("feature_sets.id"), nullable=True)
    algorithm = Column(String(100), nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    tuning_method = Column(String(50), default="manual")  # manual, grid, random, bayesian
    tuning_config = Column(JSON, nullable=True)
    feature_config = Column(JSON, nullable=False)
    imbalanced_strategy = Column(String(50), nullable=True)
    test_size = Column(Float, default=0.2)
    
    # Execution
    status = Column(String(50), default="QUEUED", index=True)  # QUEUED, RUNNING, COMPLETED, FAILED, DATA_PREPARED
    progress = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    logs = Column(Text, nullable=True)
    
    # Modes
    processing_only = Column(Boolean, default=False)
    
    # Artifacts (Output)
    model_id = Column(UUID(as_uuid=True), ForeignKey("ml_models.id"), nullable=True)
    metrics = Column(JSON, nullable=True)
    
    # Audit
    created_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: now_ist().replace(tzinfo=None), index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    dataset = relationship("Dataset")
    feature_set = relationship("FeatureSet")
    model = relationship("MLModel")
    
    def __repr__(self):
        return f"<TrainingJob {self.name} ({self.status})>"
