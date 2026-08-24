"""
ML Model SQLAlchemy Model
Database model for trained machine learning models.
"""
from app.core.time import IST, now_ist
from datetime import datetime
import uuid

from sqlalchemy import Column, String, BigInteger, Text, DateTime, JSON, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class MLModel(Base):
    """ML Model for trained models."""
    
    __tablename__ = "ml_models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    
    # Training info
    algorithm = Column(String(100), nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    feature_set_id = Column(UUID(as_uuid=True), ForeignKey("feature_sets.id"), nullable=True)
    feature_set_version = Column(String(50), nullable=True)
    
    # Artifacts
    storage_path = Column(String(500), nullable=False)
    onnx_path = Column(String(500), nullable=True)  # ONNX model for inference
    model_size_bytes = Column(BigInteger, nullable=True)
    checksum = Column(String(64), nullable=True)  # SHA-256
    
    # Metrics
    metrics = Column(JSON, nullable=False)  # precision, recall, f1, auc
    feature_names = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    
    # Lifecycle
    status = Column(String(50), default="TRAINED", index=True)  # TRAINED, STAGING, PRODUCTION, ARCHIVED
    promoted_at = Column(DateTime, nullable=True)
    archived_at = Column(DateTime, nullable=True)
    archived_reason = Column(Text, nullable=True)
    
    # Audit
    # Store auth subject/user ID as text
    created_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: now_ist().replace(tzinfo=None), index=True)
    updated_at = Column(DateTime, default=lambda: now_ist().replace(tzinfo=None), onupdate=lambda: now_ist().replace(tzinfo=None))
    
    # Relationships
    baselines = relationship("Baseline", back_populates="model", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<MLModel {self.name} v{self.version} ({self.status})>"


class Baseline(Base):
    """Baseline thresholds for model monitoring."""
    
    __tablename__ = "baselines"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False)
    
    metric_name = Column(String(100), nullable=False)  # precision, recall, f1, fpr
    threshold = Column(Float, nullable=False)
    operator = Column(String(10), nullable=False)  # gte, lte, eq
    is_active = Column(String(10), default="true")
    
    # Audit
    # Store auth subject/user ID as text
    created_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: now_ist().replace(tzinfo=None))
    
    # Relationships
    model = relationship("MLModel", back_populates="baselines")
    
    def __repr__(self):
        return f"<Baseline {self.metric_name} {self.operator} {self.threshold}>"
