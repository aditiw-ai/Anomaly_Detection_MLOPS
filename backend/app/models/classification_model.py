from app.core.time import IST, now_ist
from datetime import datetime
import uuid
from sqlalchemy import Column, String, Text, Boolean, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base


class ClassificationModel(Base):
    """
    Registry of available classification model types (algorithms).
    Each row represents a type of ML algorithm (e.g., XGBoost, LightGBM).
    """
    __tablename__ = "classification_models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    algorithm_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String(50), nullable=False)  # supervised | unsupervised
    
    # JSON schema defining hyperparameters for this algorithm
    hyperparameters_schema = Column(JSON, nullable=False)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: now_ist().replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: now_ist().replace(tzinfo=None), onupdate=lambda: now_ist().replace(tzinfo=None))
    
    def __repr__(self):
        return f"<ClassificationModel {self.name} ({self.algorithm_id})>"
