import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import String, Float, Integer, Boolean, DateTime, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

def utcnow_aware():
    return datetime.now(timezone.utc)


class InferenceLog(Base):
    """
    SQLAlchemy model for storing real-time predictions and subsequent ground truth.
    Used for monitoring model drift, performance degradation, and automated retraining triggers.
    """
    __tablename__ = "inference_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    transaction_id: Mapped[str] = mapped_column(
        String, 
        nullable=True, 
        index=True,
        doc="Optional external ID for the transaction to join back to source systems."
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        nullable=False, 
        index=True,
        doc="The UUID of the MLModel that made this prediction."
    )
    features: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, 
        nullable=False,
        doc="The full input feature dictionary at inference time."
    )
    prediction: Mapped[int] = mapped_column(
        Integer, 
        nullable=False,
        doc="Binary classification output: 0 (Legit) or 1 (Fraud)"
    )
    fraud_score: Mapped[float] = mapped_column(
        Float, 
        nullable=False,
        doc="Probability score between 0.0 and 1.0"
    )
    actual_label: Mapped[int] = mapped_column(
        Integer, 
        nullable=True,
        doc="Ground truth label populated asynchronously via feedback endpoint. 0 or 1."
    )
    response_time_ms: Mapped[float] = mapped_column(
        Float, 
        nullable=True,
        doc="How long the inference engine took to predict."
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=utcnow_aware, 
        nullable=False,
        index=True
    )

    __table_args__ = (
        CheckConstraint('prediction IN (0, 1)', name='check_prediction_binary'),
        CheckConstraint('actual_label IS NULL OR actual_label IN (0, 1)', name='check_actual_label_binary'),
        CheckConstraint('fraud_score >= 0.0 AND fraud_score <= 1.0', name='check_fraud_score_range'),
    )
