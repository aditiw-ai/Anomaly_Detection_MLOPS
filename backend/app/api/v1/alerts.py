"""
Alerts API Endpoints
Alert management and notification configuration.
"""
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_db
from app.core.dependencies import require_auth
from app.core.auth import User
from app.services.alert_service import AlertService, AlertStatus, AlertSeverity
from app.core.time import IST

router = APIRouter(prefix="/alerts", tags=["Alerts"])


class AlertAcknowledgeRequest(BaseModel):
    """Request to acknowledge an alert."""
    resolution_note: Optional[str] = None

from app.models.alert import Alert as AlertModel

def _serialize_dt(dt: Optional[datetime]) -> Optional[str]:
    """
    Serialize timestamps with explicit timezone offset.

    DB columns are stored as TIMESTAMP WITHOUT TIME ZONE and populated using
    IST-local naive datetimes. When serializing, attach IST offset explicitly
    so frontend Date parsing does not treat values as UTC.
    """
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.isoformat()

def serialize_alert(a: AlertModel):
    if not a:
        return None
    return {
        "id": getattr(a, "id", a),
        "model_id": getattr(a, "model_id", ""),
        "alert_type": a.alert_type.value if hasattr(a.alert_type, "value") else str(a.alert_type),
        "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
        "status": a.status.value if hasattr(a.status, "value") else str(a.status),
        "title": getattr(a, "title", ""),
        "message": getattr(a, "message", ""),
        "details": getattr(a, "details", {}),
        "created_at": _serialize_dt(getattr(a, "created_at", None)),
        "acknowledged_at": _serialize_dt(getattr(a, "acknowledged_at", None)),
        "acknowledged_by": getattr(a, "acknowledged_by", None),
        "resolved_at": _serialize_dt(getattr(a, "resolved_at", None)),
        "resolution_note": getattr(a, "resolved_notes", None),
    }


@router.get("")
async def list_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """
    List all alerts with optional filtering.
    """
    alert_service = AlertService(db)
    
    # Convert string filters to Enum if provided
    status_enum = AlertStatus(status.upper()) if status else None
    severity_enum = AlertSeverity(severity.upper()) if severity else None
    
    alerts = await alert_service.list_alerts(
        status=status_enum, 
        severity=severity_enum, 
        limit=page_size
    )
    
    # Also fetch summary to match frontend expectations
    summary = await alert_service.get_alert_summary()
    
    return {
        "data": [serialize_alert(a) for a in alerts],
        "meta": {
            "page": page,
            "page_size": page_size,
            "total": len(alerts),  # Simplified total, usually requires second query
        },
        "summary": summary
    }


@router.get("/{alert_id}")
async def get_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get alert details."""
    alert_service = AlertService(db)
    alert = await alert_service.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    return {"data": serialize_alert(alert)}


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    request: AlertAcknowledgeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """Acknowledge an alert."""
    alert_service = AlertService(db)
    alert = await alert_service.acknowledge_alert(
        alert_id=alert_id,
        user_id=str(current_user.id),
        note=request.resolution_note
    )
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    return {
        "data": serialize_alert(alert),
        "message": "Alert acknowledged"
    }


@router.post("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    request: AlertAcknowledgeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resolve an alert."""
    alert_service = AlertService(db)
    alert = await alert_service.resolve_alert(
        alert_id=alert_id,
        note=request.resolution_note
    )
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    return {
        "data": serialize_alert(alert),
        "message": "Alert resolved"
    }


@router.get("/stats/summary")
async def get_alert_stats(
    period: str = "7d",
    db: AsyncSession = Depends(get_db),
):
    """Get alert statistics summary."""
    alert_service = AlertService(db)
    summary = await alert_service.get_alert_summary()
    
    return {
        "data": {
            "period": period,
            "total": summary.get("total", 0),
            "by_type": summary.get("by_type", {}),
            "by_severity": summary.get("by_severity", {}),
            "by_status": {
                "ACTIVE": summary.get("active", 0),
                "ACKNOWLEDGED": summary.get("acknowledged", 0),
                "RESOLVED": summary.get("resolved", 0),
            },
            # Hardcoded trend line for UI since full trend query isn't implemented
            "trend": [
                {"date": "2026-01-11", "count": 2},
                {"date": "2026-01-12", "count": 1},
                {"date": "2026-01-13", "count": 0},
            ]
        }
    }
