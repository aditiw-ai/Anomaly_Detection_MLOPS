"""
Audit Logging Service
Track all user actions for compliance and security.
"""
from app.core.time import IST, now_ist
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import uuid4
import logging
import json

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Audit action types."""
    # Authentication
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    LOGIN_FAILED = "LOGIN_FAILED"
    
    # Data operations
    DATA_UPLOAD = "DATA_UPLOAD"
    DATA_DELETE = "DATA_DELETE"
    DATA_DOWNLOAD = "DATA_DOWNLOAD"
    
    # Model operations
    MODEL_TRAIN = "MODEL_TRAIN"
    MODEL_DEPLOY = "MODEL_DEPLOY"
    MODEL_DELETE = "MODEL_DELETE"
    MODEL_PROMOTE = "MODEL_PROMOTE"
    
    # Inference
    PREDICTION_SINGLE = "PREDICTION_SINGLE"
    PREDICTION_BATCH = "PREDICTION_BATCH"
    
    # Jobs
    JOB_CREATE = "JOB_CREATE"
    JOB_DELETE = "JOB_DELETE"
    JOB_RUN = "JOB_RUN"
    
    # A/B Testing
    ABTEST_CREATE = "ABTEST_CREATE"
    ABTEST_START = "ABTEST_START"
    ABTEST_CONCLUDE = "ABTEST_CONCLUDE"
    
    # Alerts
    ALERT_ACKNOWLEDGE = "ALERT_ACKNOWLEDGE"
    ALERT_RESOLVE = "ALERT_RESOLVE"
    
    # Admin
    USER_CREATE = "USER_CREATE"
    USER_UPDATE = "USER_UPDATE"
    USER_DELETE = "USER_DELETE"
    ROLE_ASSIGN = "ROLE_ASSIGN"
    SETTINGS_UPDATE = "SETTINGS_UPDATE"


class AuditSeverity(str, Enum):
    """Audit log severity."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class AuditLog:
    """Audit log entry."""
    id: str
    timestamp: datetime
    user_id: str
    user_email: str
    action: AuditAction
    severity: AuditSeverity
    resource_type: str
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    success: bool
    error_message: Optional[str] = None


class AuditService:
    """
    Audit logging service.
    
    Tracks all significant user actions for security and compliance.
    """
    
    def __init__(self):
        self._logs: List[AuditLog] = []
    
    def log(
        self,
        user_id: str,
        user_email: str,
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict] = None,
        ip_address: str = "0.0.0.0",
        user_agent: str = "",
        success: bool = True,
        error_message: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
    ) -> AuditLog:
        """Log an audit event."""
        entry = AuditLog(
            id=str(uuid4()),
            timestamp=now_ist(),
            user_id=user_id,
            user_email=user_email,
            action=action,
            severity=severity,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
        )
        
        self._logs.append(entry)
        
        # Log to structured logger
        log_data = {
            "audit_id": entry.id,
            "user_id": user_id,
            "user_email": user_email,
            "action": action.value,
            "resource": f"{resource_type}/{resource_id}",
            "success": success,
        }
        
        if success:
            logger.info(f"AUDIT: {action.value}", extra=log_data)
        else:
            logger.warning(f"AUDIT FAILED: {action.value} - {error_message}", extra=log_data)
        
        return entry
    
    def query(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        success_only: Optional[bool] = None,
        limit: int = 100,
    ) -> List[AuditLog]:
        """Query audit logs with filters."""
        results = self._logs
        
        if user_id:
            results = [r for r in results if r.user_id == user_id]
        
        if action:
            results = [r for r in results if r.action == action]
        
        if resource_type:
            results = [r for r in results if r.resource_type == resource_type]
        
        if start_date:
            results = [r for r in results if r.timestamp >= start_date]
        
        if end_date:
            results = [r for r in results if r.timestamp <= end_date]
        
        if success_only is not None:
            results = [r for r in results if r.success == success_only]
        
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[:limit]
    
    def get_user_activity(self, user_id: str, limit: int = 50) -> List[AuditLog]:
        """Get recent activity for a user."""
        return self.query(user_id=user_id, limit=limit)
    
    def get_security_events(self, limit: int = 100) -> List[AuditLog]:
        """Get security-related events."""
        security_actions = [
            AuditAction.LOGIN,
            AuditAction.LOGOUT,
            AuditAction.LOGIN_FAILED,
            AuditAction.USER_CREATE,
            AuditAction.USER_DELETE,
            AuditAction.ROLE_ASSIGN,
            AuditAction.SETTINGS_UPDATE,
        ]
        
        results = [r for r in self._logs if r.action in security_actions]
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[:limit]
    
    def export_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json",
    ) -> str:
        """Export logs for compliance."""
        logs = self.query(start_date=start_date, end_date=end_date, limit=10000)
        
        if format == "json":
            return json.dumps([
                {
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat(),
                    "user_id": log.user_id,
                    "user_email": log.user_email,
                    "action": log.action.value,
                    "resource_type": log.resource_type,
                    "resource_id": log.resource_id,
                    "success": log.success,
                    "ip_address": log.ip_address,
                }
                for log in logs
            ], indent=2)
        
        return ""


# Singleton service
_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Get the global audit service instance."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service
